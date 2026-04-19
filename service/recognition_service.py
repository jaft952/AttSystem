import base64
import json
import shutil
import threading
import time
import tempfile
import mediapipe as mp
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import faiss  # type: ignore

    _FAISS_AVAILABLE = True
except Exception:
    faiss = None  # type: ignore[assignment]
    _FAISS_AVAILABLE = False

from service.camera_service import CameraService

# Reuse a single CLAHE instance across all inference calls to avoid
# re-allocating the tile grid every frame.
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# This module lives in AttSystem/ml, so project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = PROJECT_ROOT / "models"
CONFIG_ROOT = PROJECT_ROOT / "config"

RUNTIME_CONFIG_PATH = CONFIG_ROOT / "realtime_model_config.json"
SSD_PROTOTXT_PATH = MODELS_ROOT / "ssd" / "deploy.prototxt"
SSD_MODEL_PATH = MODELS_ROOT / "ssd" / "res10.caffemodel"
MODEL_LOCK = threading.Lock()
MODEL_SWITCH_LOCK = threading.Lock()
CBIR_MODEL_TYPES = {"cbir_method1", "cbir_method2", "cbir_method3"}
DEFAULT_MODEL_TYPE = "cbir_method1"

DEFAULT_ACCEPTANCE_THRESHOLD = 160.0
DEFAULT_STRICT_UNKNOWN_THRESHOLD = 100.0
MAX_LBPH_MODEL_BYTES = 300 * 1024 * 1024
DEFAULT_CBIR_STRICT_UNKNOWN_SIMILARITY = 0.72
DEFAULT_CBIR_MIN_MARGIN = 0.035


def _format_size_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def resolve_strict_unknown_threshold(runtime_config: dict[str, Any], acceptance_threshold: float) -> float:
    raw_value = runtime_config.get("identity_strict_unknown_threshold")
    if raw_value is None:
        strict_threshold = min(acceptance_threshold, DEFAULT_STRICT_UNKNOWN_THRESHOLD)
    else:
        try:
            strict_threshold = float(raw_value)
        except (TypeError, ValueError):
            strict_threshold = min(acceptance_threshold, DEFAULT_STRICT_UNKNOWN_THRESHOLD)

    if strict_threshold <= 0:
        strict_threshold = min(acceptance_threshold, DEFAULT_STRICT_UNKNOWN_THRESHOLD)

    return min(strict_threshold, acceptance_threshold, DEFAULT_STRICT_UNKNOWN_THRESHOLD)


def resolve_cbir_strict_unknown_similarity(cbir_config: dict[str, Any], acceptance_threshold: float) -> float:
    raw_value = cbir_config.get("similarity_strict_unknown_threshold")
    if raw_value is None:
        strict_threshold = max(acceptance_threshold, DEFAULT_CBIR_STRICT_UNKNOWN_SIMILARITY)
    else:
        try:
            strict_threshold = float(raw_value)
        except (TypeError, ValueError):
            strict_threshold = max(acceptance_threshold, DEFAULT_CBIR_STRICT_UNKNOWN_SIMILARITY)

    if strict_threshold <= 0:
        strict_threshold = max(acceptance_threshold, DEFAULT_CBIR_STRICT_UNKNOWN_SIMILARITY)

    return float(min(max(strict_threshold, acceptance_threshold), 0.99))


def resolve_cbir_min_margin(cbir_config: dict[str, Any]) -> float:
    raw_value = cbir_config.get("similarity_min_margin")
    if raw_value is None:
        return DEFAULT_CBIR_MIN_MARGIN

    try:
        margin = float(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_CBIR_MIN_MARGIN

    if margin < 0:
        return 0.0
    if margin > 0.4:
        return 0.4
    return margin


def create_lbph_recognizer() -> Any:
    face_module = getattr(cv2, "face", None)
    factory = getattr(face_module, "LBPHFaceRecognizer_create", None)
    if factory is None:
        raise RuntimeError(
            "LBPHFaceRecognizer is unavailable. Install opencv-contrib-python in this environment."
        )
    return factory()


def load_label_map(path: Path) -> dict[int, str]:
    mapping: dict[int, str] = {}
    if not path.exists():
        raise FileNotFoundError(f"Label map not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            idx, name = line.split(",", 1)
            mapping[int(idx)] = name
    return mapping


def _get_model_config(config: dict[str, Any], model_type: str) -> dict[str, Any]:
    models_block = config.get("models")
    if isinstance(models_block, dict) and isinstance(models_block.get(model_type), dict):
        return models_block.get(model_type, {})
    legacy_block = config.get(model_type)
    if isinstance(legacy_block, dict):
        return legacy_block
    return {}


def _resolve_config_path(path_value: Any) -> Path:
    raw = str(path_value or "").strip()
    if not raw:
        return Path("")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def load_cbir_model(config: dict[str, Any], model_type: str) -> dict[str, Any]:
    cbir_config = _get_model_config(config, model_type)
    index_path = _resolve_config_path(cbir_config.get("index_path", ""))
    meta_path = _resolve_config_path(cbir_config.get("meta_path", ""))

    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"CBIR index or meta not found: {index_path}, {meta_path}")

    index_data = np.load(index_path, allow_pickle=True)
    embeddings = np.ascontiguousarray(index_data["embeddings"].astype(np.float32))
    labels = np.asarray(index_data["labels"], dtype=np.int32)

    # Normalize once at load-time so both NumPy and FAISS cosine/IP search
    # paths can reuse the same unit-length vectors.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    embeddings = embeddings / norms

    faiss_index = None
    faiss_index_path = index_path.with_suffix(".faiss")
    if _FAISS_AVAILABLE and embeddings.size > 0 and faiss is not None:
        configured_faiss_path = _resolve_config_path(cbir_config.get("faiss_index_path", ""))
        if str(configured_faiss_path).strip() != "":
            faiss_index_path = configured_faiss_path

        try:
            if faiss_index_path.exists():
                candidate = faiss.read_index(str(faiss_index_path))
                if int(getattr(candidate, "d", -1)) == int(embeddings.shape[1]):
                    faiss_index = candidate
        except Exception:
            faiss_index = None

        if faiss_index is None:
            candidate = faiss.IndexFlatIP(int(embeddings.shape[1]))
            candidate.add(embeddings)  # type: ignore[call-arg]
            faiss_index = candidate
            # Persist auto-built FAISS so subsequent startups skip rebuilding.
            try:
                faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
                faiss.write_index(candidate, str(faiss_index_path))
            except Exception:
                pass

    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    label_names = meta.get("label_names", [])
    label_map = {i: name for i, name in enumerate(label_names)}

    acceptance_threshold = float(cbir_config.get("similarity_threshold", 0.6))

    return {
        "embeddings": embeddings,
        "labels": labels,
        "labels_array": labels,
        "label_map": label_map,
        "label_names": label_names,
        "faiss_index": faiss_index,
        "faiss_enabled": bool(faiss_index is not None),
        "threshold": acceptance_threshold,
        "strict_unknown_threshold": resolve_cbir_strict_unknown_similarity(
            cbir_config, acceptance_threshold
        ),
        "min_margin": resolve_cbir_min_margin(cbir_config),
        "input_size": tuple(cbir_config.get("input_size", [128, 128])),
        "preprocess_mode": str(cbir_config.get("preprocess_mode", "method1")).strip().lower(),
        "display_name": str(cbir_config.get("display_name", model_type)).strip(),
    }


def load_lbph_model(config: dict[str, Any]) -> dict[str, Any]:
    lbph_config = _get_model_config(config, "lbph")
    model_path = Path(lbph_config.get("identity_model", ""))
    label_map_path = Path(lbph_config.get("identity_label_map", ""))
    threshold = float(lbph_config.get("identity_confidence_threshold", DEFAULT_ACCEPTANCE_THRESHOLD))
    strict_unknown_threshold = resolve_strict_unknown_threshold(lbph_config, threshold)
    input_size = tuple(lbph_config.get("input_size", [128, 128]))

    if not model_path.exists():
        raise FileNotFoundError(f"LBPH model not found: {model_path}")

    model_size = model_path.stat().st_size
    if model_size > MAX_LBPH_MODEL_BYTES:
        raise RuntimeError(
            "LBPH model file is too large to load safely "
            f"({_format_size_mb(model_size)}; limit {_format_size_mb(MAX_LBPH_MODEL_BYTES)}). "
            "This usually indicates a bad training output. Retrain and regenerate models/lbph.yml."
        )

    recognizer = create_lbph_recognizer()
    last_error: Exception | None = None
    for attempt in range(3):
        tmp_path: str | None = None
        try:
            # Read from a temp snapshot so concurrent writers do not break OpenCV FileStorage parsing.
            with tempfile.NamedTemporaryFile(suffix=".yml", delete=False) as tmp_file:
                tmp_path = tmp_file.name
            shutil.copy2(model_path, tmp_path)
            recognizer.read(str(tmp_path))
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            time.sleep(0.08 * (attempt + 1))
        finally:
            if tmp_path:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass

    if last_error is not None:
        raise RuntimeError(
            f"Failed to load LBPH model from {model_path}. The file may be temporarily incomplete; retry or retrain the model."
        ) from last_error

    label_map = load_label_map(label_map_path)

    return {
        "model_path": model_path,
        "recognizer": recognizer,
        "label_map": label_map,
        "threshold": threshold,
        "strict_unknown_threshold": strict_unknown_threshold,
        "input_size": input_size,
    }


def _load_model_for_type(config: dict[str, Any], model_type: str) -> dict[str, Any]:
    if model_type in CBIR_MODEL_TYPES:
        return load_cbir_model(config, model_type)
    raise ValueError(f"Unsupported model type '{model_type}'.")


def load_runtime_assets() -> dict[str, Any]:
    if not RUNTIME_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Runtime config not found: {RUNTIME_CONFIG_PATH}. Run the training notebook first."
        )

    with RUNTIME_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        runtime_config = json.load(handle)

    requested_model_type = str(runtime_config.get("model_type", DEFAULT_MODEL_TYPE)).strip().lower()
    if requested_model_type not in CBIR_MODEL_TYPES:
        requested_model_type = DEFAULT_MODEL_TYPE

    model_data: dict[str, Any]
    loaded_model_type: str
    try:
        model_data = load_cbir_model(runtime_config, requested_model_type)
        loaded_model_type = requested_model_type
    except Exception:
        fallback_type = "cbir_method2" if requested_model_type == "cbir_method1" else "cbir_method1"
        model_data = load_cbir_model(runtime_config, fallback_type)
        loaded_model_type = fallback_type

    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision as mp_vision
        FaceDetector = mp_vision.FaceDetector
        FaceDetectorOptions = mp_vision.FaceDetectorOptions
        BaseOptions = mp.tasks.BaseOptions
        
        model_path = MODELS_ROOT / "blaze_face_short_range.tflite"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing MediaPipe model: {model_path}")
            
        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            min_detection_confidence=0.5
        )
        face_detector = FaceDetector.create_from_options(options)
    except ImportError:
        raise RuntimeError("MediaPipe is missing. Please install it using 'pip install mediapipe'.")

    return {
        "runtime_config": runtime_config,
        "model_type": loaded_model_type,
        "model_data": model_data,
        "input_size": model_data["input_size"],
        "face_detector": face_detector,
    }


ASSETS = load_runtime_assets()


def _reload_runtime_assets_from_disk() -> None:
    fresh_assets = load_runtime_assets()
    ASSETS.update(fresh_assets)


def get_label_names() -> list[str]:
    label_map = ASSETS.get("model_data", {}).get("label_map", {})
    return [name for _, name in sorted(label_map.items(), key=lambda item: item[0])]


def get_developer_tools_template_context(app_name: str) -> dict[str, Any]:
    model_type = ASSETS.get("model_type", DEFAULT_MODEL_TYPE)
    model_data = ASSETS.get("model_data", {})
    model_name = model_data.get("display_name") or model_type

    return {
        "app_name": app_name,
        "threshold": model_data.get("threshold", 0.6),
        "input_size": model_data.get("input_size", [128, 128]),
        "model_name": model_name,
        "model_type": model_type,
        "label_names": get_label_names(),
    }


def get_health_payload() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_type": ASSETS.get("model_type", DEFAULT_MODEL_TYPE),
    }


def get_latest_payload() -> dict[str, Any]:
    CAMERA_SERVICE.keep_alive()
    payload = CAMERA_SERVICE.get_latest()
    payload["model_type"] = ASSETS.get("model_type", DEFAULT_MODEL_TYPE)
    return payload


def stream_predictions():
    """Yield prediction dicts as the inference loop produces them.

    Intended for the SSE endpoint — each yielded dict is pushed to the
    browser immediately rather than waiting for the next REST poll cycle.
    Yields a keepalive sentinel (``None``) on timeout so the caller can
    emit an SSE comment to prevent proxy / browser connection drops.
    """
    last_seq = -1
    while True:
        CAMERA_SERVICE.keep_alive()
        if not CAMERA_SERVICE.running:
            time.sleep(0.1)
            yield None  # keepalive while camera is off
            continue
        prediction, seq = CAMERA_SERVICE.wait_for_next_prediction(last_seq, timeout=1.0)
        if seq != last_seq:
            last_seq = seq
            yield prediction
        else:
            yield None  # keepalive on timeout


def stream_frames():
    last_sent_seq = -1
    while True:
        CAMERA_SERVICE.keep_alive()
        if not CAMERA_SERVICE.running:
            time.sleep(0.05)
            continue

        frame, seq = CAMERA_SERVICE.wait_for_next_frame(last_sent_seq, timeout=0.1)
        if frame is None:
            time.sleep(0.05)
            continue

        if seq == last_sent_seq:
            time.sleep(0.01)
            continue

        last_sent_seq = seq

        yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.001)


def pick_largest_face(faces: np.ndarray | None):
    if faces is None or len(faces) == 0:
        return None
    return max(faces, key=lambda box: box[2] * box[3])


def preprocess_face(
    rgb_frame: np.ndarray,
    face_box,
    input_size=(128, 128),
    padding=0.20,
    preprocess_mode: str = "method1",
):
    x, y, w, h = face_box

    # Face Alignment: align rotation based on eye landmarks
    css_location = [(y, x + w, y + h, x)]
    try:
        import face_recognition
        # Use "small" model for 5-point landmarks (EXTREMELY FAST compared to default 68-point)
        landmarks = face_recognition.face_landmarks(rgb_frame, css_location, model="small")
        if landmarks and "left_eye" in landmarks[0] and "right_eye" in landmarks[0]:
            left_eye_points = landmarks[0]["left_eye"]
            right_eye_points = landmarks[0]["right_eye"]
            
            left_center = np.mean(left_eye_points, axis=0).astype(int)
            right_center = np.mean(right_eye_points, axis=0).astype(int)
            
            dy = right_center[1] - left_center[1]
            dx = right_center[0] - left_center[0]
            angle = np.degrees(np.arctan2(dy, dx))
            
            eyes_center = (
                int((left_center[0] + right_center[0]) / 2.0),
                int((left_center[1] + right_center[1]) / 2.0)
            )
            
            M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
            rgb_frame = cv2.warpAffine(rgb_frame, M, (rgb_frame.shape[1], rgb_frame.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        pass # Skip alignment if libraries fail

    pad_x = int(w * padding)
    pad_y = int(h * padding)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(rgb_frame.shape[1], x + w + pad_x)
    y2 = min(rgb_frame.shape[0], y + h + pad_y)

    roi = rgb_frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    mode = (preprocess_mode or "method1").lower()
    if mode == "method2":
        denoised = cv2.bilateralFilter(roi, d=5, sigmaColor=50, sigmaSpace=50)
        lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
        lc, ac, bc = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lc = clahe.apply(lc)
        lab_enhanced = cv2.merge([lc, ac, bc])
        roi = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    elif mode == "method3":
        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        roi = cv2.equalizeHist(roi)
        roi = cv2.bilateralFilter(roi, d=7, sigmaColor=50, sigmaSpace=50)
        blurred = cv2.GaussianBlur(roi, (0, 0), 1.2)
        roi = cv2.addWeighted(roi, 1.35, blurred, -0.35, 0)
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
    else:
        pass

    roi = cv2.resize(roi, input_size, interpolation=cv2.INTER_CUBIC)
    return roi


def _predict_lbph(face_roi: np.ndarray, model_data: dict[str, Any]) -> dict[str, Any]:
    recognizer = model_data["recognizer"]
    label_map = model_data["label_map"]
    threshold = float(model_data["threshold"])
    strict_unknown_threshold = float(model_data.get("strict_unknown_threshold", threshold))

    label_id, confidence = recognizer.predict(face_roi)
    raw_name = label_map.get(int(label_id), "unknown")
    accepted = float(confidence) <= strict_unknown_threshold
    display_name = raw_name if accepted else "unknown"

    return {
        "label_id": int(label_id),
        "raw_name": raw_name,
        "name": display_name,
        "confidence": float(confidence),
        "accepted": bool(accepted),
        "acceptance_threshold": float(threshold),
        "strict_unknown_threshold": float(strict_unknown_threshold),
    }


def _predict_cbir(face_roi: np.ndarray, model_data: dict[str, Any]) -> dict[str, Any]:
    try:
        import face_recognition
    except ImportError:
        raise RuntimeError("face_recognition library required for CBIR. Install it with: pip install face-recognition")

    embeddings = model_data["embeddings"]
    labels = model_data["labels"]
    labels_array = model_data.get("labels_array", np.asarray(labels, dtype=np.int32))
    label_map = model_data["label_map"]
    faiss_index = model_data.get("faiss_index")
    faiss_enabled = bool(model_data.get("faiss_enabled", False) and faiss_index is not None)
    threshold = float(model_data["threshold"])
    strict_unknown_threshold = float(model_data.get("strict_unknown_threshold", threshold))
    min_margin = float(model_data.get("min_margin", DEFAULT_CBIR_MIN_MARGIN))

    face_rgb = face_roi.copy() if len(face_roi.shape) == 3 else cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
    h_roi, w_roi = face_rgb.shape[:2]
    # Pass the full ROI as the known face location so face_recognition skips
    # its internal HOG/CNN face detector (we already detected the face with
    # the Haar cascade).  Location order is (top, right, bottom, left).
    known_locations = [(0, w_roi, h_roi, 0)]
    encodings = face_recognition.face_encodings(face_rgb, known_face_locations=known_locations)
    if len(encodings) == 0:
        return {
            "label_id": -1,
            "raw_name": "unknown",
            "name": "unknown",
            "confidence": 1.0,
            "accepted": False,
            "acceptance_threshold": threshold,
            "strict_unknown_threshold": threshold,
            "message": "Could not extract face encoding.",
        }

    query_encoding = np.array(encodings[0], dtype=np.float32)
    norm = np.linalg.norm(query_encoding)
    if norm > 1e-12:
        query_encoding = query_encoding / norm

    if faiss_enabled:
        # Query a shortlist instead of scanning all vectors in Python.
        k = int(min(len(labels_array), 16))
        q = np.ascontiguousarray(query_encoding.reshape(1, -1), dtype=np.float32)
        similarities, indices = faiss_index.search(q, k)  # type: ignore[union-attr]

        valid_positions = [
            int(pos)
            for pos in indices[0].tolist()
            if isinstance(pos, (int, np.integer)) and int(pos) >= 0
        ]
        if len(valid_positions) == 0:
            return {
                "label_id": -1,
                "raw_name": "unknown",
                "name": "unknown",
                "confidence": 0.0,
                "accepted": False,
                "acceptance_threshold": threshold,
                "strict_unknown_threshold": strict_unknown_threshold,
                "message": "No nearest neighbor candidates found.",
            }

        closest_idx = valid_positions[0]
        closest_label_id = int(labels_array[closest_idx])
        raw_name = label_map.get(closest_label_id, "unknown")
        similarity = float(similarities[0][0])

        second_similarity = 0.0
        for rank, idx in enumerate(valid_positions[1:], start=1):
            if int(labels_array[idx]) != closest_label_id:
                second_similarity = float(similarities[0][rank])
                break
    else:
        # Fallback path when FAISS is unavailable.
        distances = 1.0 - (embeddings @ query_encoding)
        closest_idx = int(np.argmin(distances))
        closest_distance = float(distances[closest_idx])
        closest_label_id = int(labels_array[closest_idx])
        raw_name = label_map.get(closest_label_id, "unknown")

        different_identity_mask = labels_array != closest_label_id
        if np.any(different_identity_mask):
            second_distance = float(np.min(distances[different_identity_mask]))
        else:
            second_distance = 1.0

        similarity = 1.0 - closest_distance
        second_similarity = 1.0 - second_distance

    similarity = float(max(-1.0, min(1.0, similarity)))
    second_similarity = float(max(-1.0, min(1.0, second_similarity)))
    similarity_margin = max(0.0, similarity - second_similarity)

    accepted = (
        similarity >= threshold
        and similarity >= strict_unknown_threshold
        and similarity_margin >= min_margin
    )
    display_name = raw_name if accepted else "unknown"

    return {
        "label_id": closest_label_id,
        "raw_name": raw_name,
        "name": display_name,
        "confidence": float(similarity),
        "accepted": bool(accepted),
        "acceptance_threshold": threshold,
        "strict_unknown_threshold": strict_unknown_threshold,
        "second_best_confidence": float(second_similarity),
        "similarity_margin": float(similarity_margin),
        "required_margin": float(min_margin),
    }


def predict_face(face_roi: np.ndarray) -> dict[str, Any]:
    with MODEL_LOCK:
        model_data = ASSETS["model_data"]
        return _predict_cbir(face_roi, model_data)


def process_camera_frame(frame: np.ndarray):
    face_detector = ASSETS.get("face_detector")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = face_detector.detect(mp_image)
    
    faces = []
    if results.detections:
        for detection in results.detections:
            bbox = detection.bounding_box
            faces.append((max(0, bbox.origin_x), max(0, bbox.origin_y), bbox.width, bbox.height))
            
    face_box = pick_largest_face(faces)
    if face_box is None:
        prediction = {
            "status": "no_face",
            "name": "No face",
            "raw_name": "unknown",
            "confidence": None,
            "accepted": False,
            "bbox": None,
            "message": "No face detected.",
        }
        return None, prediction, None

    model_data = ASSETS.get("model_data", {})
    face_roi = preprocess_face(
        rgb,
        face_box,
        input_size=ASSETS.get("input_size", (128, 128)),
        padding=0.2,
        preprocess_mode=str(model_data.get("preprocess_mode", "method1")),
    )
    if face_roi is None:
        prediction = {
            "status": "invalid_roi",
            "name": "Invalid ROI",
            "raw_name": "unknown",
            "confidence": None,
            "accepted": False,
            "bbox": None,
            "message": "Face ROI could not be prepared.",
        }
        return None, prediction, None

    prediction = predict_face(face_roi)
    x, y, w, h = map(int, face_box)
    prediction.update(
        {
            "status": "ok",
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "message": "Prediction updated.",
        }
    )
    return None, prediction, face_roi

def decode_image_data(image_data: str) -> np.ndarray:
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    binary = base64.b64decode(image_data)
    buffer = np.frombuffer(binary, dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode the uploaded frame.")
    return frame


def predict_from_payload(image_data: str) -> dict[str, Any]:
    frame = decode_image_data(image_data)
    frame = cv2.flip(frame, 1)
    
    face_detector = ASSETS.get("face_detector")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = face_detector.detect(mp_image)
    
    faces = []
    if results.detections:
        for detection in results.detections:
            bbox = detection.bounding_box
            faces.append((max(0, bbox.origin_x), max(0, bbox.origin_y), bbox.width, bbox.height))
            
    face_box = pick_largest_face(faces)
    if face_box is None:
        return {
            "status": "no_face",
            "message": "No face detected.",
        }

    model_data = ASSETS.get("model_data", {})
    face_roi = preprocess_face(
        rgb,
        face_box,
        input_size=ASSETS.get("input_size", (128, 128)),
        padding=0.20,
        preprocess_mode=str(model_data.get("preprocess_mode", "method1")),
    )
    if face_roi is None:
        return {
            "status": "invalid_roi",
            "message": "Face ROI could not be prepared.",
        }

    prediction = predict_face(face_roi)
    x, y, w, h = map(int, face_box)
    return {
        "status": "ok",
        "prediction": prediction,
        "bbox": {"x": x, "y": y, "w": w, "h": h},
        "threshold": model_data.get("threshold", 0.6),
        "model_type": ASSETS.get("model_type", DEFAULT_MODEL_TYPE),
    }


def switch_model(new_model_type: str) -> dict[str, Any]:
    if new_model_type not in CBIR_MODEL_TYPES:
        raise ValueError("Model type must be 'cbir_method1' or 'cbir_method2'.")

    with MODEL_SWITCH_LOCK:
        if ASSETS.get("model_type") == new_model_type:
            model_data = ASSETS.get("model_data", {})
            return {
                "status": "ok",
                "message": f"Model already set to {new_model_type}.",
                "current_model": new_model_type,
                "label_names": get_label_names(),
                "threshold": model_data.get("threshold"),
                "input_size": model_data.get("input_size"),
            }

        runtime_config = dict(ASSETS["runtime_config"])
        runtime_config["model_type"] = new_model_type

        target_model_config = _get_model_config(runtime_config, new_model_type)
        if not target_model_config:
            raise ValueError(f"Model configuration for '{new_model_type}' not found.")

        # Load once outside model-state lock, then swap assets atomically.
        model_data = _load_model_for_type(runtime_config, new_model_type)

        with RUNTIME_CONFIG_PATH.open("w", encoding="utf-8") as handle:
            json.dump(runtime_config, handle, indent=2)

        with MODEL_LOCK:
            ASSETS["runtime_config"] = runtime_config
            ASSETS["model_type"] = new_model_type
            ASSETS["model_data"] = model_data
            ASSETS["input_size"] = model_data["input_size"]

        if CAMERA_SERVICE.running:
            CAMERA_SERVICE.restart_inference_worker()

        return {
            "status": "ok",
            "message": f"Switched to {new_model_type} model.",
            "current_model": new_model_type,
            "label_names": get_label_names(),
            "threshold": model_data.get("threshold"),
            "input_size": model_data.get("input_size"),
        }


def start_camera() -> dict[str, Any]:
    if not CAMERA_SERVICE.running:
        CAMERA_SERVICE.start()
    return {"status": "ok", "message": "Camera started"}


def stop_camera() -> dict[str, Any]:
    if CAMERA_SERVICE.running:
        CAMERA_SERVICE.stop()
    return {"status": "ok", "message": "Camera stopped"}


CAMERA_SERVICE = CameraService(
    frame_processor=process_camera_frame,
    inference_interval_sec=0.12,
    jpeg_quality=65,
    stream_fps=15,
) # type: ignore
