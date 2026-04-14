import base64
import json
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import scipy.spatial.distance as sp_distance

from service.camera_service import CameraService

# This module lives in AttSystem/ml, so project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = PROJECT_ROOT / "models"

RUNTIME_CONFIG_PATH = MODELS_ROOT / "realtime_model_config.json"
HAAR_CASCADE_PATH = (
    Path(cv2.__file__).resolve().parent / "data" / "haarcascade_frontalface_default.xml"
)
MODEL_LOCK = threading.Lock()

DEFAULT_ACCEPTANCE_THRESHOLD = 160.0
DEFAULT_STRICT_UNKNOWN_THRESHOLD = 100.0


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


def load_cbir_model(config: dict[str, Any]) -> dict[str, Any]:
    cbir_config = _get_model_config(config, "cbir")
    index_path = Path(cbir_config.get("index_path", ""))
    meta_path = Path(cbir_config.get("meta_path", ""))

    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"CBIR index or meta not found: {index_path}, {meta_path}")

    index_data = np.load(index_path, allow_pickle=True)
    embeddings = index_data["embeddings"].astype(np.float32)
    labels = index_data["labels"]

    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    label_names = meta.get("label_names", [])
    label_map = {i: name for i, name in enumerate(label_names)}

    return {
        "embeddings": embeddings,
        "labels": labels,
        "label_map": label_map,
        "label_names": label_names,
        "threshold": float(cbir_config.get("similarity_threshold", 0.6)),
        "input_size": tuple(cbir_config.get("input_size", [128, 128])),
    }


def load_lbph_model(config: dict[str, Any]) -> dict[str, Any]:
    lbph_config = _get_model_config(config, "lbph")
    model_path = Path(lbph_config.get("identity_model", ""))
    label_map_path = Path(lbph_config.get("identity_label_map", ""))
    threshold = float(lbph_config.get("identity_confidence_threshold", DEFAULT_ACCEPTANCE_THRESHOLD))
    strict_unknown_threshold = resolve_strict_unknown_threshold(lbph_config, threshold)
    input_size = tuple(lbph_config.get("input_size", [128, 128]))

    recognizer = create_lbph_recognizer()
    recognizer.read(str(model_path))
    label_map = load_label_map(label_map_path)

    return {
        "model_path": model_path,
        "recognizer": recognizer,
        "label_map": label_map,
        "threshold": threshold,
        "strict_unknown_threshold": strict_unknown_threshold,
        "input_size": input_size,
    }


def load_runtime_assets() -> dict[str, Any]:
    if not RUNTIME_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Runtime config not found: {RUNTIME_CONFIG_PATH}. Run the training notebook first."
        )

    with RUNTIME_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        runtime_config = json.load(handle)

    requested_model_type = str(runtime_config.get("model_type", "lbph")).strip().lower()
    if requested_model_type not in {"lbph", "cbir"}:
        requested_model_type = "lbph"

    model_data: dict[str, Any]
    loaded_model_type: str
    try:
        if requested_model_type == "cbir":
            model_data = load_cbir_model(runtime_config)
            loaded_model_type = "cbir"
        else:
            model_data = load_lbph_model(runtime_config)
            loaded_model_type = "lbph"
    except Exception:
        if requested_model_type == "cbir":
            model_data = load_lbph_model(runtime_config)
            loaded_model_type = "lbph"
        else:
            model_data = load_cbir_model(runtime_config)
            loaded_model_type = "cbir"

    face_cascade = cv2.CascadeClassifier(str(HAAR_CASCADE_PATH))
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade: {HAAR_CASCADE_PATH}")

    return {
        "runtime_config": runtime_config,
        "model_type": loaded_model_type,
        "model_data": model_data,
        "input_size": model_data["input_size"],
        "face_cascade": face_cascade,
    }


ASSETS = load_runtime_assets()


def _reload_runtime_assets_from_disk() -> None:
    fresh_assets = load_runtime_assets()
    ASSETS.update(fresh_assets)


def get_label_names() -> list[str]:
    label_map = ASSETS.get("model_data", {}).get("label_map", {})
    return [name for _, name in sorted(label_map.items(), key=lambda item: item[0])]


def get_developer_tools_template_context(app_name: str) -> dict[str, Any]:
    model_type = ASSETS.get("model_type", "lbph")
    model_data = ASSETS.get("model_data", {})

    if model_type == "cbir":
        model_name = "CBIR"
    else:
        lbph_config = _get_model_config(ASSETS.get("runtime_config", {}), "lbph")
        model_path = Path(lbph_config.get("identity_model", "lbph.yml"))
        model_name = model_path.name

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
        "model_type": ASSETS.get("model_type", "lbph"),
    }


def get_latest_payload() -> dict[str, Any]:
    payload = CAMERA_SERVICE.get_latest()
    payload["model_type"] = ASSETS.get("model_type", "lbph")
    return payload


def stream_frames():
    while True:
        if not CAMERA_SERVICE.running:
            time.sleep(0.05)
            continue

        frame = CAMERA_SERVICE.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.005)


def pick_largest_face(faces: np.ndarray | None):
    if faces is None or len(faces) == 0:
        return None
    return max(faces, key=lambda box: box[2] * box[3])


def preprocess_face(gray_frame: np.ndarray, face_box, input_size=(128, 128), padding=0.20):
    x, y, w, h = face_box
    pad_x = int(w * padding)
    pad_y = int(h * padding)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(gray_frame.shape[1], x + w + pad_x)
    y2 = min(gray_frame.shape[0], y + h + pad_y)

    roi = gray_frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    roi = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi)
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
    label_map = model_data["label_map"]
    threshold = float(model_data["threshold"])

    face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
    encodings = face_recognition.face_encodings(face_rgb)
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

    distances = sp_distance.cdist([query_encoding], embeddings, metric="cosine")[0]
    closest_idx = int(np.argmin(distances))
    closest_distance = float(distances[closest_idx])
    closest_label_id = int(labels[closest_idx])
    raw_name = label_map.get(closest_label_id, "unknown")

    similarity = 1.0 - closest_distance
    accepted = similarity >= threshold
    display_name = raw_name if accepted else "unknown"

    return {
        "label_id": closest_label_id,
        "raw_name": raw_name,
        "name": display_name,
        "confidence": float(similarity),
        "accepted": bool(accepted),
        "acceptance_threshold": threshold,
        "strict_unknown_threshold": threshold,
    }


def predict_face(face_roi: np.ndarray) -> dict[str, Any]:
    with MODEL_LOCK:
        model_type = ASSETS["model_type"]
        model_data = ASSETS["model_data"]

        if model_type == "cbir":
            return _predict_cbir(face_roi, model_data)
        return _predict_lbph(face_roi, model_data)


def process_camera_frame(frame: np.ndarray):
    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detect_scale = 0.6
    small_gray = cv2.resize(
        gray,
        (0, 0),
        fx=detect_scale,
        fy=detect_scale,
        interpolation=cv2.INTER_AREA,
    )
    faces = ASSETS["face_cascade"].detectMultiScale(
        small_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    if faces is not None and len(faces) > 0:
        inv_scale = 1.0 / detect_scale
        faces = np.array(
            [
                [
                    int(x * inv_scale),
                    int(y * inv_scale),
                    int(w * inv_scale),
                    int(h * inv_scale),
                ]
                for (x, y, w, h) in faces
            ],
            dtype=np.int32,
        )

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
        return display, prediction, None

    face_roi = preprocess_face(gray, face_box, input_size=ASSETS["input_size"], padding=0.20)
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
        return display, prediction, None

    prediction = predict_face(face_roi)
    x, y, w, h = map(int, face_box)
    cv2.rectangle(display, (x, y), (x + w, y + h), (255, 255, 255), 2)
    prediction.update(
        {
            "status": "ok",
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "message": "Prediction updated.",
        }
    )
    return display, prediction, None


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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = ASSETS["face_cascade"].detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
    )

    face_box = pick_largest_face(faces)
    if face_box is None:
        return {
            "status": "no_face",
            "message": "No face detected.",
        }

    model_data = ASSETS.get("model_data", {})
    face_roi = preprocess_face(gray, face_box, input_size=model_data.get("input_size", ASSETS["input_size"]), padding=0.20)
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
        "model_type": ASSETS.get("model_type", "lbph"),
    }


def switch_model(new_model_type: str) -> dict[str, Any]:
    if new_model_type not in {"lbph", "cbir"}:
        raise ValueError("Model type must be 'lbph' or 'cbir'.")

    runtime_config = dict(ASSETS["runtime_config"])
    runtime_config["model_type"] = new_model_type

    target_model_config = _get_model_config(runtime_config, new_model_type)
    if not target_model_config:
        raise ValueError(f"Model configuration for '{new_model_type}' not found.")

    with RUNTIME_CONFIG_PATH.open("w", encoding="utf-8") as handle:
        json.dump(runtime_config, handle, indent=2)

    _reload_runtime_assets_from_disk()

    return {
        "status": "ok",
        "message": f"Switched to {new_model_type} model.",
        "current_model": new_model_type,
        "label_names": get_label_names(),
    }


def start_camera() -> dict[str, Any]:
    if not CAMERA_SERVICE.running:
        CAMERA_SERVICE.start()
    return {"status": "ok", "message": "Camera started"}


def stop_camera() -> dict[str, Any]:
    if CAMERA_SERVICE.running:
        CAMERA_SERVICE.stop()
    return {"status": "ok", "message": "Camera stopped"}


CAMERA_SERVICE = CameraService(frame_processor=process_camera_frame)
