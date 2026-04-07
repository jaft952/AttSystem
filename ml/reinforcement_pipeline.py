import base64
import gc
import json
import multiprocessing as mp
import threading
import time
from datetime import datetime
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any
import cv2
import numpy as np

from service.camera_service import CameraService

# This module lives in AttSystem/ml, so project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = PROJECT_ROOT / "models"

RUNTIME_CONFIG_PATH = MODELS_ROOT / "realtime_model_config.json"
HAAR_CASCADE_PATH = (
    Path(cv2.__file__).resolve().parent / "data" / "haarcascade_frontalface_default.xml"
)
FEEDBACK_ROOT = PROJECT_ROOT / "data" / "8_feedback"
RETRAIN_RESULT_PATH = MODELS_ROOT / ".retrain_result.json"
MODEL_LOCK = threading.Lock()
CLIENT_SAMPLE_LOCK = threading.Lock()

FEEDBACK_ROOT.mkdir(parents=True, exist_ok=True)

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

    # Strict threshold must never be looser than acceptance and never exceed
    # the hard unknown gate, even if config is set too high.
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


def load_runtime_assets() -> dict[str, Any]:
    if not RUNTIME_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Runtime config not found: {RUNTIME_CONFIG_PATH}. Run the training notebook first."
        )

    with RUNTIME_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        runtime_config = json.load(handle)

    model_path = Path(runtime_config["identity_model"])
    label_map_path = Path(runtime_config["identity_label_map"])
    threshold = float(runtime_config.get("identity_confidence_threshold", DEFAULT_ACCEPTANCE_THRESHOLD))
    strict_unknown_threshold = resolve_strict_unknown_threshold(runtime_config, threshold)
    input_size = tuple(runtime_config.get("input_size", [128, 128]))

    recognizer = create_lbph_recognizer()
    recognizer.read(str(model_path))
    label_map = load_label_map(label_map_path)
    face_cascade = cv2.CascadeClassifier(str(HAAR_CASCADE_PATH))
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade: {HAAR_CASCADE_PATH}")

    return {
        "runtime_config": runtime_config,
        "model_path": model_path,
        "label_map_path": label_map_path,
        "threshold": threshold,
        "strict_unknown_threshold": strict_unknown_threshold,
        "input_size": input_size,
        "recognizer": recognizer,
        "label_map": label_map,
        "face_cascade": face_cascade,
    }


ASSETS = load_runtime_assets()
CLIENT_FEEDBACK_SAMPLE: dict[str, Any] = {
    "face_roi": None,
    "prediction": None,
    "captured_at": 0.0,
}
RETRAIN_LOCK = threading.Lock()
RETRAIN_PROCESS: BaseProcess | None = None
RETRAIN_STATE: dict[str, Any] = {
    "status": "idle",
    "message": "No retraining job has started yet.",
    "started_at": None,
    "finished_at": None,
    "error": None,
    "result": None,
    "camera_paused": False,
}


def get_label_names() -> list[str]:
    return [name for _, name in sorted(ASSETS["label_map"].items(), key=lambda item: item[0])]


def is_retraining_running() -> bool:
    with RETRAIN_LOCK:
        return RETRAIN_STATE.get("status") == "running"


def ensure_camera_service():
    if is_retraining_running():
        return
    if not CAMERA_SERVICE.running:
        CAMERA_SERVICE.start()


def get_retrain_status() -> dict[str, Any]:
    _finalize_retrain_if_finished()
    with RETRAIN_LOCK:
        return dict(RETRAIN_STATE)


def _set_retrain_state(**updates: Any):
    with RETRAIN_LOCK:
        RETRAIN_STATE.update(updates)


def _reload_runtime_assets_from_disk():
    fresh_assets = load_runtime_assets()
    ASSETS.update(fresh_assets)


def _write_retrain_result(payload: dict[str, Any]):
    RETRAIN_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = RETRAIN_RESULT_PATH.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    temp_path.replace(RETRAIN_RESULT_PATH)


def _retrain_worker_process():
    try:
        result = retrain_runtime_model(include_feedback=True)
        _write_retrain_result(
            {
                "status": "completed",
                "message": "Retraining completed successfully.",
                "error": None,
                "result": result,
                "finished_at": datetime.now().isoformat(),
            }
        )
    except Exception as exc:
        _write_retrain_result(
            {
                "status": "failed",
                "message": "Retraining failed.",
                "error": str(exc),
                "result": None,
                "finished_at": datetime.now().isoformat(),
            }
        )


def _finalize_retrain_if_finished():
    global RETRAIN_PROCESS

    with RETRAIN_LOCK:
        process = RETRAIN_PROCESS
        if process is None:
            return
        if process.is_alive():
            return
        RETRAIN_PROCESS = None
        camera_paused = bool(RETRAIN_STATE.get("camera_paused"))

    exit_code = process.exitcode
    result_payload: dict[str, Any] | None = None
    if RETRAIN_RESULT_PATH.exists():
        try:
            with RETRAIN_RESULT_PATH.open("r", encoding="utf-8") as handle:
                result_payload = json.load(handle)
        except Exception:
            result_payload = None

    if exit_code == 0 and result_payload and result_payload.get("status") == "completed":
        try:
            _reload_runtime_assets_from_disk()
            _set_retrain_state(
                status="completed",
                message=result_payload.get("message") or "Retraining completed successfully.",
                error=None,
                result=result_payload.get("result"),
                finished_at=result_payload.get("finished_at") or datetime.now().isoformat(),
            )
        except Exception as exc:
            _set_retrain_state(
                status="failed",
                message="Retraining finished, but runtime assets could not be reloaded.",
                error=str(exc),
                result=None,
                finished_at=datetime.now().isoformat(),
            )
    else:
        error_message = (
            (result_payload or {}).get("error")
            or f"Retraining worker exited unexpectedly (exit code {exit_code})."
        )
        _set_retrain_state(
            status="failed",
            message=(result_payload or {}).get("message") or "Retraining failed.",
            error=error_message,
            result=(result_payload or {}).get("result"),
            finished_at=(result_payload or {}).get("finished_at") or datetime.now().isoformat(),
        )

    if camera_paused:
        try:
            CAMERA_SERVICE.start()
        except Exception as exc:
            _set_retrain_state(
                message="Retraining finished, but camera failed to restart.",
                error=f"{RETRAIN_STATE.get('error') or ''} | Camera restart error: {exc}".strip(" |"),
            )


def start_retrain_with_feedback_async() -> dict[str, Any]:
    global RETRAIN_PROCESS

    _finalize_retrain_if_finished()

    with RETRAIN_LOCK:
        if RETRAIN_STATE.get("status") == "running":
            return {
                "status": "running",
                "message": "Retraining is already in progress.",
                "started_at": RETRAIN_STATE.get("started_at"),
            }

        resume_camera_after = bool(CAMERA_SERVICE.running)
        if resume_camera_after:
            CAMERA_SERVICE.stop()

        if RETRAIN_RESULT_PATH.exists():
            RETRAIN_RESULT_PATH.unlink(missing_ok=True)

        RETRAIN_STATE.update(
            {
                "status": "running",
                "message": "Retraining started in background.",
                "started_at": datetime.now().isoformat(),
                "finished_at": None,
                "error": None,
                "result": None,
                "camera_paused": resume_camera_after,
            }
        )

    context = mp.get_context("spawn")
    retrain_process = context.Process(target=_retrain_worker_process, daemon=True)
    retrain_process.start()
    RETRAIN_PROCESS = retrain_process

    return {
        "status": "started",
        "message": "Retraining started in background.",
        "camera_paused": resume_camera_after,
    }


def get_training_template_context(app_name: str) -> dict[str, Any]:
    return {
        "app_name": app_name,
        "threshold": ASSETS["threshold"],
        "input_size": ASSETS["input_size"],
        "model_name": ASSETS["model_path"].name,
        "label_names": get_label_names(),
    }


def get_health_payload() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": ASSETS["model_path"].name,
        "label_map": ASSETS["label_map_path"].name,
    }


def get_latest_payload() -> dict[str, Any]:
    payload = CAMERA_SERVICE.get_latest()
    payload["retraining"] = get_retrain_status()
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


def save_feedback_sample(face_roi: np.ndarray, label_name: str, metadata: dict[str, Any]) -> Path:
    safe_label = label_name.strip()
    if not safe_label:
        raise ValueError("Label name is required.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    label_root = FEEDBACK_ROOT / safe_label
    label_root.mkdir(parents=True, exist_ok=True)

    image_path = label_root / f"{timestamp}.jpg"
    metadata_path = label_root / f"{timestamp}.json"

    if not cv2.imwrite(str(image_path), face_roi):
        raise RuntimeError(f"Could not save feedback image: {image_path}")

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return image_path


def load_gray_image(path: Path, input_size: tuple[int, int]) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    if image.shape != input_size:
        image = cv2.resize(image, input_size, interpolation=cv2.INTER_CUBIC)
    return image


def build_training_samples(
    root: Path, name_to_label: dict[str, int], input_size: tuple[int, int]
):
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    x_train = []
    y_train = []

    if not root.exists():
        return x_train, np.array(y_train, dtype=np.int32)

    for person_name, label_id in sorted(name_to_label.items(), key=lambda item: item[1]):
        person_dir = root / person_name
        if not person_dir.exists() or not person_dir.is_dir():
            continue

        for file_path in sorted(person_dir.iterdir()):
            if not file_path.is_file() or file_path.suffix.lower() not in image_exts:
                continue
            image = load_gray_image(file_path, input_size)
            if image is None:
                continue
            x_train.append(image)
            y_train.append(label_id)

    return x_train, np.array(y_train, dtype=np.int32)


def retrain_runtime_model(include_feedback: bool = True) -> dict[str, Any]:
    with MODEL_LOCK:
        gc.collect()
        name_to_label = {name: label_id for label_id, name in ASSETS["label_map"].items()}
        base_root = PROJECT_ROOT / "data" / "7_augmented"
        if not base_root.exists():
            base_root = PROJECT_ROOT / "data" / "7_final_processed"

        x_train, y_train = build_training_samples(
            base_root, name_to_label, ASSETS["input_size"]
        )
        feedback_samples = 0

        if include_feedback:
            x_feedback, y_feedback = build_training_samples(
                FEEDBACK_ROOT, name_to_label, ASSETS["input_size"]
            )
            feedback_samples = len(x_feedback)
            x_train.extend(x_feedback)
            if len(y_feedback) > 0:
                y_train = (
                    np.concatenate([y_train, y_feedback])
                    if len(y_train) > 0
                    else y_feedback
                )

        if len(x_train) == 0:
            raise RuntimeError("No training images available for retraining.")

        total_samples = len(x_train)

        model = create_lbph_recognizer()
        model.setRadius(2)
        model.setNeighbors(8)
        model.setGridX(8)
        model.setGridY(8)
        model.train(list(x_train), y_train)

        model_path = MODELS_ROOT / "lbph_final.yml"
        model.save(str(model_path))

        runtime_config = dict(ASSETS["runtime_config"])
        runtime_config["identity_model"] = str(model_path)
        with RUNTIME_CONFIG_PATH.open("w", encoding="utf-8") as handle:
            json.dump(runtime_config, handle, indent=2)

        ASSETS["runtime_config"] = runtime_config
        ASSETS["model_path"] = model_path
        ASSETS["recognizer"] = model

        del x_train
        del y_train
        gc.collect()

        return {
            "status": "ok",
            "model": model_path.name,
            "training_samples": int(total_samples),
            "feedback_samples": int(feedback_samples),
        }


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

    face_roi = preprocess_face(
        gray, face_box, input_size=ASSETS["input_size"], padding=0.20
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
    return display, prediction, face_roi


def update_client_feedback_sample(face_roi: np.ndarray | None, prediction: dict[str, Any] | None):
    with CLIENT_SAMPLE_LOCK:
        CLIENT_FEEDBACK_SAMPLE["face_roi"] = None if face_roi is None else face_roi.copy()
        CLIENT_FEEDBACK_SAMPLE["prediction"] = None if prediction is None else dict(prediction)
        CLIENT_FEEDBACK_SAMPLE["captured_at"] = time.time() if face_roi is not None else 0.0


def get_client_feedback_sample(max_age_sec: float = 4.0):
    with CLIENT_SAMPLE_LOCK:
        face_roi = CLIENT_FEEDBACK_SAMPLE.get("face_roi")
        prediction = CLIENT_FEEDBACK_SAMPLE.get("prediction")
        captured_at = float(CLIENT_FEEDBACK_SAMPLE.get("captured_at") or 0.0)

        if face_roi is None or prediction is None:
            return None, None
        if captured_at <= 0.0 or (time.time() - captured_at) > max_age_sec:
            return None, None

        return face_roi.copy(), dict(prediction)


def predict_face(face_roi: np.ndarray) -> dict[str, Any]:
    with MODEL_LOCK:
        recognizer = ASSETS["recognizer"]
        label_map = ASSETS["label_map"]
        threshold = float(ASSETS["threshold"])
        strict_unknown_threshold = float(ASSETS.get("strict_unknown_threshold", threshold))

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
        update_client_feedback_sample(None, None)
        return {
            "status": "no_face",
            "message": "No face detected.",
        }

    face_roi = preprocess_face(gray, face_box, input_size=ASSETS["input_size"], padding=0.20)
    if face_roi is None:
        update_client_feedback_sample(None, None)
        return {
            "status": "invalid_roi",
            "message": "Face ROI could not be prepared.",
        }

    prediction = predict_face(face_roi)
    update_client_feedback_sample(face_roi, prediction)
    x, y, w, h = map(int, face_box)
    return {
        "status": "ok",
        "prediction": prediction,
        "bbox": {"x": x, "y": y, "w": w, "h": h},
        "threshold": float(ASSETS["threshold"]),
        "model": ASSETS["model_path"].name,
    }


def save_feedback(action: str, selected_label: str) -> dict[str, Any]:
    face_roi, prediction = CAMERA_SERVICE.get_feedback_sample()
    if face_roi is None:
        face_roi, prediction = get_client_feedback_sample(max_age_sec=4.0)

    if face_roi is None or prediction is None:
        raise ValueError("No recent face ROI is available yet. Keep your face in frame and try again.")

    if action == "confirm":
        if not prediction.get("accepted") or prediction.get("raw_name") in {None, "unknown", ""}:
            raise ValueError(
                "Current prediction is not confirmed. Select the correct label instead."
            )
        label_name = str(prediction.get("raw_name"))
    elif action == "correct":
        if not selected_label:
            raise ValueError("Please choose the correct label before saving correction.")
        if selected_label not in get_label_names():
            raise ValueError("Selected label is not in the known identity list.")
        label_name = selected_label
    else:
        raise ValueError("Invalid feedback action.")

    sample_path = save_feedback_sample(
        face_roi,
        label_name,
        {
            "action": action,
            "saved_label": label_name,
            "prediction": prediction,
            "timestamp": datetime.now().isoformat(),
        },
    )

    return {
        "status": "ok",
        "saved_label": label_name,
        "saved_path": str(sample_path),
    }


def retrain_with_feedback() -> dict[str, Any]:
    return retrain_runtime_model(include_feedback=True)


def retrain_with_feedback_async() -> dict[str, Any]:
    return start_retrain_with_feedback_async()


def start_camera() -> dict[str, Any]:
    if is_retraining_running():
        return {"status": "error", "message": "Camera is paused during retraining."}

    if not CAMERA_SERVICE.running:
        CAMERA_SERVICE.start()

    return {"status": "ok", "message": "Camera started"}


def stop_camera() -> dict[str, Any]:
    if CAMERA_SERVICE.running:
        CAMERA_SERVICE.stop()
    return {"status": "ok", "message": "Camera stopped"}


CAMERA_SERVICE = CameraService(frame_processor=process_camera_frame)
