import base64
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_ROOT = PROJECT_ROOT / "models"
VIEWS_ROOT = PROJECT_ROOT / "presentation" / "views"
UI_ROOT = PROJECT_ROOT / "presentation" / "ui"

RUNTIME_CONFIG_PATH = MODELS_ROOT / "realtime_model_config.json"
HAAR_CASCADE_PATH = Path(cv2.__file__).resolve().parent / "data" / "haarcascade_frontalface_default.xml"
FEEDBACK_ROOT = PROJECT_ROOT / "data" / "8_feedback"
MODEL_LOCK = threading.Lock()

FEEDBACK_ROOT.mkdir(parents=True, exist_ok=True)


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
    threshold = float(runtime_config.get("identity_confidence_threshold", 160.0))
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
        "input_size": input_size,
        "recognizer": recognizer,
        "label_map": label_map,
        "face_cascade": face_cascade,
    }


ASSETS = load_runtime_assets()
APP = Flask(__name__, template_folder=str(VIEWS_ROOT), static_folder=str(UI_ROOT), static_url_path="/ui")


def get_label_names() -> list[str]:
    return [name for _, name in sorted(ASSETS["label_map"].items(), key=lambda item: item[0])]


class CameraService:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.capture = None
        self.thread = None
        self.running = False
        self.lock = threading.Lock()
        self.frame_bytes: bytes | None = None
        self.latest_face_roi: np.ndarray | None = None
        self.latest_prediction: dict[str, Any] = {
            "status": "idle",
            "name": "Waiting...",
            "raw_name": "unknown",
            "confidence": None,
            "accepted": False,
            "bbox": None,
            "message": "Camera stream not started.",
        }
        self.frame_count = 0
        self.last_error: str | None = None

    def start(self):
        if self.running:
            return

        capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            raise RuntimeError("Could not open the webcam on this machine.")

        self.capture = capture
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _update_loop(self):
        while self.running and self.capture is not None:
            ok, frame = self.capture.read()
            if not ok:
                with self.lock:
                    self.last_error = "Failed to read a frame from the webcam."
                    self.latest_prediction = {
                        "status": "error",
                        "name": "Camera error",
                        "raw_name": "unknown",
                        "confidence": None,
                        "accepted": False,
                        "bbox": None,
                        "message": self.last_error,
                    }
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)
            annotated_frame, prediction, face_roi = self._process_frame(frame)

            ok, buffer = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                with self.lock:
                    self.frame_bytes = buffer.tobytes()
                    self.latest_face_roi = None if face_roi is None else face_roi.copy()
                    self.latest_prediction = prediction
                    self.last_error = None
                    self.frame_count += 1

            time.sleep(0.03)

        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _process_frame(self, frame: np.ndarray):
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = ASSETS["face_cascade"].detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
        )

        face_box = pick_largest_face(faces)
        if face_box is None:
            cv2.putText(
                display,
                "No face detected",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
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
        color = (0, 255, 0) if prediction["accepted"] else (0, 0, 255)
        cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            display,
            f"{prediction['name']} | conf={prediction['confidence']:.1f}",
            (x, max(30, y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        prediction.update(
            {
                "status": "ok",
                "bbox": {"x": x, "y": y, "w": w, "h": h},
                "message": "Prediction updated.",
            }
        )
        return display, prediction, face_roi

    def get_latest(self):
        with self.lock:
            return {
                "prediction": self.latest_prediction,
                "frame_count": self.frame_count,
                "last_error": self.last_error,
                "running": self.running,
            }

    def get_frame(self):
        with self.lock:
            return self.frame_bytes

    def get_feedback_sample(self):
        with self.lock:
            if self.latest_face_roi is None:
                return None, dict(self.latest_prediction)
            return self.latest_face_roi.copy(), dict(self.latest_prediction)


CAMERA_SERVICE = CameraService()


def ensure_camera_service():
    if not CAMERA_SERVICE.running:
        CAMERA_SERVICE.start()


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


def build_training_samples(root: Path, name_to_label: dict[str, int], input_size: tuple[int, int]):
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    X_train = []
    y_train = []

    if not root.exists():
        return X_train, np.array(y_train, dtype=np.int32)

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
            X_train.append(image)
            y_train.append(label_id)

    return X_train, np.array(y_train, dtype=np.int32)


def retrain_runtime_model(include_feedback: bool = True) -> dict[str, Any]:
    with MODEL_LOCK:
        name_to_label = {name: label_id for label_id, name in ASSETS["label_map"].items()}
        base_root = PROJECT_ROOT / "data" / "7_augmented"
        if not base_root.exists():
            base_root = PROJECT_ROOT / "data" / "7_final_processed"

        X_train, y_train = build_training_samples(base_root, name_to_label, ASSETS["input_size"])
        feedback_samples = 0

        if include_feedback:
            X_feedback, y_feedback = build_training_samples(FEEDBACK_ROOT, name_to_label, ASSETS["input_size"])
            feedback_samples = len(X_feedback)
            X_train.extend(X_feedback)
            if len(y_feedback) > 0:
                y_train = np.concatenate([y_train, y_feedback]) if len(y_train) > 0 else y_feedback

        if len(X_train) == 0:
            raise RuntimeError("No training images available for retraining.")

        model = create_lbph_recognizer()
        model.setRadius(2)
        model.setNeighbors(8)
        model.setGridX(8)
        model.setGridY(8)
        model.train(list(X_train), y_train)

        model_path = MODELS_ROOT / "lbph_final.yml"
        model.save(str(model_path))

        runtime_config = dict(ASSETS["runtime_config"])
        runtime_config["identity_model"] = str(model_path)
        with RUNTIME_CONFIG_PATH.open("w", encoding="utf-8") as handle:
            json.dump(runtime_config, handle, indent=2)

        ASSETS["runtime_config"] = runtime_config
        ASSETS["model_path"] = model_path
        ASSETS["recognizer"] = model

        return {
            "status": "ok",
            "model": model_path.name,
            "training_samples": int(len(X_train)),
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


def predict_face(face_roi: np.ndarray) -> dict[str, Any]:
    with MODEL_LOCK:
        recognizer = ASSETS["recognizer"]
        label_map = ASSETS["label_map"]
        threshold = float(ASSETS["threshold"])

        label_id, confidence = recognizer.predict(face_roi)
    raw_name = label_map.get(int(label_id), "unknown")
    accepted = float(confidence) <= threshold
    display_name = raw_name if accepted else "unknown"

    return {
        "label_id": int(label_id),
        "raw_name": raw_name,
        "name": display_name,
        "confidence": float(confidence),
        "accepted": bool(accepted),
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


@APP.get("/")
def index():
    return render_template(
        "index.html",
        app_name="AttSystem",
        threshold=ASSETS["threshold"],
        input_size=ASSETS["input_size"],
        model_name=ASSETS["model_path"].name,
        label_names=get_label_names(),
    )


@APP.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model": ASSETS["model_path"].name,
        "label_map": ASSETS["label_map_path"].name,
    })


@APP.get("/api/latest")
def api_latest():
    ensure_camera_service()
    return jsonify(CAMERA_SERVICE.get_latest())


@APP.get("/video_feed")
def video_feed():
    ensure_camera_service()

    def generate():
        while True:
            frame = CAMERA_SERVICE.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
            time.sleep(0.03)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@APP.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")
    if not image_data:
        return jsonify({"error": "Missing image payload."}), 400

    try:
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
            return jsonify({
                "status": "no_face",
                "message": "No face detected.",
            })

        face_roi = preprocess_face(gray, face_box, input_size=ASSETS["input_size"], padding=0.20)
        if face_roi is None:
            return jsonify({
                "status": "invalid_roi",
                "message": "Face ROI could not be prepared.",
            })

        prediction = predict_face(face_roi)
        x, y, w, h = map(int, face_box)
        return jsonify({
            "status": "ok",
            "prediction": prediction,
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "threshold": float(ASSETS["threshold"]),
            "model": ASSETS["model_path"].name,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@APP.post("/api/feedback")
def api_feedback():
    payload = request.get_json(silent=True) or {}
    action = str(payload.get("action", "confirm")).strip().lower()
    selected_label = str(payload.get("label", "")).strip()

    ensure_camera_service()
    face_roi, prediction = CAMERA_SERVICE.get_feedback_sample()

    if face_roi is None:
        return jsonify({"error": "No face ROI is available yet."}), 400

    if action == "confirm":
        if not prediction.get("accepted") or prediction.get("raw_name") in {None, "unknown", ""}:
            return jsonify({"error": "Current prediction is not confirmed. Select the correct label instead."}), 400
        label_name = str(prediction.get("raw_name"))
    elif action == "correct":
        if not selected_label:
            return jsonify({"error": "Please choose the correct label before saving correction."}), 400
        if selected_label not in get_label_names():
            return jsonify({"error": "Selected label is not in the known identity list."}), 400
        label_name = selected_label
    else:
        return jsonify({"error": "Invalid feedback action."}), 400

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

    return jsonify({
        "status": "ok",
        "saved_label": label_name,
        "saved_path": str(sample_path),
    })


@APP.post("/api/retrain")
def api_retrain():
    try:
        result = retrain_runtime_model(include_feedback=True)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=5000, debug=True)
