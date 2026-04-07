import threading
import time
from typing import Any, Callable
import cv2
import numpy as np


class CameraService:
    def __init__(
        self,
        frame_processor: Callable[[np.ndarray], tuple[np.ndarray, dict[str, Any], np.ndarray | None]],
        camera_index: int = 0,
        inference_interval_sec: float = 0.11,
        jpeg_quality: int = 75,
        stale_grab_count: int = 2,
    ):
        self.frame_processor = frame_processor
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

        self.inference_interval_sec = inference_interval_sec
        self.jpeg_quality = int(max(30, min(95, jpeg_quality)))
        self.stale_grab_count = max(0, stale_grab_count)

        self.last_inference_ts = 0.0
        self.last_face_box: tuple[int, int, int, int] | None = None
        self.last_face_roi_cache: np.ndarray | None = None
        self.last_prediction_cache: dict[str, Any] = dict(self.latest_prediction)

    def start(self):
        if self.running:
            return

        capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            raise RuntimeError("Could not open the webcam on this machine.")

        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.CAP_PROP_FPS, 30)

        self.capture = capture
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.capture is not None:
            self.capture.release()
            self.capture = None

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.thread = None

        with self.lock:
            self.frame_bytes = None
            self.latest_face_roi = None

    def _update_loop(self):
        while self.running and self.capture is not None:
            for _ in range(self.stale_grab_count):
                self.capture.grab()

            ok, frame = self.capture.retrieve()
            if not ok:
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
            now = time.perf_counter()
            run_inference = (now - self.last_inference_ts) >= self.inference_interval_sec

            if run_inference:
                annotated_frame, prediction, face_roi = self.frame_processor(frame)
                self.last_inference_ts = now
                self.last_prediction_cache = dict(prediction)
                self.last_face_roi_cache = None if face_roi is None else face_roi.copy()

                bbox = prediction.get("bbox") if isinstance(prediction, dict) else None
                self.last_face_box = (
                    (int(bbox["x"]), int(bbox["y"]), int(bbox["w"]), int(bbox["h"]))
                    if isinstance(bbox, dict)
                    else None
                )
            else:
                annotated_frame = frame.copy()
                if self.last_face_box is not None:
                    x, y, w, h = self.last_face_box
                    cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

                prediction = dict(self.last_prediction_cache)
                if prediction.get("status") == "ok":
                    prediction["message"] = "Prediction refreshed at low-latency cadence."
                face_roi = self.last_face_roi_cache

            ok, buffer = cv2.imencode(
                ".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            )
            if ok:
                with self.lock:
                    self.frame_bytes = buffer.tobytes()
                    self.latest_face_roi = None if face_roi is None else face_roi.copy()
                    self.latest_prediction = prediction
                    self.last_error = None
                    self.frame_count += 1

            time.sleep(0.001)

        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def get_latest(self) -> dict[str, Any]:
        with self.lock:
            return {
                "prediction": self.latest_prediction,
                "frame_count": self.frame_count,
                "last_error": self.last_error,
                "running": self.running,
            }

    def get_frame(self) -> bytes | None:
        with self.lock:
            return self.frame_bytes

    def get_feedback_sample(self):
        with self.lock:
            if self.latest_face_roi is None:
                return None, dict(self.latest_prediction)
            return self.latest_face_roi.copy(), dict(self.latest_prediction)
