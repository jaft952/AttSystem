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
        inference_interval_sec: float = 0.08,
        jpeg_quality: int = 75,
        stale_grab_count: int = 2,
    ):
        self.frame_processor = frame_processor
        self.camera_index = camera_index
        self.capture = None
        self.capture_thread = None
        self.inference_thread = None
        self.running = False
        
        self.lock = threading.Lock()
        
        self.latest_frame = None
        self.annotated_frame = None
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
        
        self.last_face_box: tuple[int, int, int, int] | None = None
        self.prediction_ttl = 0
        self.prediction_history = []

    def start(self):
        if self.running: return
        self.capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = cv2.VideoCapture(self.camera_index)
        if not self.capture.isOpened():
            raise RuntimeError('Could not open the webcam')

        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FPS, 30)

        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.capture_thread.start()
        self.inference_thread.start()

    def stop(self):
        self.running = False
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if self.inference_thread and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=1.0)

    def _capture_loop(self):
        while self.running and self.capture is not None:
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.1)
                continue
            
            frame = cv2.flip(frame, 1)
            with self.lock:
                self.latest_frame = frame.copy()
                
            annotated = frame.copy()
            with self.lock:
                if self.last_face_box is not None:
                    x, y, w, h = self.last_face_box
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 255, 255), 2)
            
            ok, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if ok:
                with self.lock:
                    self.frame_bytes = buffer.tobytes()
                    self.frame_count += 1
            time.sleep(0.01)

    def _inference_loop(self):
        while self.running:
            with self.lock:
                frame_to_process = self.latest_frame.copy() if self.latest_frame is not None else None
            
            if frame_to_process is None:
                time.sleep(0.05)
                continue
                
            _, prediction, face_roi = self.frame_processor(frame_to_process)
            
            with self.lock:
                bbox = prediction.get("bbox") if isinstance(prediction, dict) else None
                if isinstance(bbox, dict) and bbox:
                    new_box = (int(bbox["x"]), int(bbox["y"]), int(bbox["w"]), int(bbox["h"]))
                    if self.last_face_box is not None and self.prediction_ttl > 0:
                        # EMA Smoothing
                        alpha = 0.6
                        ox, oy, ow, oh = self.last_face_box
                        nx, ny, nw, nh = new_box
                        self.last_face_box = (
                            int(alpha * nx + (1.0 - alpha) * ox),
                            int(alpha * ny + (1.0 - alpha) * oy),
                            int(alpha * nw + (1.0 - alpha) * ow),
                            int(alpha * nh + (1.0 - alpha) * oh),
                        )
                    else:
                        self.last_face_box = new_box
                    self.prediction_ttl = 5
                    self.latest_face_roi = face_roi
                else:
                    if self.prediction_ttl > 0:
                        self.prediction_ttl -= 1
                    else:
                        self.last_face_box = None
                        self.latest_face_roi = face_roi

                if isinstance(prediction, dict) and prediction.get("status") == "ok":
                    self.prediction_history.append({
                        "name": prediction.get("raw_name", "unknown"),
                        "confidence": prediction.get("confidence", 0.0),
                        "accepted": prediction.get("accepted", False),
                    })
                else:
                    self.prediction_history.append(None)

               
                if len(self.prediction_history) > 10:
                    self.prediction_history.pop(0)

                votes = {}
                for p in self.prediction_history:
                    
                    if p is not None and p["accepted"] and p["confidence"] and p["confidence"] >= 0.50:
                        valid_name = p["name"]
                        if valid_name != "unknown":
                            votes[valid_name] = votes.get(valid_name, 0) + 1

                stable_name = None
                is_stable = False
                
          
                if votes:
                    top_name, top_votes = max(votes.items(), key=lambda x: x[1])
                    if top_votes >= 2:
                        stable_name = top_name
                        is_stable = True

      
                if isinstance(prediction, dict):
                    prediction["is_stable"] = is_stable
                    prediction["stable_name"] = stable_name

                self.latest_prediction = prediction

            time.sleep(self.inference_interval_sec)

    def get_latest(self) -> dict[str, Any]:
        with self.lock:
            return {
                "prediction": self.latest_prediction,
                "frame_count": self.frame_count,
                "last_error": self.last_error,
                "running": self.running,
            }

    def get_frame(self) -> bytes | None:
        with self.lock: return self.frame_bytes

    def get_feedback_sample(self):
        with self.lock:
            if self.latest_face_roi is None:
                return None, dict(self.latest_prediction)
            return self.latest_face_roi.copy(), dict(self.latest_prediction)