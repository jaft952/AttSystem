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
        stream_fps: int = 20,
    ):
        self.frame_processor = frame_processor
        self.camera_index = camera_index
        self.capture = None
        self.thread = None
        self._inference_thread = None
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
        # Target encoded-stream frame rate (caps JPEG encoding CPU use).
        self.stream_fps = max(1, min(30, stream_fps))

        self.last_inference_ts = 0.0
        self.last_face_box: tuple[int, int, int, int] | None = None
        self.last_face_roi_cache: np.ndarray | None = None
        self.last_prediction_cache: dict[str, Any] = dict(self.latest_prediction)

        # Adaptive inference throttle: starts at the configured interval and
        # is updated each cycle to equal the actual measured inference latency.
        # This prevents submitting new frames faster than they can be processed.
        _MIN_INFERENCE_INTERVAL = 0.05  # cap at 20 Hz maximum submission rate
        self._min_inference_interval = _MIN_INFERENCE_INTERVAL
        self._last_inference_duration: float = max(inference_interval_sec, _MIN_INFERENCE_INTERVAL)

        # Decoupled inference: pending frame submitted by the camera loop.
        self._pending_frame: np.ndarray | None = None
        self._pending_frame_lock = threading.Lock()
        self._inference_event = threading.Event()

        # Frame-ready condition: signals stream consumers the instant a new
        # JPEG is encoded so they do not need to busy-poll.
        # Uses self.lock as the underlying mutex so frame_bytes / frame_count
        # updates and the notify are always atomic.
        self._frame_condition = threading.Condition(self.lock)

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
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._inference_thread.start()

    def stop(self):
        self.running = False
        # Wake the inference thread so it can observe running=False and exit.
        self._inference_event.set()
        # Wake any stream_frames() generators blocked in wait_for_next_frame().
        with self._frame_condition:
            self._frame_condition.notify_all()

        if self.capture is not None:
            self.capture.release()
            self.capture = None

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.thread = None

        if self._inference_thread is not None and self._inference_thread.is_alive():
            # Allow up to one full inference cycle (dlib can take ~300 ms on CPU).
            self._inference_thread.join(timeout=2.0)
        self._inference_thread = None

        with self.lock:
            self.frame_bytes = None
            self.latest_face_roi = None

    def _inference_loop(self):
        """Background thread: runs the heavy frame_processor without blocking the camera loop."""
        while self.running:
            triggered = self._inference_event.wait(timeout=0.5)
            if not self.running:
                break
            if not triggered:
                continue
            self._inference_event.clear()

            with self._pending_frame_lock:
                frame = self._pending_frame

            if frame is None:
                continue

            try:
                t0 = time.perf_counter()
                _annotated, prediction, face_roi = self.frame_processor(frame)
                elapsed = time.perf_counter() - t0
                # Update adaptive interval: use the measured latency so the
                # camera loop backs off automatically when inference is slow.
                self._last_inference_duration = max(self._min_inference_interval, elapsed)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning("Inference error: %s", exc, exc_info=True)
                continue

            bbox = prediction.get("bbox") if isinstance(prediction, dict) else None
            new_face_box = (
                (int(bbox["x"]), int(bbox["y"]), int(bbox["w"]), int(bbox["h"]))
                if isinstance(bbox, dict)
                else None
            )

            with self.lock:
                self.last_prediction_cache = dict(prediction)
                self.last_face_roi_cache = None if face_roi is None else face_roi.copy()
                self.last_face_box = new_face_box
                self.latest_face_roi = self.last_face_roi_cache
                self.latest_prediction = dict(prediction)

    def _update_loop(self):
        stream_interval = 1.0 / self.stream_fps
        last_stream_ts = 0.0

        while self.running and self.capture is not None:
            # Blocking read: waits until the camera delivers the next frame.
            # This naturally paces the loop to the camera's frame rate (~30 Hz)
            # instead of busy-looping at 1000 Hz with redundant grab() calls,
            # which wasted CPU and caused GIL contention with the inference and
            # stream threads.
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

            # Submit the latest frame to the inference thread using the adaptive
            # interval (actual measured inference latency). This prevents
            # submitting frames faster than they can be processed.
            if (now - self.last_inference_ts) >= self._last_inference_duration:
                self.last_inference_ts = now
                with self._pending_frame_lock:
                    self._pending_frame = frame.copy()
                self._inference_event.set()

            # Throttle JPEG encoding to the target stream FPS so that encoding
            # does not consume CPU on every camera frame (e.g. at 30 FPS input
            # but 20 FPS stream, one in three encodes is skipped).
            if (now - last_stream_ts) >= stream_interval:
                last_stream_ts = now

                # Encode the current frame using the most-recently cached
                # face box — never wait for inference to finish.
                # Avoid a full-frame copy when there is no box to draw.
                annotated_frame = frame
                with self.lock:
                    cached_box = self.last_face_box

                if cached_box is not None:
                    annotated_frame = frame.copy()
                    x, y, w, h = cached_box
                    cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

                ok_enc, buffer = cv2.imencode(
                    ".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                )
                if ok_enc:
                    # Notify stream consumers atomically with the frame update
                    # so wait_for_next_frame() wakes up immediately instead of
                    # waiting for the next polling interval.
                    with self._frame_condition:
                        self.frame_bytes = buffer.tobytes()
                        self.last_error = None
                        self.frame_count += 1
                        self._frame_condition.notify_all()

        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def wait_for_next_frame(self, last_seq: int, timeout: float = 0.1) -> tuple[bytes | None, int]:
        """Block until a new encoded JPEG frame is available or *timeout* elapses.

        Unlike polling with ``time.sleep``, this releases the GIL while waiting
        so the camera loop and inference thread can run unimpeded.  Multiple
        concurrent stream consumers (e.g. two browser tabs) are all woken by
        ``notify_all()`` and each independently track their own *last_seq*.

        Returns:
            A ``(frame_bytes, frame_count)`` tuple.  ``frame_bytes`` is ``None``
            when no frame has been encoded yet; ``frame_count`` is the sequence
            number of the returned frame (equal to *last_seq* on timeout).
        """
        with self._frame_condition:
            deadline = time.monotonic() + timeout
            while self.frame_count == last_seq and self.running:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._frame_condition.wait(timeout=remaining)
            return self.frame_bytes, self.frame_count

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

    def get_frame_packet(self) -> tuple[bytes | None, int]:
        with self.lock:
            return self.frame_bytes, self.frame_count

    def get_feedback_sample(self):
        with self.lock:
            if self.latest_face_roi is None:
                return None, dict(self.latest_prediction)
            return self.latest_face_roi.copy(), dict(self.latest_prediction)
