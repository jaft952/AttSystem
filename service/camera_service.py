from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
from typing import Any, Callable
import cv2
import numpy as np


def _inference_worker_loop(
    frame_processor: Callable[[np.ndarray], tuple[np.ndarray | None, dict[str, Any], np.ndarray | None]],
    input_queue: mp.Queue,
    output_queue: mp.Queue,
) -> None:
    while True:
        item = input_queue.get()
        if item is None:
            break

        frame_seq, frame = item
        try:
            started = time.perf_counter()
            _annotated, prediction, face_roi = frame_processor(frame)
            elapsed = time.perf_counter() - started
            output_queue.put((frame_seq, prediction, face_roi, elapsed, None))
        except Exception as exc:  # pragma: no cover - worker-side safety net
            output_queue.put((frame_seq, None, None, 0.0, repr(exc)))


class CameraService:
    def __init__(
        self,
        frame_processor: Callable[
            [np.ndarray], tuple[np.ndarray | None, dict[str, Any], np.ndarray | None]
        ],
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
        self._inference_process: Any | None = None
        self._result_thread: threading.Thread | None = None
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
        self.last_client_ts = time.time()

        self.inference_interval_sec = inference_interval_sec
        self.jpeg_quality = int(max(30, min(95, jpeg_quality)))
        self.stale_grab_count = max(0, stale_grab_count)
        # Target encoded-stream frame rate (caps JPEG encoding CPU use).
        self.stream_fps = max(1, min(30, stream_fps))

        self.last_inference_ts = 0.0
        self.last_face_roi_cache: np.ndarray | None = None
        self.last_prediction_cache: dict[str, Any] = dict(self.latest_prediction)

        # Adaptive inference throttle: starts at the configured interval and
        # is updated each cycle to equal the actual measured inference latency.
        # This prevents submitting new frames faster than they can be processed.
        _MIN_INFERENCE_INTERVAL = 0.05  # cap at 20 Hz maximum submission rate
        self._min_inference_interval = _MIN_INFERENCE_INTERVAL
        self._last_inference_duration: float = max(inference_interval_sec, _MIN_INFERENCE_INTERVAL)

        self._mp_context = mp.get_context("spawn")
        self._inference_input_queue: mp.Queue | None = None
        self._inference_output_queue: mp.Queue | None = None
        self._submitted_frame_seq = 0
        self._last_result_seq = 0
        self._worker_error: str | None = None
        self._inference_worker_running = False

        # Signals stream consumers as soon as a new JPEG frame is encoded.
        self._frame_condition = threading.Condition(self.lock)

        # Signals SSE consumers as soon as a new prediction is available.
        self._prediction_condition = threading.Condition(self.lock)
        self._prediction_seq: int = 0
        self.last_face_box: tuple[int, int, int, int] | None = None

    def keep_alive(self):
        self.last_client_ts = time.time()

    def start(self):
        if self.running:
            self.keep_alive()
            return

        self.keep_alive()
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
        self._start_inference_worker()

    def _start_inference_worker(self):
        self._inference_input_queue = self._mp_context.Queue(maxsize=1)
        self._inference_output_queue = self._mp_context.Queue(maxsize=1)
        self._inference_worker_running = True
        self._inference_process = self._mp_context.Process(
            target=_inference_worker_loop,
            args=(self.frame_processor, self._inference_input_queue, self._inference_output_queue),
            daemon=True,
        )
        self._inference_process.start()
        self._result_thread = threading.Thread(target=self._result_loop, daemon=True)
        self._result_thread.start()

    def _stop_inference_worker(self):
        self._inference_worker_running = False

        if self._inference_input_queue is not None:
            try:
                self._inference_input_queue.put_nowait(None)
            except Exception:
                pass

        if self._inference_process is not None and self._inference_process.is_alive():
            self._inference_process.join(timeout=2.0)
            if self._inference_process.is_alive():
                self._inference_process.terminate()
                self._inference_process.join(timeout=1.0)

        if self._result_thread is not None and self._result_thread.is_alive():
            self._result_thread.join(timeout=1.5)

        self._inference_process = None
        self._result_thread = None

        if self._inference_input_queue is not None:
            self._inference_input_queue.close()
            self._inference_input_queue = None
        if self._inference_output_queue is not None:
            self._inference_output_queue.close()
            self._inference_output_queue = None

    def restart_inference_worker(self):
        if not self.running:
            return
        self._stop_inference_worker()
        self._start_inference_worker()

    def stop(self):
        self.running = False
        # Wake any waiting frame/prediction consumers.
        with self._frame_condition:
            self._frame_condition.notify_all()
        with self._prediction_condition:
            self._prediction_condition.notify_all()

        if self.capture is not None:
            self.capture.release()
            self.capture = None

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.thread = None

        self._stop_inference_worker()

        with self.lock:
            self.frame_bytes = None
            self.latest_face_roi = None

    def _result_loop(self):
        """Collect predictions from the worker process and publish them to the UI state."""
        assert self._inference_output_queue is not None

        while self.running and self._inference_worker_running:
            try:
                frame_seq, prediction, face_roi, elapsed, worker_error = self._inference_output_queue.get(
                    timeout=0.5
                )
            except queue.Empty:
                continue
            except Exception:
                if not self.running:
                    break
                continue

            if not self.running or not self._inference_worker_running:
                break

            if worker_error is not None:
                self._worker_error = str(worker_error)
                continue

            if frame_seq < self._last_result_seq:
                continue
            self._last_result_seq = frame_seq
            self._last_inference_duration = max(self._min_inference_interval, float(elapsed))

            if prediction is None:
                continue

            with self._prediction_condition:
                self.last_prediction_cache = dict(prediction)
                self.last_face_roi_cache = None if face_roi is None else face_roi.copy()
                self.latest_face_roi = self.last_face_roi_cache
                self.latest_prediction = dict(prediction)
                bbox = prediction.get("bbox")
                if isinstance(bbox, dict):
                    try:
                        self.last_face_box = (
                            int(bbox.get("x", 0)),
                            int(bbox.get("y", 0)),
                            int(bbox.get("w", 0)),
                            int(bbox.get("h", 0)),
                        )
                    except (TypeError, ValueError):
                        self.last_face_box = None
                else:
                    self.last_face_box = None
                self._prediction_seq += 1
                self._prediction_condition.notify_all()

    def _update_loop(self):
        stream_interval = 1.0 / self.stream_fps
        last_stream_ts = 0.0

        while self.running and self.capture is not None:
            if time.time() - self.last_client_ts > 5.0:
                # Auto-stop camera to save resources if no clients are connected
                threading.Thread(target=self.stop, daemon=True).start()
                break

            try:
                ok, frame = self.capture.retrieve()
                if not ok:
                    ok, frame = self.capture.read()
            except AttributeError:
                break
            except cv2.error:
                # Catch random C++ driver exceptions from OpenCV DirectShow backend
                ok = False
                frame = None

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
                if self._inference_input_queue is not None:
                    self._submitted_frame_seq += 1
                    try:
                        self._inference_input_queue.put_nowait((self._submitted_frame_seq, frame.copy()))
                    except queue.Full:
                        try:
                            while True:
                                self._inference_input_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self._inference_input_queue.put_nowait((self._submitted_frame_seq, frame.copy()))
                        except queue.Full:
                            pass

            # Throttle JPEG encoding to the target stream FPS so that encoding
            # does not consume CPU on every camera frame (e.g. at 30 FPS input
            # but 20 FPS stream, one in three encodes is skipped).
            if (now - last_stream_ts) >= stream_interval:
                last_stream_ts = now

                ok, buffer = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                )
                if ok:
                    with self._frame_condition:
                        self.frame_bytes = buffer.tobytes()
                        self.last_error = None
                        self.frame_count += 1
                        self._frame_condition.notify_all()

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
                "worker_error": self._worker_error,
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

    def wait_for_next_frame(self, last_seq: int, timeout: float = 0.1) -> tuple[bytes | None, int]:
        with self._frame_condition:
            deadline = time.monotonic() + timeout
            while self.frame_count == last_seq and self.running:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._frame_condition.wait(timeout=remaining)
            return self.frame_bytes, self.frame_count

    def wait_for_next_prediction(self, last_seq: int, timeout: float = 1.0) -> tuple[dict[str, Any], int]:
        with self._prediction_condition:
            deadline = time.monotonic() + timeout
            while self._prediction_seq == last_seq and self.running:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._prediction_condition.wait(timeout=remaining)
            return dict(self.latest_prediction), self._prediction_seq
