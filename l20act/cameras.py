"""Camera sources and hand landmark detection.

Two cameras are involved and they serve different purposes:

* the robot camera (RealSense D405) produces the observation the policy sees;
* the operator camera (an HTTP MJPEG stream from a laptop or phone) is only used
  to read the demonstrator's hand and must never appear in the dataset, or the
  policy would learn to look for a human hand that is absent at deployment.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "assets" / "models" / "hand_landmarker.task"


class RobotCamera:
    """RealSense color stream."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30,
                 warmup: int = 10):
        import pyrealsense2 as rs

        self.width = width
        self.height = height
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        try:
            self._pipeline.start(config)
        except Exception as exc:
            raise SystemExit(
                f"cannot start the RealSense pipeline: {exc}\n"
                "  1. Check the USB link with: "
                'python -c "import pyrealsense2 as rs;'
                'print([d.get_info(rs.camera_info.usb_type_descriptor) '
                'for d in rs.context().query_devices()])"\n'
                "  2. The D405 needs a USB 3 port; a USB 2 link enumerates but "
                "never delivers frames"
            ) from None
        # Auto-exposure needs a few frames before the image is usable.
        for _ in range(warmup):
            self._pipeline.wait_for_frames()

    def read(self):
        frames = self._pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            return None
        return np.asanyarray(color.get_data()).copy()

    def close(self) -> None:
        self._pipeline.stop()


class ThreadedCamera:
    """Reads a video source in a background thread and keeps the latest frame.

    A network stream can stall for a few hundred milliseconds; reading it inline
    would freeze the whole control loop, so the loop always gets the newest
    frame that arrived instead of waiting for the next one.
    """

    def __init__(self, source, width: int = 640, height: int = 480,
                 timeout: float = 8.0):
        import cv2

        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise SystemExit(
                f"cannot open the operator camera: {source}\n"
                "Start the stream on the other machine first, then confirm the "
                "URL is reachable from this one."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = width
        self.height = height

        self._latest = None
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

        deadline = time.monotonic() + timeout
        while self._latest is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if self._latest is None:
            self.close()
            raise SystemExit(f"no frames arrived from {source} within {timeout:.0f}s")

    def _loop(self) -> None:
        while not self._stop:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._latest = frame

    def read(self):
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def close(self) -> None:
        self._stop = True
        self._thread.join(timeout=1.0)
        self._cap.release()


class HandTracker:
    """MediaPipe Tasks hand landmarker returning normalized (x, y) points."""

    def __init__(self, model_path: Path | str = DEFAULT_MODEL, num_hands: int = 1):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not Path(model_path).exists():
            raise SystemExit(
                f"hand landmark model not found: {model_path}\n"
                "Download it with:\n"
                "  curl -L -o assets/models/hand_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/1/hand_landmarker.task"
            )

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._mp = mp
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, rgb_frame, timestamp_ms: int):
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.hand_landmarks:
            return None
        return [(lm.x, lm.y) for lm in result.hand_landmarks[0]]

    def close(self) -> None:
        self._landmarker.close()
