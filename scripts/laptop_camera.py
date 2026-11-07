"""Serve this machine's webcam as an MJPEG stream over the local network.

Run this on the laptop, then read the stream from the desktop with:

    cv2.VideoCapture("http://<laptop-ip>:8080/video")

No cable between the two machines is needed, only the same Wi-Fi or router.
The laptop camera is used to track the operator's hand with MediaPipe, which is
a different view from the RealSense that watches the robot hand.

Usage:
    python scripts/laptop_camera.py
    python scripts/laptop_camera.py --camera 0 --width 640 --height 480 --port 8080
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2


def local_ip() -> str:
    """Best-effort LAN address of this machine (no traffic is actually sent)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


class FrameSource:
    """Capture thread that always exposes the most recent frame as JPEG."""

    def __init__(self, camera: int, width: int, height: int, fps: float) -> None:
        # CAP_DSHOW opens faster on Windows; macOS and Linux use the default backend.
        if sys.platform == "win32":
            self.cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(camera)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(camera)
        if not self.cap.isOpened():
            raise RuntimeError(f"camera {camera} could not be opened")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.jpeg: bytes | None = None
        self.lock = threading.Lock()
        self.running = True
        self.frames = 0

        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        interval = 1.0 / 30.0
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self.lock:
                    self.jpeg = buf.tobytes()
                    self.frames += 1
            time.sleep(interval)

    def latest(self) -> bytes | None:
        with self.lock:
            return self.jpeg

    def close(self) -> None:
        self.running = False
        self.cap.release()


class StreamHandler(BaseHTTPRequestHandler):
    source: FrameSource

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path not in ("/", "/video"):
            self.send_error(404)
            return

        if self.path == "/":
            body = b"ok - stream at /video\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while True:
                jpeg = self.source.latest()
                if jpeg is None:
                    time.sleep(0.02)
                    continue
                self.wfile.write(b"--frame\r\n")
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                time.sleep(1.0 / 30.0)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, fmt: str, *args) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a webcam as an MJPEG stream")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    source = FrameSource(args.camera, args.width, args.height, args.fps)
    StreamHandler.source = source

    server = ThreadingHTTPServer(("0.0.0.0", args.port), StreamHandler)
    ip = local_ip()

    print(f"camera          : {args.camera} at {args.width}x{args.height}@{args.fps:g}")
    print(f"listening on    : 0.0.0.0:{args.port}")
    print(f"open on desktop : http://{ip}:{args.port}/video")
    print("press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.shutdown()
        source.close()


if __name__ == "__main__":
    main()
