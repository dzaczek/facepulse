import logging
import os
import platform
import threading
import time
from typing import Callable

import cv2
import numpy as np

_IS_MACOS = platform.system() == "Darwin"

logger = logging.getLogger(__name__)


def _open_capture(source: str | int) -> cv2.VideoCapture:
    """
    Open a VideoCapture with the correct backend.

    On macOS, integer indices MUST use cv2.CAP_AVFOUNDATION — plain
    cv2.VideoCapture(N) sometimes silently fails for USB cameras (e.g. Brio)
    even though the device is detected by ffmpeg.
    """
    if _IS_MACOS and isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            return cap
        cap.release()
        # fallback: try without explicit backend
        return cv2.VideoCapture(source)
    return cv2.VideoCapture(source)


class CameraLoop:
    """
    Captures frames from a webcam or video file and calls registered callbacks.

    Camera source can be changed at any time via reset() — the switch happens
    inside the capture thread so there is never more than one VideoCapture open
    simultaneously (avoids AVFoundation crashes on macOS).
    """

    def __init__(self, source: str | int = 0, fps: float = 5.0) -> None:
        self._source   = self._parse_source(source)
        self._interval = 1.0 / max(fps, 0.1)
        self._running  = False
        self._dirty    = False          # True = source changed, reopen cap
        self._thread: threading.Thread | None = None
        self._callbacks: list[Callable[[np.ndarray], None]] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def add_callback(self, cb: Callable[[np.ndarray], None]) -> None:
        self._callbacks.append(cb)

    def set_fps(self, fps: float) -> None:
        self._interval = 1.0 / max(fps, 0.1)

    def reset(self, source: str | int) -> None:
        """Queue a camera-source switch — no thread restart, no race condition."""
        self._source = self._parse_source(source)
        self._dirty  = True
        logger.info("camera source queued: %s (switches on next frame)", self._source)

    def start(self) -> None:
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="camera-loop"
        )
        self._thread.start()
        logger.info("camera loop started (source=%s  fps=%.1f)",
                    self._source, 1.0 / self._interval)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=6)

    # ── Capture loop (single thread owns VideoCapture) ────────────────────────

    def _loop(self) -> None:
        cap: cv2.VideoCapture | None = None
        active_source = None

        try:
            while self._running:
                # Reopen when source changed or first run
                if cap is None or self._dirty or active_source != self._source:
                    if cap is not None:
                        cap.release()
                        cap = None
                        time.sleep(0.15)          # short pause for driver cleanup

                    active_source = self._source
                    self._dirty   = False
                    logger.info("opening camera: %s", active_source)

                    cap = _open_capture(active_source)
                    if not cap.isOpened():
                        cap.release()
                        cap = None
                        logger.error("cannot open camera: %s", active_source)
                        time.sleep(1.0)
                        continue

                    # Verify camera delivers actual frames (some cameras open but give nothing)
                    ok, _ = cap.read()
                    if not ok:
                        cap.release()
                        cap = None
                        logger.error("camera opened but returned no frames: %s", active_source)
                        time.sleep(1.0)
                        continue

                    logger.info("camera ready: %dx%d",
                                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

                ret, frame = cap.read()
                if not ret:
                    # Loop video files; retry briefly for live cameras
                    if isinstance(active_source, str) and os.path.isfile(active_source):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    else:
                        logger.warning("failed to read frame, retrying")
                        time.sleep(0.4)
                    continue

                for cb in self._callbacks:
                    try:
                        cb(frame)
                    except Exception:
                        logger.exception("callback error")

                time.sleep(self._interval)

        finally:
            if cap is not None:
                cap.release()
            logger.info("camera released")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_source(source: str | int) -> str | int:
        try:
            return int(source)
        except (TypeError, ValueError):
            return source
