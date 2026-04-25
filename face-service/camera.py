import logging
import os
import threading
import time
from typing import Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraLoop:
    """Captures frames from a webcam or video file and calls registered callbacks."""

    def __init__(self, source: str | int = 0, fps: float = 5.0):
        try:
            source = int(source)
        except (TypeError, ValueError):
            pass
        self._source = source
        self._interval = 1.0 / max(fps, 0.1)
        self._running = False
        self._thread: threading.Thread | None = None
        self._callbacks: list[Callable[[np.ndarray], None]] = []

    def set_fps(self, fps: float) -> None:
        self._interval = 1.0 / max(fps, 0.1)

    def reset(self, source: str | int) -> None:
        """Switch to a different camera source without losing callbacks."""
        logger.info("switching camera source → %s", source)
        self.stop()
        try:
            self._source = int(source)
        except (TypeError, ValueError):
            self._source = source
        self.start()

    def add_callback(self, cb: Callable[[np.ndarray], None]) -> None:
        self._callbacks.append(cb)

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="camera-loop")
        self._thread.start()
        logger.info("camera loop started (source=%s, fps=%.1f)", self._source, 1 / self._interval)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            logger.error("cannot open camera source: %s", self._source)
            return

        logger.info("camera opened: %dx%d", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(self._source, str):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    logger.warning("failed to read frame, retrying")
                    time.sleep(0.5)
                    continue

                for cb in self._callbacks:
                    try:
                        cb(frame)
                    except Exception:
                        logger.exception("callback error")

                time.sleep(self._interval)
        finally:
            cap.release()
            logger.info("camera released")
