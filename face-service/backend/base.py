from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable
import numpy as np


@dataclass
class DetectedFace:
    bbox: list[float]
    embedding: list[float]
    confidence: float
    kps: list | None = None                  # 5 keypoints: left_eye, right_eye, nose, left_mouth, right_mouth
    gaze_score_override: float | None = None # set by MediaPipe (iris-based); None = use geometric fallback


@runtime_checkable
class FaceBackend(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def process(self, frame: np.ndarray) -> list[DetectedFace]: ...
