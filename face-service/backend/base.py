from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable
import numpy as np


@dataclass
class DetectedFace:
    bbox: list[float]
    embedding: list[float]
    confidence: float


@runtime_checkable
class FaceBackend(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def process(self, frame: np.ndarray) -> list[DetectedFace]: ...
