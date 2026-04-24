import logging
import numpy as np
from .base import DetectedFace

logger = logging.getLogger(__name__)


class OnnxBackend:
    """
    InsightFace + ONNX Runtime backend. Works on macOS/Linux/Windows CPU.
    Downloads buffalo_l model on first run (~300MB).
    """

    def __init__(self, model_name: str = "buffalo_l", det_size: tuple = (640, 640)):
        self._model_name = model_name
        self._det_size = det_size
        self._app = None

    def start(self) -> None:
        import insightface
        from insightface.app import FaceAnalysis

        logger.info("loading InsightFace model '%s'", self._model_name)
        self._app = FaceAnalysis(
            name=self._model_name,
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=0, det_size=self._det_size)
        logger.info("OnnxBackend ready")

    def stop(self) -> None:
        self._app = None

    def process(self, frame: np.ndarray) -> list[DetectedFace]:
        if self._app is None:
            return []
        faces = self._app.get(frame)
        return [
            DetectedFace(
                bbox=face.bbox.tolist(),
                embedding=face.embedding.tolist(),
                confidence=float(face.det_score),
                kps=face.kps.tolist() if face.kps is not None else None,
            )
            for face in faces
            if face.embedding is not None
        ]
