"""
MediaPipe Face Mesh + InsightFace ArcFace hybrid backend.

Detection / landmarks / iris gaze  →  MediaPipe FaceMesh (478 pts, refine_landmarks=True)
Embedding / recognition             →  InsightFace ArcFace R50 (buffalo_l)

The iris landmark indices (468, 473) give a pixel-accurate gaze score
without any separate eye-tracking model.
"""

import logging
import threading
import numpy as np

from .base import DetectedFace

logger = logging.getLogger(__name__)

# ── MediaPipe landmark indices ────────────────────────────────────────────────
_LEFT_IRIS   = 468   # iris centre — only with refine_landmarks=True
_RIGHT_IRIS  = 473
_L_EYE_OUT   = 33    # temporal corner of left eye
_L_EYE_IN    = 133   # nasal corner of left eye
_R_EYE_IN    = 362   # nasal corner of right eye
_R_EYE_OUT   = 263   # temporal corner of right eye
_NOSE_TIP    = 1
_MOUTH_L     = 61
_MOUTH_R     = 291


class _ISFaceProxy(dict):
    """
    Minimal duck-type of insightface.app.common.Face.
    Lets us call rec_model.get(img, face) without importing internal classes.
    """
    def __init__(self, kps5: np.ndarray):
        super().__init__()
        self["kps"] = kps5

    @property
    def kps(self) -> np.ndarray:
        return self["kps"]

    @property
    def embedding(self) -> np.ndarray | None:
        return self.get("embedding")

    @embedding.setter
    def embedding(self, v: np.ndarray) -> None:
        self["embedding"] = v


class MediaPipeBackend:
    """
    Set FACE_BACKEND=mediapipe to use this backend.

    install:  pip install mediapipe>=0.10.14
    """

    def __init__(self) -> None:
        self._lock      = threading.Lock()
        self._face_mesh = None
        self._rec_model  = None

    def start(self) -> None:
        import mediapipe as mp
        from insightface.app import FaceAnalysis

        logger.info("loading MediaPipe FaceMesh (iris landmarks)")
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=10,
            refine_landmarks=True,      # enables iris indices 468-477
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        )

        logger.info("loading InsightFace ArcFace R50 (recognition only)")
        app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["recognition"],
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0)
        self._rec_model = app.models.get("recognition")
        if self._rec_model is None:
            raise RuntimeError("InsightFace recognition model not loaded")

        logger.info("MediaPipeBackend ready")

    def stop(self) -> None:
        with self._lock:
            if self._face_mesh:
                self._face_mesh.close()
            self._face_mesh = None
            self._rec_model  = None

    def process(self, frame: np.ndarray) -> list[DetectedFace]:
        import cv2
        with self._lock:
            if self._face_mesh is None:
                return []
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return []

        h, w = frame.shape[:2]
        faces: list[DetectedFace] = []

        for face_lms in results.multi_face_landmarks:
            lms = face_lms.landmark

            # ── Bounding box ───────────────────────────────────────────────
            xs = [lm.x * w for lm in lms]
            ys = [lm.y * h for lm in lms]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

            # ── 5-point keypoints (ArcFace-compatible order) ───────────────
            kps5 = np.array([
                [lms[_LEFT_IRIS].x  * w, lms[_LEFT_IRIS].y  * h],
                [lms[_RIGHT_IRIS].x * w, lms[_RIGHT_IRIS].y * h],
                [lms[_NOSE_TIP].x   * w, lms[_NOSE_TIP].y   * h],
                [lms[_MOUTH_L].x    * w, lms[_MOUTH_L].y    * h],
                [lms[_MOUTH_R].x    * w, lms[_MOUTH_R].y    * h],
            ], dtype=np.float32)

            # ── Iris gaze score ────────────────────────────────────────────
            gaze = _iris_gaze_score(lms)

            # ── Embedding via ArcFace R50 (uses kps5 for alignment) ────────
            embedding = self._get_embedding(frame, kps5)
            if embedding is None:
                continue

            faces.append(DetectedFace(
                bbox=[x1, y1, x2, y2],
                embedding=embedding,
                confidence=0.92,            # MediaPipe has no raw det score
                kps=kps5.tolist(),
                gaze_score_override=gaze,
            ))

        return faces

    def _get_embedding(self, frame: np.ndarray, kps5: np.ndarray) -> list[float] | None:
        """
        Align the face to 112×112 using ArcFace standard keypoints via
        InsightFace's internal alignment, then run the recognition model.
        """
        try:
            face_proxy = _ISFaceProxy(kps5)
            self._rec_model.get(frame, face_proxy)
            if face_proxy.embedding is None:
                return None
            return face_proxy.embedding.tolist()
        except Exception as e:
            logger.debug("embedding failed: %s", e)
            return None


def _iris_gaze_score(lms) -> float:
    """
    Returns [0, 1].  1.0 = iris centred in eye socket = looking at camera.

    For each eye: compute how far the iris centre is from the midpoint of
    the eye's horizontal span.  Average both eyes.
    """
    def _eye(iris_idx: int, outer_idx: int, inner_idx: int) -> float:
        iris_x  = lms[iris_idx].x
        outer_x = lms[outer_idx].x
        inner_x = lms[inner_idx].x
        span    = abs(outer_x - inner_x)
        if span < 1e-6:
            return 0.5
        pos = (iris_x - min(outer_x, inner_x)) / span
        return max(0.0, 1.0 - 2.0 * abs(pos - 0.5))

    left  = _eye(_LEFT_IRIS,  _L_EYE_OUT, _L_EYE_IN)
    right = _eye(_RIGHT_IRIS, _R_EYE_OUT, _R_EYE_IN)
    return (left + right) / 2.0
