import base64
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field

import cv2
import httpx
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI

from camera import CameraLoop
from backend.base import DetectedFace

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

FACE_BACKEND   = os.getenv("FACE_BACKEND", "onnx")
API_URL        = os.getenv("API_SERVICE_URL", "http://localhost:8080")
CAMERA_SOURCE  = os.getenv("CAMERA_SOURCE", "0")
CAMERA_FPS     = float(os.getenv("CAMERA_FPS", "5"))
THUMB_PADDING  = 0.25


# ─── Detection config (kept in sync with api-service settings) ────────────────

@dataclass
class DetectionConfig:
    min_confidence: float   = 0.50
    camera_fps: float       = 5.0
    min_face_size_px: int   = 60
    require_both_eyes: bool = False
    max_yaw_deg: float      = 90.0   # 90 = disabled
    max_pitch_deg: float    = 90.0   # 90 = disabled

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def update_from_api(self, data: dict) -> None:
        with self._lock:
            self.min_confidence   = float(data.get("min_confidence",   self.min_confidence))
            self.camera_fps       = float(data.get("camera_fps",       self.camera_fps))
            self.min_face_size_px = int(  data.get("min_face_size_px", self.min_face_size_px))
            self.require_both_eyes= bool( data.get("require_both_eyes",self.require_both_eyes))
            self.max_yaw_deg      = float(data.get("max_yaw_deg",      self.max_yaw_deg))
            self.max_pitch_deg    = float(data.get("max_pitch_deg",    self.max_pitch_deg))

    def snapshot(self) -> "DetectionConfig":
        with self._lock:
            return DetectionConfig(
                min_confidence   = self.min_confidence,
                camera_fps       = self.camera_fps,
                min_face_size_px = self.min_face_size_px,
                require_both_eyes= self.require_both_eyes,
                max_yaw_deg      = self.max_yaw_deg,
                max_pitch_deg    = self.max_pitch_deg,
            )


cfg = DetectionConfig(camera_fps=CAMERA_FPS)


# ─── Face filtering helpers ───────────────────────────────────────────────────

def face_width_px(face: DetectedFace) -> float:
    return face.bbox[2] - face.bbox[0]


def has_both_eyes(face: DetectedFace) -> bool:
    """Check both eye keypoints are present and spread apart enough."""
    kps = getattr(face, "kps", None)
    if kps is None or len(kps) < 2:
        return False
    left_eye  = np.array(kps[0])
    right_eye = np.array(kps[1])
    eye_dist  = float(np.linalg.norm(right_eye - left_eye))
    face_w    = face_width_px(face)
    # Both eyes must be at least 10% of face width apart
    return eye_dist > face_w * 0.10


def estimate_yaw(face: DetectedFace) -> float:
    """Rough horizontal rotation in degrees derived from keypoints."""
    kps = getattr(face, "kps", None)
    if kps is None or len(kps) < 3:
        return 0.0
    left_eye, right_eye, nose = np.array(kps[0]), np.array(kps[1]), np.array(kps[2])
    eye_span = abs(float(right_eye[0]) - float(left_eye[0]))
    if eye_span < 1:
        return 0.0
    eye_center_x = (float(left_eye[0]) + float(right_eye[0])) / 2
    # Nose offset relative to eye span; empirically maps to ~yaw in degrees
    ratio = (float(nose[0]) - eye_center_x) / eye_span
    return abs(ratio) * 90.0


def estimate_pitch(face: DetectedFace) -> float:
    """Rough vertical rotation in degrees derived from keypoints."""
    kps = getattr(face, "kps", None)
    if kps is None or len(kps) < 3:
        return 0.0
    left_eye, right_eye, nose = np.array(kps[0]), np.array(kps[1]), np.array(kps[2])
    eye_span = abs(float(right_eye[0]) - float(left_eye[0]))
    if eye_span < 1:
        return 0.0
    eye_center_y = (float(left_eye[1]) + float(right_eye[1])) / 2
    # Typical frontal ratio is ~0.7; deviation maps to pitch
    ratio = (float(nose[1]) - eye_center_y) / eye_span
    return abs(ratio - 0.7) * 90.0


def passes_filters(face: DetectedFace, c: DetectionConfig) -> tuple[bool, str]:
    """Return (ok, reason_if_rejected)."""
    if face.confidence < c.min_confidence:
        return False, f"confidence {face.confidence:.2f} < {c.min_confidence}"
    if face_width_px(face) < c.min_face_size_px:
        return False, f"face too small ({face_width_px(face):.0f}px < {c.min_face_size_px}px)"
    if c.require_both_eyes and not has_both_eyes(face):
        return False, "both eyes not visible"
    yaw = estimate_yaw(face)
    if yaw > c.max_yaw_deg:
        return False, f"yaw {yaw:.0f}° > {c.max_yaw_deg}°"
    pitch = estimate_pitch(face)
    if pitch > c.max_pitch_deg:
        return False, f"pitch {pitch:.0f}° > {c.max_pitch_deg}°"
    return True, ""


# ─── Crop helper ──────────────────────────────────────────────────────────────

def crop_face(frame: np.ndarray, bbox: list[float]) -> str:
    x1, y1, x2, y2 = [int(c) for c in bbox]
    w, h = x2 - x1, y2 - y1
    pad = int(max(w, h) * THUMB_PADDING)
    x1 = max(0, x1 - pad);  y1 = max(0, y1 - pad)
    x2 = min(frame.shape[1], x2 + pad);  y2 = min(frame.shape[0], y2 + pad)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return ""
    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode()


# ─── Services ─────────────────────────────────────────────────────────────────

def build_backend():
    if FACE_BACKEND == "hailo":
        from backend.hailo_backend import HailoBackend
        return HailoBackend(
            scrfd_hef  = os.getenv("HAILO_SCRFD_HEF",   "/models/scrfd_10g.hef"),
            arcface_hef= os.getenv("HAILO_ARCFACE_HEF", "/models/arcface_r50.hef"),
        )
    from backend.onnx_backend import OnnxBackend
    return OnnxBackend()


backend = build_backend()
camera  = CameraLoop(source=CAMERA_SOURCE, fps=CAMERA_FPS)
client  = httpx.Client(base_url=API_URL, timeout=5.0)


# ─── Settings polling thread ──────────────────────────────────────────────────

def _settings_poller():
    while True:
        time.sleep(30)
        try:
            r = client.get("/api/settings")
            if r.status_code == 200:
                data = r.json()
                old_fps = cfg.camera_fps
                cfg.update_from_api(data)
                if abs(cfg.camera_fps - old_fps) > 0.1:
                    camera.set_fps(cfg.camera_fps)
                    logger.info("FPS updated to %.1f", cfg.camera_fps)
        except Exception as e:
            logger.debug("settings poll: %s", e)


# ─── Frame callback ──────────────────────────────────────────────────────────

def on_frame(frame: np.ndarray) -> None:
    c = cfg.snapshot()
    faces: list[DetectedFace] = backend.process(frame)
    for face in faces:
        ok, reason = passes_filters(face, c)
        if not ok:
            logger.debug("face filtered: %s", reason)
            continue
        thumbnail = crop_face(frame, face.bbox)
        try:
            client.post("/api/appearance", json={
                "embedding":  face.embedding,
                "bbox":       face.bbox,
                "confidence": face.confidence,
                "thumbnail":  thumbnail,
            })
        except httpx.RequestError as e:
            logger.warning("api unreachable: %s", e)


# ─── App lifecycle ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fetch settings once on startup
    try:
        r = client.get("/api/settings")
        if r.status_code == 200:
            cfg.update_from_api(r.json())
            camera.set_fps(cfg.camera_fps)
            logger.info("initial settings loaded from API")
    except Exception:
        logger.warning("could not fetch initial settings, using defaults")

    backend.start()
    camera.add_callback(on_frame)
    camera.start()

    t = threading.Thread(target=_settings_poller, daemon=True, name="settings-poller")
    t.start()

    yield
    camera.stop()
    backend.stop()
    client.close()


app = FastAPI(title="FacePulse face-service", lifespan=lifespan)


@app.get("/health")
def health():
    c = cfg.snapshot()
    return {
        "status":          "ok",
        "backend":         FACE_BACKEND,
        "min_confidence":  c.min_confidence,
        "require_eyes":    c.require_both_eyes,
        "max_yaw_deg":     c.max_yaw_deg,
        "max_pitch_deg":   c.max_pitch_deg,
        "min_face_size_px":c.min_face_size_px,
    }
