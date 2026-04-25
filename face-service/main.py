import base64
import dataclasses
import glob
import logging
import math
import os
import platform
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Flag, auto

import cv2
import httpx
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

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
    require_gaze: bool      = False
    gaze_threshold: float   = 0.80
    camera_source: str      = "0"
    face_backend: str       = "onnx"   # "onnx" | "mediapipe" | "hailo"
    frame_rotate: int       = 0        # 0 | 90 | 180 | 270 degrees clockwise

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def update_from_api(self, data: dict) -> None:
        with self._lock:
            self.min_confidence   = float(data.get("min_confidence",   self.min_confidence))
            self.camera_fps       = float(data.get("camera_fps",       self.camera_fps))
            self.min_face_size_px = int(  data.get("min_face_size_px", self.min_face_size_px))
            self.require_both_eyes= bool( data.get("require_both_eyes",self.require_both_eyes))
            self.max_yaw_deg      = float(data.get("max_yaw_deg",      self.max_yaw_deg))
            self.max_pitch_deg    = float(data.get("max_pitch_deg",    self.max_pitch_deg))
            self.require_gaze     = bool( data.get("require_gaze",     self.require_gaze))
            self.gaze_threshold   = float(data.get("gaze_threshold",   self.gaze_threshold))
            self.camera_source    = str(  data.get("camera_source",    self.camera_source))
            self.face_backend     = str(  data.get("face_backend",     self.face_backend))
            self.frame_rotate     = int(  data.get("frame_rotate",     self.frame_rotate))

    def snapshot(self) -> "DetectionConfig":
        with self._lock:
            return DetectionConfig(
                min_confidence   = self.min_confidence,
                camera_fps       = self.camera_fps,
                min_face_size_px = self.min_face_size_px,
                require_both_eyes= self.require_both_eyes,
                max_yaw_deg      = self.max_yaw_deg,
                max_pitch_deg    = self.max_pitch_deg,
                require_gaze     = self.require_gaze,
                gaze_threshold   = self.gaze_threshold,
                camera_source    = self.camera_source,
                face_backend     = self.face_backend,
                frame_rotate     = self.frame_rotate,
            )


cfg = DetectionConfig(camera_fps=CAMERA_FPS)


# ─── Face filtering helpers ───────────────────────────────────────────────────

def face_width_px(face: DetectedFace) -> float:
    return face.bbox[2] - face.bbox[0]


def has_both_eyes(face: DetectedFace) -> bool:
    """
    True only when the face is plausibly frontal.

    Two geometric conditions must both hold:
    1. The nose x-coordinate sits BETWEEN the two eye x-coordinates.
       InsightFace predicts the hidden eye in profiles, but the nose is still
       at the tip of the nose — outside the eye x-range for strong profiles.
    2. Eye-to-eye distance is at least 22 % of face width.
       Frontal faces: ~30-45 %.  Profiles: often < 15 %.
    """
    kps = getattr(face, "kps", None)
    if kps is None or len(kps) < 3:
        return False
    lx = float(kps[0][0])
    rx = float(kps[1][0])
    nx = float(kps[2][0])

    # Nose must be horizontally bracketed by the two eyes
    lo, hi = min(lx, rx), max(lx, rx)
    if not (lo < nx < hi):
        return False

    left_eye  = np.array(kps[0])
    right_eye = np.array(kps[1])
    eye_dist  = float(np.linalg.norm(right_eye - left_eye))
    return eye_dist > face_width_px(face) * 0.22


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


def gaze_score(face: DetectedFace) -> float:
    """
    Combined frontal-gaze score in [0, 1].  1.0 = looking straight at camera.

    Two components multiplied together:
    - Horizontal symmetry: nose equidistant from both eyes.
      Profiles fail this because the nose is much closer to one eye.
    - Vertical symmetry: both eyes at the same height (catches head roll /
      side tilt that can fool horizontal symmetry alone).
      Score = 1 - clamp(Δy_eyes / eye_dist, 0, 1).

    The product is strict: both components must be high simultaneously.
    """
    kps = getattr(face, "kps", None)
    if kps is None or len(kps) < 3:
        return 0.0

    left_eye  = np.array(kps[0], dtype=float)
    right_eye = np.array(kps[1], dtype=float)
    nose      = np.array(kps[2], dtype=float)

    # ── Horizontal symmetry ──────────────────────────────────────────────────
    d_left  = float(np.linalg.norm(nose - left_eye))
    d_right = float(np.linalg.norm(nose - right_eye))
    if d_left + d_right < 1.0:
        return 0.0
    h_sym = min(d_left, d_right) / max(d_left, d_right)

    # ── Vertical symmetry (eye-height parity) ─────────────────────────────
    eye_dist = float(np.linalg.norm(right_eye - left_eye))
    if eye_dist < 1.0:
        return h_sym  # can't compute, fall back to h_sym only
    v_diff = abs(float(left_eye[1]) - float(right_eye[1])) / eye_dist
    # v_diff ≈ 0 for frontal, > 0.5 for 45°+ tilt → zero the score
    v_sym = max(0.0, 1.0 - v_diff * 2.0)

    return h_sym * v_sym


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


class FilterBlock(Flag):
    """Bitmask of active filter rejections. NONE means the face passed."""
    NONE       = 0
    CONFIDENCE = auto()
    SIZE       = auto()
    EYES       = auto()
    YAW        = auto()
    PITCH      = auto()
    GAZE       = auto()

    def reason(self) -> str:
        labels = {
            FilterBlock.CONFIDENCE: "low confidence",
            FilterBlock.SIZE:       "face too small",
            FilterBlock.EYES:       "profile / eyes not frontal",
            FilterBlock.YAW:        "yaw too large",
            FilterBlock.PITCH:      "pitch too large",
            FilterBlock.GAZE:       "not looking at camera",
        }
        parts = [v for k, v in labels.items() if k in self]
        return " + ".join(parts) if parts else "pass"


def passes_filters(face: DetectedFace, c: DetectionConfig) -> tuple[bool, str]:
    """Return (passed, reason_string) using FilterBlock bitmask internally."""
    blocked = FilterBlock.NONE

    if face.confidence < c.min_confidence:
        blocked |= FilterBlock.CONFIDENCE
    if face_width_px(face) < c.min_face_size_px:
        blocked |= FilterBlock.SIZE
    if c.require_both_eyes and not has_both_eyes(face):
        blocked |= FilterBlock.EYES

    yaw = estimate_yaw(face)
    if yaw > c.max_yaw_deg:
        blocked |= FilterBlock.YAW

    pitch = estimate_pitch(face)
    if pitch > c.max_pitch_deg:
        blocked |= FilterBlock.PITCH

    if c.require_gaze:
        # Prefer iris-based score (MediaPipe) over geometric approximation
        score = (face.gaze_score_override
                 if face.gaze_score_override is not None
                 else gaze_score(face))
        if score < c.gaze_threshold:
            blocked |= FilterBlock.GAZE

    return blocked == FilterBlock.NONE, blocked.reason()


def log_face_scores(face: DetectedFace) -> None:
    """Log all filter scores at DEBUG level — useful for tuning thresholds."""
    kps = getattr(face, "kps", None)
    eyes_ok  = has_both_eyes(face)
    g_score  = gaze_score(face)
    yaw      = estimate_yaw(face)
    pitch    = estimate_pitch(face)
    logger.debug(
        "face w=%.0fpx conf=%.2f eyes=%s yaw=%.0f° pitch=%.0f° gaze=%.2f",
        face_width_px(face), face.confidence, eyes_ok, yaw, pitch, g_score,
    )


# ─── Debug state ─────────────────────────────────────────────────────────────

@dataclasses.dataclass
class _DebugDet:
    face:    DetectedFace
    passed:  bool
    reason:  str
    gaze:    float
    yaw:     float
    pitch:   float
    eyes_ok: bool

_dbg_lock  = threading.Lock()
_dbg_frame: np.ndarray | None = None
_dbg_dets:  list[_DebugDet]   = []

# Keypoint colours and short labels (InsightFace 5-pt order)
_KPS_STYLE = [
    ((0,   230, 230), "LE"),   # left eye   – cyan
    ((0,   230, 100), "RE"),   # right eye  – green
    ((0,   165, 255), "N"),    # nose       – orange
    ((220,   0, 220), "LM"),   # left mouth – magenta
    ((100,  80, 255), "RM"),   # right mouth– purple
]


def _txt(img, text, pos, scale=0.42, color=(255,255,255), thick=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color,  thick,    cv2.LINE_AA)


def draw_debug_frame(frame: np.ndarray, dets: list[_DebugDet]) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]

    for d in dets:
        clr = (30, 220, 30) if d.passed else (30, 30, 220)   # BGR: green / red

        # ── Bounding box ──────────────────────────────────────────────────
        x1, y1, x2, y2 = [int(c) for c in d.face.bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), clr, 2)

        # ── Keypoints ─────────────────────────────────────────────────────
        kps = d.face.kps or []
        for i, (pt, (kclr, klbl)) in enumerate(zip(kps, _KPS_STYLE)):
            px, py = int(pt[0]), int(pt[1])
            cv2.circle(img, (px, py), 5, kclr, -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), 5, (0,0,0), 1,  cv2.LINE_AA)
            _txt(img, klbl, (px + 6, py - 4), 0.35, kclr)

        # Draw line connecting eyes
        if len(kps) >= 2:
            le = (int(kps[0][0]), int(kps[0][1]))
            re = (int(kps[1][0]), int(kps[1][1]))
            cv2.line(img, le, re, (200, 200, 0), 1, cv2.LINE_AA)

        # ── Score labels above bbox ───────────────────────────────────────
        lines = [
            f"conf {d.face.confidence:.2f}  w {d.face.bbox[2]-d.face.bbox[0]:.0f}px",
            f"gaze {d.gaze:.2f}  yaw {d.yaw:.0f}°  pitch {d.pitch:.0f}°",
            f"eyes {'OK' if d.eyes_ok else 'FAIL'}",
        ]
        if not d.passed:
            lines.append(f"❌ {d.reason[:36]}")

        line_h = 15
        y_start = max(y1 - len(lines) * line_h - 4, 2)
        for i, ln in enumerate(lines):
            (tw, th), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            yp = y_start + i * line_h + th
            cv2.rectangle(img, (x1, yp - th - 1), (x1 + tw + 3, yp + 2), (0,0,0), -1)
            _txt(img, ln, (x1 + 2, yp), 0.42, clr if i < 3 else (80, 80, 255))

    # ── Frame overlay (bottom-left) ───────────────────────────────────────
    ts = time.strftime("%H:%M:%S")
    _txt(img, f"{w}×{h}  {ts}  faces:{len(dets)}", (6, h - 8), 0.42, (180,180,180))

    return img


# ─── Camera enumeration ───────────────────────────────────────────────────────

def list_cameras() -> list[dict]:
    """
    List available camera devices.

    macOS  → ffmpeg AVFoundation (real device names) + OpenCV fallback
    Linux  → /dev/video* + OpenCV integer indices
    Other  → OpenCV integer indices 0-9
    """
    found: dict[str, dict] = {}
    sys = platform.system()

    if sys == "Darwin":
        _scan_avfoundation(found)   # ffmpeg listing with real names
    elif sys == "Linux":
        _scan_v4l(found)            # /dev/video*

    # OpenCV fallback: only try indices not yet found, up to max_found+2 or 4
    known_int = [int(k) for k in found if str(k).isdigit()]
    limit = max(known_int) + 2 if known_int else 4
    _scan_opencv_indices(found, limit=limit)

    return sorted(found.values(), key=lambda c: (
        int(c["source"]) if str(c["source"]).isdigit() else 999
    ))


def _scan_opencv_indices(found: dict, limit: int = 10) -> None:
    for idx in range(limit):
        key = str(idx)
        if key in found:
            continue
        # On macOS always use AVFoundation backend so results match camera.py
        if platform.system() == "Darwin":
            cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        else:
            cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            found[key] = {"source": key, "label": f"Camera {idx} ({w}×{h})", "width": w, "height": h}
        else:
            cap.release()


def _scan_avfoundation(found: dict) -> None:
    """macOS: use ffmpeg -f avfoundation to list all AVFoundation video devices."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True, text=True, timeout=8,
        )
        in_video = False
        for line in r.stderr.splitlines():
            if "AVFoundation video devices" in line:
                in_video = True
                continue
            if "AVFoundation audio" in line:
                in_video = False
                continue
            if not in_video:
                continue
            m = re.search(r"\[(\d+)\]\s+(.+)", line)
            if not m:
                continue
            idx, name = m.group(1), m.group(2).strip()
            cap = cv2.VideoCapture(int(idx), cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                found[idx] = {"source": idx, "label": f"[{idx}] {name} ({w}×{h})", "width": w, "height": h}
            else:
                cap.release()
                # Include even if we can't open — user might want to try
                found[idx] = {"source": idx, "label": f"[{idx}] {name} (unavailable)", "width": 0, "height": 0}
    except FileNotFoundError:
        logger.debug("ffmpeg not found — falling back to OpenCV index scan")
    except subprocess.TimeoutExpired:
        logger.debug("ffmpeg device listing timed out")


def _scan_v4l(found: dict) -> None:
    """Linux: scan /dev/video* nodes."""
    for dev in sorted(glob.glob("/dev/video*")):
        if dev in found:
            continue
        cap = cv2.VideoCapture(dev)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            found[dev] = {"source": dev, "label": f"{dev} ({w}×{h})", "width": w, "height": h}
        else:
            cap.release()


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

def _create_backend(name: str):
    if name == "hailo":
        from backend.hailo_backend import HailoBackend
        return HailoBackend(
            scrfd_hef  = os.getenv("HAILO_SCRFD_HEF",   "/models/scrfd_10g.hef"),
            arcface_hef= os.getenv("HAILO_ARCFACE_HEF", "/models/arcface_r50.hef"),
        )
    if name == "mediapipe":
        from backend.mediapipe_backend import MediaPipeBackend
        return MediaPipeBackend()
    from backend.onnx_backend import OnnxBackend
    return OnnxBackend()


class BackendProxy:
    """Thread-safe wrapper that allows hot-swapping the face backend at runtime."""

    def __init__(self, initial: str) -> None:
        self._lock    = threading.Lock()
        self._backend = _create_backend(initial)
        self._name    = initial

    def switch(self, name: str) -> None:
        with self._lock:
            logger.info("switching backend: %s → %s", self._name, name)
            self._backend.stop()
            self._backend = _create_backend(name)
            self._backend.start()
            self._name = name

    def start(self) -> None:
        with self._lock:
            self._backend.start()

    def stop(self) -> None:
        with self._lock:
            self._backend.stop()

    def process(self, frame: np.ndarray) -> list[DetectedFace]:
        with self._lock:
            return self._backend.process(frame)

    @property
    def name(self) -> str:
        return self._name


backend = BackendProxy(FACE_BACKEND)
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
                old_fps    = cfg.camera_fps
                old_source  = cfg.camera_source
                old_backend = cfg.face_backend
                cfg.update_from_api(data)
                if abs(cfg.camera_fps - old_fps) > 0.1:
                    camera.set_fps(cfg.camera_fps)
                    logger.info("FPS updated to %.1f", cfg.camera_fps)
                if cfg.camera_source != old_source:
                    logger.info("camera source changed: %s → %s", old_source, cfg.camera_source)
                    camera.reset(cfg.camera_source)
                if cfg.face_backend != old_backend:
                    backend.switch(cfg.face_backend)
        except Exception as e:
            logger.debug("settings poll: %s", e)


# ─── Frame callback ──────────────────────────────────────────────────────────

_ROTATE_CODE = {
    90:  cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def on_frame(frame: np.ndarray) -> None:
    c = cfg.snapshot()
    if c.frame_rotate and c.frame_rotate in _ROTATE_CODE:
        frame = cv2.rotate(frame, _ROTATE_CODE[c.frame_rotate])
    faces: list[DetectedFace] = backend.process(frame)

    dets: list[_DebugDet] = []
    for face in faces:
        log_face_scores(face)
        ok, reason = passes_filters(face, c)
        dets.append(_DebugDet(
            face=face, passed=ok, reason=reason,
            gaze=gaze_score(face), yaw=estimate_yaw(face),
            pitch=estimate_pitch(face), eyes_ok=has_both_eyes(face),
        ))
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

    with _dbg_lock:
        global _dbg_frame, _dbg_dets
        _dbg_frame = frame
        _dbg_dets  = dets


# ─── App lifecycle ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fetch settings once on startup
    try:
        r = client.get("/api/settings")
        if r.status_code == 200:
            cfg.update_from_api(r.json())
            camera.set_fps(cfg.camera_fps)
            camera._source = int(cfg.camera_source) if cfg.camera_source.isdigit() else cfg.camera_source
            logger.info("initial settings loaded: camera=%s fps=%.1f", cfg.camera_source, cfg.camera_fps)
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/debug/frame")
def debug_frame():
    with _dbg_lock:
        frame = _dbg_frame
        dets  = list(_dbg_dets)

    if frame is None:
        ph = np.zeros((360, 640, 3), dtype=np.uint8)
        _txt(ph, "No frame captured yet — is the camera running?",
             (40, 180), 0.6, (120, 120, 120))
        annotated = ph
    else:
        annotated = draw_debug_frame(frame, dets)

    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return Response(content=buf.tobytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "no-cache, no-store"})


@app.get("/cameras")
def cameras():
    return list_cameras()


@app.get("/health")
def health():
    c = cfg.snapshot()
    return {
        "status":           "ok",
        "backend":          FACE_BACKEND,
        "min_confidence":   c.min_confidence,
        "require_eyes":     c.require_both_eyes,
        "max_yaw_deg":      c.max_yaw_deg,
        "max_pitch_deg":    c.max_pitch_deg,
        "min_face_size_px": c.min_face_size_px,
        "require_gaze":     c.require_gaze,
        "gaze_threshold":   c.gaze_threshold,
    }
