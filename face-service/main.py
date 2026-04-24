import base64
import logging
import os
import time

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

FACE_BACKEND    = os.getenv("FACE_BACKEND", "onnx")
API_URL         = os.getenv("API_SERVICE_URL", "http://localhost:8080")
CAMERA_SOURCE   = os.getenv("CAMERA_SOURCE", "0")
CAMERA_FPS      = float(os.getenv("CAMERA_FPS", "5"))
MIN_CONFIDENCE  = float(os.getenv("MIN_CONFIDENCE", "0.5"))
THUMB_PADDING   = 0.25   # fraction of face size added as border


def build_backend():
    if FACE_BACKEND == "hailo":
        from backend.hailo_backend import HailoBackend
        return HailoBackend(
            scrfd_hef=os.getenv("HAILO_SCRFD_HEF", "/models/scrfd_10g.hef"),
            arcface_hef=os.getenv("HAILO_ARCFACE_HEF", "/models/arcface_r50.hef"),
        )
    from backend.onnx_backend import OnnxBackend
    return OnnxBackend()


def crop_face(frame: np.ndarray, bbox: list[float]) -> str:
    """Crop face with padding, encode as base64 JPEG."""
    x1, y1, x2, y2 = [int(c) for c in bbox]
    w, h = x2 - x1, y2 - y1
    pad = int(max(w, h) * THUMB_PADDING)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(frame.shape[1], x2 + pad)
    y2 = min(frame.shape[0], y2 + pad)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return ""
    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode()


backend = build_backend()
camera  = CameraLoop(source=CAMERA_SOURCE, fps=CAMERA_FPS)
client  = httpx.Client(base_url=API_URL, timeout=5.0)


def on_frame(frame: np.ndarray) -> None:
    faces: list[DetectedFace] = backend.process(frame)
    for face in faces:
        if face.confidence < MIN_CONFIDENCE:
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    backend.start()
    camera.add_callback(on_frame)
    camera.start()
    yield
    camera.stop()
    backend.stop()
    client.close()


app = FastAPI(title="FacePulse face-service", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "backend": FACE_BACKEND}
