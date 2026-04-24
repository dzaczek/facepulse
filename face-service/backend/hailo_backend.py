"""
Hailo-10H backend for Raspberry Pi AI HAT+ (40 TOPS).

Requirements on RPi OS Bookworm:
    sudo apt install hailo-all        # HailoRT + firmware + Python bindings
    # Download models from Hailo Model Zoo:
    # scrfd_10g.hef   — face detection
    # arcface_r50.hef — face recognition (512-dim embeddings)

Set environment variables:
    HAILO_SCRFD_HEF=/models/scrfd_10g.hef
    HAILO_ARCFACE_HEF=/models/arcface_r50.hef
"""

import logging
import numpy as np
from .base import DetectedFace

logger = logging.getLogger(__name__)

_EMBED_DIM = 512


class HailoBackend:
    def __init__(self, scrfd_hef: str, arcface_hef: str):
        self._scrfd_hef = scrfd_hef
        self._arcface_hef = arcface_hef
        self._device = None
        self._scrfd = None
        self._arcface = None

    def start(self) -> None:
        from hailo_platform import HEF, VDevice, HailoStreamInterface, ConfigureParams  # noqa

        logger.info("initializing Hailo-10H device")
        self._device = VDevice()

        scrfd_hef = HEF(self._scrfd_hef)
        arcface_hef = HEF(self._arcface_hef)

        scrfd_params = ConfigureParams.create_from_hef(
            scrfd_hef, interface=HailoStreamInterface.PCIe
        )
        arcface_params = ConfigureParams.create_from_hef(
            arcface_hef, interface=HailoStreamInterface.PCIe
        )

        self._scrfd = self._device.configure(scrfd_hef, scrfd_params)[0]
        self._arcface = self._device.configure(arcface_hef, arcface_params)[0]
        logger.info("HailoBackend ready (SCRFD + ArcFace on Hailo-10H)")

    def stop(self) -> None:
        self._scrfd = None
        self._arcface = None
        self._device = None

    def process(self, frame: np.ndarray) -> list[DetectedFace]:
        from hailo_platform import InferVStreams, InputVStreamParams, OutputVStreamParams  # noqa

        h, w = frame.shape[:2]
        detections = self._run_scrfd(frame)
        results = []

        for det in detections:
            x1, y1, x2, y2, score = det
            face_crop = self._align_crop(frame, x1, y1, x2, y2)
            embedding = self._run_arcface(face_crop)
            results.append(DetectedFace(
                bbox=[float(x1), float(y1), float(x2), float(y2)],
                embedding=embedding,
                confidence=float(score),
            ))

        return results

    def _run_scrfd(self, frame: np.ndarray) -> list:
        """Run SCRFD face detection — returns list of [x1,y1,x2,y2,score]."""
        from hailo_platform import InferVStreams, InputVStreamParams, OutputVStreamParams

        input_data = self._preprocess_det(frame)
        with InferVStreams(
            self._scrfd,
            InputVStreamParams.make(self._scrfd),
            OutputVStreamParams.make(self._scrfd),
        ) as pipeline:
            results = {}
            pipeline.send({"input_layer1": input_data})
            pipeline.flush()
            for name, _ in pipeline.get_output_vstreams():
                results[name] = pipeline.recv(name)
        return self._postprocess_scrfd(results, frame.shape)

    def _run_arcface(self, face_img: np.ndarray) -> list[float]:
        """Run ArcFace embedding extraction."""
        from hailo_platform import InferVStreams, InputVStreamParams, OutputVStreamParams

        input_data = self._preprocess_rec(face_img)
        with InferVStreams(
            self._arcface,
            InputVStreamParams.make(self._arcface),
            OutputVStreamParams.make(self._arcface),
        ) as pipeline:
            pipeline.send({"input_layer1": input_data})
            pipeline.flush()
            emb = pipeline.recv("output_layer1")
        emb = emb.flatten().astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.tolist()

    @staticmethod
    def _preprocess_det(frame: np.ndarray) -> np.ndarray:
        import cv2
        img = cv2.resize(frame, (640, 640))
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)
        return img

    @staticmethod
    def _preprocess_rec(face: np.ndarray) -> np.ndarray:
        import cv2
        img = cv2.resize(face, (112, 112))
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    @staticmethod
    def _align_crop(frame: np.ndarray, x1, y1, x2, y2) -> np.ndarray:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _postprocess_scrfd(outputs: dict, shape: tuple) -> list:
        """Minimal NMS-free passthrough — SCRFD HEF outputs decoded boxes."""
        h, w = shape[:2]
        detections = []
        for v in outputs.values():
            arr = np.array(v).reshape(-1, 5)
            for row in arr:
                score = row[4]
                if score < 0.5:
                    continue
                x1, y1, x2, y2 = row[0] * w, row[1] * h, row[2] * w, row[3] * h
                detections.append([x1, y1, x2, y2, score])
        return detections
