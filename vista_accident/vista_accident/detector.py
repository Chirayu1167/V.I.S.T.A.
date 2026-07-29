"""
Thin wrapper around Ultralytics YOLOv8n (COCO-pretrained) restricted to the
classes we care about (vehicles + persons/cyclists). FP16 inference on GPU
per the "GPU-optimized inference" feature in VISTA.md.
"""

from typing import List, Tuple

import numpy as np

from .config import ALL_TRACKED_CLASSES


class Detector:
    def __init__(self, weights: str = "yolov8n.pt", device: str = "cuda", half: bool = True,
                 conf_threshold: float = 0.35, classes=None):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.device = device
        self.half = half
        self.conf_threshold = conf_threshold
        self.classes = list(classes) if classes is not None else list(ALL_TRACKED_CLASSES)

        # Default to GPU; fall back to CPU with a warning if unavailable.
        try:
            import torch
            if device == "cuda" and not torch.cuda.is_available():
                import warnings
                warnings.warn("CUDA not available. Falling back to CPU (inference will be slower).")
                self.device = "cpu"
        except ImportError:
            self.device = "cpu"

        # FP16 half-precision only makes sense (and is only supported) on CUDA.
        if self.device != "cuda":
            self.half = False

    def detect(self, frame: np.ndarray) -> List[Tuple[Tuple[float, float, float, float], float, int]]:
        """Returns list of (bbox_xyxy, confidence, class_id)."""
        results = self.model.predict(
            frame,
            device=self.device,
            half=self.half,
            conf=self.conf_threshold,
            classes=self.classes,
            verbose=False,
        )
        out = []
        if not results:
            return out
        boxes = results[0].boxes
        if boxes is None:
            return out
        for box in boxes:
            xyxy = tuple(box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            out.append((xyxy, conf, cls))
        return out
