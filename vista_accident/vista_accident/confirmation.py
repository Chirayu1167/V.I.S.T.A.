"""
Secondary ML confirmation, run ONLY on frames the heuristic layer already
flagged (not every frame) — this is a confidence boost, not the primary
detector, so it's fine for it to be heavier.

Model: Enos-123/traffic-accident-detection-yolo11x on HuggingFace
  https://huggingface.co/Enos-123/traffic-accident-detection-yolo11x

This step is OPTIONAL and pluggable: if no local weights are provided, the
pipeline just skips confirmation and dispatches on heuristic+verification
confidence alone (logged as "unconfirmed" so you can report both numbers to
judges — e.g. "N alerts from heuristics, M confirmed by secondary model").

To enable it:
    huggingface-cli download Enos-123/traffic-accident-detection-yolo11x \\
        --local-dir ./weights
    # then point SecondaryConfirmation(weights_path="./weights/best.pt")
"""

import os
from typing import Optional

import numpy as np


class SecondaryConfirmation:
    def __init__(self, weights_path: Optional[str] = None, conf_threshold: float = 0.5,
                 device: str = "cuda"):
        self.conf_threshold = conf_threshold
        self.device = device
        self.model = None
        self.enabled = False

        if weights_path and os.path.exists(weights_path):
            from ultralytics import YOLO
            self.model = YOLO(weights_path)
            self.enabled = True
        elif weights_path:
            print(f"[SecondaryConfirmation] weights not found at '{weights_path}' — "
                  f"running WITHOUT secondary confirmation (heuristic+verification only).")

    def confirm(self, frame: np.ndarray) -> dict:
        """
        Runs the accident classifier on a single flagged frame.
        Returns {"confirmed": bool, "confidence": float, "ran": bool}
        """
        if not self.enabled:
            return {"confirmed": True, "confidence": None, "ran": False}

        results = self.model.predict(frame, device=self.device, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return {"confirmed": False, "confidence": 0.0, "ran": True}

        max_conf = float(results[0].boxes.conf.max())
        return {"confirmed": max_conf > self.conf_threshold, "confidence": max_conf, "ran": True}
