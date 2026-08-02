"""
Optional license-plate OCR, run ONLY on frames already flagged by the
heuristic+verification layer (same "confidence boost, not the primary
detector" pattern as confirmation.py) — specifically hit_and_run events,
where "the vehicle kept moving" makes a plate the single most useful piece
of evidence in the alert.

Uses `easyocr` if installed; degrades to a no-op (enabled=False) otherwise
so it never becomes a hard dependency of the rest of the pipeline.

    pip install easyocr

Usage:
    reader = PlateReader()  # enabled=False if easyocr isn't installed
    result = reader.read(vehicle_crop_bgr)
    # {"plate_text": "KA01AB1234", "confidence": 0.83, "ran": True}
    # or {"plate_text": None, "confidence": 0.0, "ran": True} if no plate found
    # or {"plate_text": None, "confidence": None, "ran": False} if disabled
"""

from typing import Optional, Tuple

import numpy as np


class PlateReader:
    def __init__(self, languages=("en",), conf_threshold: float = 0.4, gpu: bool = False):
        self.conf_threshold = conf_threshold
        self.enabled = False
        self._reader = None

        try:
            import easyocr
            self._reader = easyocr.Reader(list(languages), gpu=gpu)
            self.enabled = True
        except ImportError:
            print("[PlateReader] easyocr not installed — running WITHOUT plate OCR "
                  "(pip install easyocr to enable). Hit-and-run alerts will omit plate_text.")
        except Exception as e:  # model/download failures etc. — degrade, don't crash the pipeline
            print(f"[PlateReader] failed to initialize ({e}) — running WITHOUT plate OCR.")

    def read(self, vehicle_crop: Optional[np.ndarray]) -> dict:
        """vehicle_crop: BGR image cropped to (ideally) just the vehicle's
        bounding box — callers should crop to the vehicle bbox (with a small
        margin) before calling, not pass the full frame, since OCR on a full
        scene wastes time and picks up unrelated text."""
        if not self.enabled or vehicle_crop is None or vehicle_crop.size == 0:
            return {"plate_text": None, "confidence": None, "ran": False}

        try:
            results = self._reader.readtext(vehicle_crop)
        except Exception as e:
            print(f"[PlateReader] OCR failed on this frame: {e}")
            return {"plate_text": None, "confidence": 0.0, "ran": True}

        if not results:
            return {"plate_text": None, "confidence": 0.0, "ran": True}

        # Plates are short, mostly-alphanumeric strings — prefer the
        # highest-confidence candidate that looks plate-like over the
        # highest-confidence text overall (which could be a billboard/sign
        # in the background of the crop).
        best_text, best_conf = None, 0.0
        for _, text, conf in results:
            cleaned = "".join(ch for ch in text if ch.isalnum()).upper()
            if not (4 <= len(cleaned) <= 12):
                continue
            if conf > best_conf:
                best_text, best_conf = cleaned, conf

        if best_text is None or best_conf < self.conf_threshold:
            return {"plate_text": None, "confidence": best_conf, "ran": True}
        return {"plate_text": best_text, "confidence": best_conf, "ran": True}

    @staticmethod
    def crop_bbox(frame: np.ndarray, bbox: Tuple[float, float, float, float],
                  margin_px: int = 8) -> Optional[np.ndarray]:
        """Crop `frame` to `bbox` (xyxy) with a small margin, clamped to the
        frame bounds. Returns None if the resulting crop is empty."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(x1) - margin_px)
        y1 = max(0, int(y1) - margin_px)
        x2 = min(w, int(x2) + margin_px)
        y2 = min(h, int(y2) + margin_px)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]
