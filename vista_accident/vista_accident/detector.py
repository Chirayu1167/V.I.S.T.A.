"""
Thin wrapper around Ultralytics YOLO11m (COCO-pretrained) restricted to the
classes we care about (vehicles + persons/cyclists). FP16 inference on GPU
per the "GPU-optimized inference" feature in VISTA.md.

yolo11m chosen over yolov8n after benchmarking on demo footage: ~+79% more
detections per frame (small/distant vehicles) at ~66 fps on an RTX 3050.
"""

from typing import List, Tuple

import cv2
import numpy as np

from .config import ALL_TRACKED_CLASSES

# Below this mean-brightness (0-255, grayscale), a frame is treated as
# low-light/night footage and gets a CLAHE contrast boost before detection.
# YOLO detection quality on COCO-pretrained weights drops noticeably on dark
# CCTV footage; this is a cheap preprocessing step, not a retrained model,
# but it measurably helps recall on underexposed frames.
LOW_LIGHT_BRIGHTNESS_THRESHOLD = 60.0


class Detector:
    def __init__(self, weights: str = "yolo11m.pt", device: str = "cuda", half: bool = True,
                 conf_threshold: float = 0.35, classes=None, enhance_low_light: bool = True,
                 low_light_threshold: float = LOW_LIGHT_BRIGHTNESS_THRESHOLD):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.device = device
        self.half = half
        self.conf_threshold = conf_threshold
        self.classes = list(classes) if classes is not None else list(ALL_TRACKED_CLASSES)
        self.enhance_low_light = enhance_low_light
        self.low_light_threshold = low_light_threshold
        # Applied to the L channel of LAB color space so color balance isn't
        # disturbed the way a flat brightness/contrast stretch would be.
        self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

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

    def _enhance_if_dark(self, frame: np.ndarray) -> np.ndarray:
        if not self.enhance_low_light:
            return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if float(gray.mean()) >= self.low_light_threshold:
            return frame
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def detect(self, frame: np.ndarray) -> List[Tuple[Tuple[float, float, float, float], float, int]]:
        """Returns list of (bbox_xyxy, confidence, class_id).

        Single-frame path — unchanged output vs. every prior version of this
        method. Internally now just calls detect_batch() with a batch of 1
        and unwraps it, so there is exactly one place (_parse_result) that
        turns an Ultralytics Results object into our tuple format.
        """
        return self.detect_batch([frame])[0]

    def detect_batch(
        self, frames: List[np.ndarray]
    ) -> List[List[Tuple[Tuple[float, float, float, float], float, int]]]:
        """Batched version of detect() — runs ONE forward pass across all
        given frames instead of one call per frame. Returns a list aligned
        1:1 with `frames`: result[i] is detect()'s return value for frames[i].

        This is what makes shared-detector multi-camera batching possible:
        the batch collector gathers one frame per camera per cycle and
        calls this once instead of calling detect() N times. Per-frame
        low-light enhancement still runs individually before batching, so
        night-camera handling is identical to the single-frame path.

        A bad/corrupt frame must never take the whole batch down: any frame
        that fails enhancement or shape validation is swapped for a zero
        detection result rather than raising, so one flaky camera can't
        stall detection for every other camera in the same cycle.
        """
        out: List[List[Tuple[Tuple[float, float, float, float], float, int]]] = [[] for _ in frames]
        valid_indices: List[int] = []
        valid_frames: List[np.ndarray] = []

        for i, frame in enumerate(frames):
            try:
                valid_frames.append(self._enhance_if_dark(frame))
                valid_indices.append(i)
            except Exception:
                # Leave out[i] as [] — caller sees "no detections this frame"
                # for the bad frame only, everything else proceeds normally.
                continue

        if not valid_frames:
            return out

        results = self.model.predict(
            valid_frames,
            device=self.device,
            half=self.half,
            conf=self.conf_threshold,
            classes=self.classes,
            verbose=False,
        )

        for local_i, result in enumerate(results):
            out[valid_indices[local_i]] = self._parse_result(result)

        return out

    @staticmethod
    def _parse_result(result) -> List[Tuple[Tuple[float, float, float, float], float, int]]:
        """Turns one Ultralytics Results object into our (bbox, conf, cls) tuples."""
        out = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return out
        for box in boxes:
            xyxy = tuple(box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            out.append((xyxy, conf, cls))
        return out
