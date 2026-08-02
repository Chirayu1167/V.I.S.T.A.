"""
Smoke / dust-cloud detector (Signal 6).

A hard impact (car into a wall, median, tree, another vehicle) throws up a
dust/smoke cloud that appears suddenly and grows over ~1-2 seconds. This is a
purely CV detector — no extra ML model, it rides the same frame the rest of
the pipeline already decodes.

Idea: smoke/dust is a large blob of near-gray pixels (low saturation in HSV),
mid-bright, with very low local texture (haze is smooth — no edges). We mask
those pixels, blob them with connected components, then require the blob to
(1) persist across several frames and (2) GROW vs. its first-seen area.
That growth requirement is what separates a crash dust cloud from fog, a
light-colored parked truck, or a bright wall — those don't expand.

Emits RawTrigger(kind="smoke") per qualifying frame; verification.py then
requires the signal to persist (smoke_persist_frames) before an alert.
"""

from typing import List, Optional

import cv2
import numpy as np

from .config import HeuristicConfig
from .heuristics import RawTrigger


class SmokeDetector:
    def __init__(self, cfg: Optional[HeuristicConfig] = None):
        self.cfg = cfg or HeuristicConfig()
        # blob_key -> (first_seen_t, first_area, last_area)
        self._blobs = {}
        self._last_t = 0.0

    def process(self, frame: np.ndarray, t: float) -> List[RawTrigger]:
        """Run one frame; return smoke triggers (empty if none)."""
        triggers = []
        if not self.cfg.smoke_detector_enabled or frame is None or frame.size == 0:
            return triggers

        mask = self._smoke_mask(frame)
        blobs = self._blob_centers(mask)
        if not blobs:
            self._blobs.clear()
            self._last_t = t
            return triggers

        current_keys = set()
        for cx, cy, area in blobs:
            key = (int(cx // 40), int(cy // 40))
            current_keys.add(key)
            st = self._blobs.get(key)
            if st is None:
                self._blobs[key] = [t, area, area]
                continue
            first_t, first_area, last_area = st
            st[2] = area
            # Must persist AND grow vs. its first-seen area: a dust cloud
            # expands; fog / static bright regions don't.
            if t - first_t > 0.15 and area > self.cfg.smoke_min_area_px:
                if area >= last_area and area > first_area * self.cfg.smoke_growth_ratio:
                    triggers.append(RawTrigger(
                        kind="smoke", track_ids=(), t=t,
                        meta={"cx": cx, "cy": cy, "area": area,
                              "growth": area / max(first_area, 1.0)},
                    ))

        # Forget blobs that vanished.
        self._blobs = {k: v for k, v in self._blobs.items() if k in current_keys}
        self._last_t = t
        return triggers

    def _smoke_mask(self, frame: np.ndarray) -> np.ndarray:
        """Pixels that look like smoke: grayish (low saturation), mid-bright,
        and locally smooth (low texture)."""
        cfg = self.cfg
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        # Local texture via box blur of (gray, gray^2): std = sqrt(E[x²]-E[x]²).
        k = 11
        mean = cv2.boxFilter(gray.astype(np.float32), -1, (k, k))
        sq_mean = cv2.boxFilter((gray.astype(np.float32) ** 2), -1, (k, k))
        var = np.maximum(0.0, sq_mean - mean * mean)
        std = np.sqrt(var)

        mask = (
            (sat < cfg.smoke_max_saturation)
            & (val >= cfg.smoke_min_value)
            & (val <= cfg.smoke_max_value)
            & (std < cfg.smoke_max_local_std)
        ).astype(np.uint8) * 255

        # Close small gaps inside a cloud, drop isolated specks.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        return mask

    def _blob_centers(self, mask: np.ndarray) -> List[tuple]:
        """Connected components of the smoke mask -> [(cx, cy, area)]."""
        out = []
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.cfg.smoke_min_area_px or area > self.cfg.smoke_max_area_px:
                continue
            x, y, w, h = (int(stats[i, k]) for k in
                          (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                           cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
            out.append((x + w / 2.0, y + h / 2.0, area))
        return out

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Overlay the current smoke mask (for debugging/demo)."""
        if frame is None or frame.size == 0:
            return frame
        mask = self._smoke_mask(frame)
        frame[mask > 0] = (frame[mask > 0] * 0.6 + np.array([0, 0, 200], np.uint8) * 0.4)
        return frame
