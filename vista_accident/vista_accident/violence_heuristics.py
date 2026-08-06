"""
Pose-based violence detection heuristics (VISTA violence branch).

A fight/assault is detected geometrically instead of with a fight classifier:
two people close together (proximity gate) whose wrist/elbow keypoints are
moving fast (limb-motion gate). This is scene-agnostic — it works on unseen
CCTV angles because it reasons about keypoint motion, not appearance, and it
needs no fight-specific training data.

Output feeds the SAME verification/fusion/severity/dispatch chain the
accident branch uses (RawTrigger -> Verifier -> IncidentFuser), so the two
branches stay architecturally parallel.

COCO 17-keypoint order (yolo11n-pose):
    0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders, 7-8 elbows, 9-10 wrists,
    11-12 hips, 13-14 knees, 15-16 ankles.
"""

from collections import deque, namedtuple
from itertools import pairwise
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import ViolenceConfig
from .heuristics import RawTrigger

# Keypoints below this visibility confidence are treated as missing (NaN).
KEYPOINT_CONF_THRESHOLD = 0.3

# COCO skeleton pairs for rendering (also documents which joints are connected).
SKELETON_PAIRS = [
    (5, 7), (7, 9),    # left arm
    (6, 8), (8, 10),   # right arm
    (5, 6),            # shoulders
    (5, 11), (6, 12),  # torso
    (11, 12),          # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

# Track id -> (t, keypoint xy, bbox) per check.
PosePoint = namedtuple("PosePoint", ["t", "kpts", "bbox"])


def _iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


class PoseHistory:
    """Rolling per-track keypoint history: {track_id: deque[(t, kpts, bbox)]}.

    Points arrive once per pose cadence check (every N frames), so retention
    is sized off the check rate, not the raw frame rate.
    """

    def __init__(self, history_seconds: float = 2.0, check_rate_hz: float = 8.0):
        self.history_seconds = history_seconds
        self._maxlen = max(16, int(history_seconds * max(1.0, check_rate_hz) * 2))
        self.tracks: Dict[int, deque] = {}
        self.first_seen: Dict[int, float] = {}

    def update(self, t: float, persons: List[Tuple[int, Tuple[float, float, float, float], np.ndarray]]):
        """persons: list of (track_id, bbox_xyxy, kpts_xy (17,2) float, NaN = missing)."""
        seen = set()
        for tid, bbox, kpts in persons:
            buf = self.tracks.setdefault(tid, deque(maxlen=self._maxlen))
            buf.append(PosePoint(t, kpts, bbox))
            self.first_seen.setdefault(tid, t)
            seen.add(tid)
        # Prune tracks that have left the frame.
        for tid in [k for k in self.tracks if k not in seen]:
            if t - self.tracks[tid][-1].t > self.history_seconds:
                del self.tracks[tid]
                self.first_seen.pop(tid, None)

    def latest(self, track_id: int) -> Optional[PosePoint]:
        buf = self.tracks.get(track_id)
        return buf[-1] if buf else None

    def active_ids(self) -> List[int]:
        return [tid for tid, buf in self.tracks.items() if buf]

    def pair_duration_s(self, tid_a: int, tid_b: int, t: float) -> Optional[float]:
        """Seconds both tracks have been co-visible (bounded by history)."""
        fa = self.first_seen.get(tid_a)
        fb = self.first_seen.get(tid_b)
        if fa is None or fb is None:
            return None
        start = max(fa, fb)
        return max(0.0, t - start)

    def limb_speed(self, track_id: int, window_s: float, kpt_ids: List[int],
                   min_sample_gap_s: float = 0.0) -> Optional[float]:
        """Mean speed (px/s) of the given limb keypoints over the window,
        median-averaged across point pairs spaced >= min_sample_gap_s apart,
        to resist a single jittery keypoint spike (the gap floors the
        frame-to-frame jitter amplification seen on high-fps feeds).
        None if too few points."""
        buf = self.tracks.get(track_id)
        if not buf or len(buf) < 2:
            return None
        t_now = buf[-1].t
        recent = [p for p in buf if p.t >= t_now - window_s]
        if len(recent) < 2:
            recent = list(buf)[-2:]  # fall back to the last pair
        pair_speeds = []
        for pa, pb in pairwise(recent):
            dt = pb.t - pa.t
            if dt < min_sample_gap_s:
                continue  # too close in time — pure keypoint jitter
            dt = max(1e-6, dt)
            displacements = []
            for k in kpt_ids:
                a, b = pa.kpts[k], pb.kpts[k]
                if np.isfinite(a[0]) and np.isfinite(b[0]):
                    displacements.append(float(np.hypot(b[0] - a[0], b[1] - a[1])))
            if displacements:
                pair_speeds.append(sum(displacements) / len(displacements) / dt)
        if not pair_speeds:
            return None
        pair_speeds.sort()
        return float(pair_speeds[len(pair_speeds) // 2])


def check_violence(history: PoseHistory, t: float, cfg: ViolenceConfig,
                   person_count: Optional[int] = None) -> List[RawTrigger]:
    """One signal: a person pair that is close together AND shows either
    aggressive limb motion or a sustained strong box overlap (entanglement).
    Returns RawTrigger(kind='violence') per violating pair.

    Two evidence paths, OR'ed:
      - "limb": wrist/elbow motion above threshold (works when keypoints are
        usable — sparse-scene punches).
      - "overlap": bbox IoU >= pair_overlap_min_iou sustained >=
        pair_overlap_min_duration_s, with NO limb speed required. Distant
        CCTV fights have small people whose keypoints are NaN/unreliable,
        but grappling boxes stay strongly overlapped.

    person_count: concurrent people in the CURRENT frame. Geometry-only limb
    motion cannot separate a walker's arm gesture from a punch (both are
    150-300 px/s on 1080p), so dense scenes are excluded outright — crowd
    violence is deferred to the VideoMAE secondary confirmation."""
    triggers = []
    if person_count is not None and person_count > cfg.pair_max_persons:
        return triggers
    ids = history.active_ids()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id_a, id_b = ids[i], ids[j]
            pa, pb = history.latest(id_a), history.latest(id_b)
            if not pa or not pb:
                continue
            if t - pa.t > cfg.pair_max_stale_s or t - pb.t > cfg.pair_max_stale_s:
                continue  # ghost pair: one track left the frame, its buffer is stale
            ca = ((pa.bbox[0] + pa.bbox[2]) / 2.0, (pa.bbox[1] + pa.bbox[3]) / 2.0)
            cb = ((pb.bbox[0] + pb.bbox[2]) / 2.0, (pb.bbox[1] + pb.bbox[3]) / 2.0)
            dist = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
            iou = _iou(pa.bbox, pb.bbox)
            if dist > cfg.pair_max_distance_px and iou < cfg.pair_min_iou:
                continue  # not physically close — no confrontation possible

            duration = history.pair_duration_s(id_a, id_b, t)
            if duration is None or duration < cfg.pair_min_duration_s:
                continue  # transient pair (tracker blip) — let it stabilize

            speed_a = history.limb_speed(id_a, cfg.limb_window_s, cfg.limb_keypoints,
                                         cfg.limb_min_sample_gap_s)
            speed_b = history.limb_speed(id_b, cfg.limb_window_s, cfg.limb_keypoints,
                                         cfg.limb_min_sample_gap_s)
            available = [s for s in (speed_a, speed_b) if s is not None]
            limb_speed = max(available) if available else None

            use_limb = limb_speed is not None and limb_speed >= cfg.limb_speed_threshold_px_s
            use_overlap = (iou >= cfg.pair_overlap_min_iou
                           and duration >= cfg.pair_overlap_min_duration_s)
            if not use_limb and not use_overlap:
                continue

            cx, cy = (ca[0] + cb[0]) / 2.0, (ca[1] + cb[1]) / 2.0
            triggers.append(RawTrigger(
                kind="violence", track_ids=(id_a, id_b), t=t,
                meta={
                    "limb_speed": limb_speed,
                    "limb_speed_a": speed_a,
                    "limb_speed_b": speed_b,
                    "distance_px": dist,
                    "iou": iou,
                    "duration_s": duration,
                    "cx": cx, "cy": cy,
                    "signal": "limb" if use_limb else "overlap",
                },
            ))
    return triggers
