"""
Rolling per-track history: {track_id: deque[(t, cx, cy, bbox, cls)]}

This is the shared data structure every heuristic reads from. Keeping it
separate from the tracker means we can swap ByteTrack for any other tracker
without touching heuristic logic.

Now supports optional ML-based speed estimator for real-world speed (m/s, km/h).
"""

from collections import deque, namedtuple
from typing import Dict, List, Optional, Tuple, Union

from .speed_estimator import MlSpeedEstimator, TrackSpeed

TrackPoint = namedtuple("TrackPoint", ["t", "cx", "cy", "bbox", "cls"])


def _center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


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


class TrackHistory:
    """Stores a rolling time-window of positions for every active track id.
    
    Can optionally use an MlSpeedEstimator for real-world speed estimation.
    If no estimator is provided, falls back to pixel-based velocity calculations.
    """

    def __init__(
        self,
        history_seconds: float = 3.0,
        assumed_fps: float = 25.0,
        speed_estimator: Optional[MlSpeedEstimator] = None,
    ):
        self.history_seconds = history_seconds
        self.assumed_fps = assumed_fps
        self.speed_estimator = speed_estimator
        # Generous buffer length; actual retention is time-based via prune().
        self._maxlen = max(30, int(history_seconds * assumed_fps * 2))
        self.tracks: Dict[int, deque] = {}
        self.last_seen: Dict[int, float] = {}

    def update(
        self,
        t: float,
        detections: List[Tuple[int, Tuple[float, float, float, float], int]],
        frame: Optional[object] = None,  # unused, kept for call-site compatibility
    ):
        """
        detections: list of (track_id, bbox_xyxy, class_id)

        `frame` is accepted but no longer used: the speed estimator now
        reuses these same (track_id, bbox, cls) tuples -- the ones produced
        by the pipeline's own Tracker -- instead of re-running detection
        internally. This keeps track IDs consistent between heuristics and
        speed, and avoids a second GPU inference pass per frame.
        """
        seen_ids = set()
        for track_id, bbox, cls in detections:
            cx, cy = _center(bbox)
            buf = self.tracks.setdefault(track_id, deque(maxlen=self._maxlen))
            buf.append(TrackPoint(t, cx, cy, bbox, cls))
            self.last_seen[track_id] = t
            seen_ids.add(track_id)

        if self.speed_estimator is not None:
            self.speed_estimator.update(t, detections)

        self._prune(t)

    def _prune(self, now: float):
        stale = [tid for tid, last in self.last_seen.items() if now - last > self.history_seconds * 2]
        for tid in stale:
            self.tracks.pop(tid, None)
            self.last_seen.pop(tid, None)

    def latest(self, track_id: int) -> Optional[TrackPoint]:
        buf = self.tracks.get(track_id)
        return buf[-1] if buf else None

    def point_near(self, track_id: int, t_target: float) -> Optional[TrackPoint]:
        """Closest recorded point to t_target (looking backward)."""
        buf = self.tracks.get(track_id)
        if not buf:
            return None
        best = None
        for p in buf:
            if p.t <= t_target:
                best = p
            else:
                break
        return best or buf[0]

    def velocity(self, track_id: int, window_s: float) -> Optional[float]:
        """Average speed (m/s) over the last `window_s` seconds.
        
        Uses ML estimator if available, otherwise falls back to pixel-based.
        """
        # Prefer ML estimator for real-world speeds
        if self.speed_estimator is not None:
            return self.speed_estimator.velocity(track_id, window_s)
        
        # Fallback: pixel-based
        buf = self.tracks.get(track_id)
        if not buf or len(buf) < 2:
            return None
        latest = buf[-1]
        past = self.point_near(track_id, latest.t - window_s)
        if past is None or past.t == latest.t:
            return None
        dist = ((latest.cx - past.cx) ** 2 + (latest.cy - past.cy) ** 2) ** 0.5
        dt = latest.t - past.t
        return dist / dt if dt > 0 else None

    def velocity_between(self, track_id: int, t0: float, t1: float) -> Optional[float]:
        """Average speed (m/s) between two (backward-looking) timestamps.
        Uses ML estimator if available, otherwise falls back to pixel-based."""
        if self.speed_estimator is not None:
            return self.speed_estimator.velocity_between(track_id, t0, t1)
        
        p2 = self.point_near(track_id, t1)
        p1 = self.point_near(track_id, t0)
        if not p1 or not p2 or p1.t == p2.t:
            return None
        dist = ((p2.cx - p1.cx) ** 2 + (p2.cy - p1.cy) ** 2) ** 0.5
        dt = p2.t - p1.t
        return dist / dt if dt > 0 else None

    def instantaneous_velocity(self, track_id: int) -> Optional[float]:
        """Instantaneous velocity (m/s). Uses ML estimator if available."""
        if self.speed_estimator is not None:
            return self.speed_estimator.instantaneous_velocity(track_id)
        
        buf = self.tracks.get(track_id)
        if not buf or len(buf) < 2:
            return None
        p2, p1 = buf[-1], buf[-2]
        dt = p2.t - p1.t
        if dt <= 0:
            return None
        dist = ((p2.cx - p1.cx) ** 2 + (p2.cy - p1.cy) ** 2) ** 0.5
        return dist / dt

    def stationary_duration(self, track_id: int, max_velocity: float) -> float:
        """How long (seconds) the track has stayed below max_velocity (m/s), ending now."""
        if self.speed_estimator is not None:
            return self.speed_estimator.stationary_duration(track_id, max_velocity)
        
        buf = self.tracks.get(track_id)
        if not buf or len(buf) < 2:
            return 0.0
        pts = list(buf)
        end_t = pts[-1].t
        start_t = end_t
        for i in range(len(pts) - 1, 0, -1):
            p2, p1 = pts[i], pts[i - 1]
            dt = p2.t - p1.t
            if dt <= 0:
                continue
            dist = ((p2.cx - p1.cx) ** 2 + (p2.cy - p1.cy) ** 2) ** 0.5
            v = dist / dt
            if v <= max_velocity:
                start_t = p1.t
            else:
                break
        return end_t - start_t

    def deceleration(self, track_id: int, window_s: float) -> Optional[float]:
        """Peak positive deceleration (px/s²) over the last `window_s` seconds.
        Delegates to the ML estimator (m/s²) when available; otherwise reads the
        raw pixel series directly."""
        if self.speed_estimator is not None:
            return self.speed_estimator.deceleration(track_id, window_s)

        buf = self.tracks.get(track_id)
        if not buf or len(buf) < 3:
            return None
        pts = list(buf)
        cutoff = pts[-1].t - window_s
        seg = [p for p in pts if p.t >= cutoff]
        vs = []
        for i in range(1, len(seg)):
            dt = seg[i].t - seg[i - 1].t
            if dt <= 0:
                continue
            dist = ((seg[i].cx - seg[i - 1].cx) ** 2 + (seg[i].cy - seg[i - 1].cy) ** 2) ** 0.5
            vs.append((seg[i].t, dist / dt))
        if len(vs) < 2:
            return None
        decels = []
        for i in range(1, len(vs)):
            dt = vs[i][0] - vs[i - 1][0]
            if dt <= 0:
                continue
            decels.append((vs[i - 1][1] - vs[i][1]) / dt)
        return max(0.0, max(decels)) if decels else None

    def active_ids(self, cls_filter=None) -> List[int]:
        ids = list(self.tracks.keys())
        if cls_filter is None:
            return ids
        out = []
        for tid in ids:
            p = self.latest(tid)
            if p and p.cls in cls_filter:
                out.append(tid)
        return out

    def get_ml_speed(self, track_id: int) -> Optional[TrackSpeed]:
        """Get ML-based speed estimate with full details (m/s, km/h, world position)."""
        if self.speed_estimator is not None:
            return self.speed_estimator.get_speed(track_id)
        return None

    def get_all_ml_speeds(self) -> Dict[int, TrackSpeed]:
        """Get all ML-based speed estimates."""
        if self.speed_estimator is not None:
            return self.speed_estimator.get_all_speeds()
        return {}

    @staticmethod
    def iou(box_a, box_b) -> float:
        return _iou(box_a, box_b)
