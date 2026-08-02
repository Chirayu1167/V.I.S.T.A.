"""
Incident fusion: collapses multiple confirmed events that belong to one
physical crash (e.g. a collision plus the resulting speed_drops/anomaly_stops
on the same vehicles, or the same pileup re-keyed under re-assigned tracker
IDs) into a single incident, so dispatch and the UI show ONE alert per crash
instead of several overlapping ones.

Fusion keys:
  - shared track ids (direct membership)
  - same spot (centroid within radius) within a short time window
The most severe kind (hit_and_run > collision > speed_drop > anomaly_stop)
becomes the representative of the incident.
"""

from typing import List

from .verification import ConfirmedEvent

EVENT_PRIORITY = {"hit_and_run": 5, "collision": 4, "smoke": 3, "jerk": 3, "speed_drop": 2, "anomaly_stop": 1}


class _Incident:
    __slots__ = ("kind", "t", "cx", "cy", "track_ids", "meta", "consecutive_frames")

    def __init__(self, kind, t, cx, cy, track_ids, meta, consecutive_frames):
        self.kind = kind
        self.t = t
        self.cx = cx
        self.cy = cy
        self.track_ids = track_ids
        self.meta = meta
        self.consecutive_frames = consecutive_frames


class IncidentFuser:
    """Groups confirmed events across a short window into incidents."""

    def __init__(self, window_s: float = 1.5, radius_px: float = 120.0):
        self.window_s = window_s
        self.radius_px = radius_px
        self._incidents: List[_Incident] = []

    def process(self, t: float, events: List[ConfirmedEvent]) -> List[ConfirmedEvent]:
        merged: List[ConfirmedEvent] = []
        for ev in events:
            cx, cy = ev.meta.get("cx", 0.0), ev.meta.get("cy", 0.0)
            inc = self._find_incident(t, ev, cx, cy)
            if inc is None:
                inc = _Incident(ev.kind, ev.t, cx, cy, ev.track_ids,
                                dict(ev.meta), ev.consecutive_frames)
                self._incidents.append(inc)
                merged.append(ConfirmedEvent(
                    kind=inc.kind, track_ids=inc.track_ids, t=inc.t,
                    consecutive_frames=inc.consecutive_frames, meta=inc.meta,
                ))
            else:
                inc.track_ids = tuple(sorted(set(inc.track_ids) | set(ev.track_ids)))
                inc.t = min(inc.t, ev.t)
                had_smoke = inc.meta.get("has_smoke", False) or inc.kind == "smoke"
                smoke_area = inc.meta.get("smoke_area", 0.0)
                if EVENT_PRIORITY.get(ev.kind, 0) > EVENT_PRIORITY.get(inc.kind, 0):
                    # More severe kind wins the identity; keep the merged tracks.
                    inc.kind = ev.kind
                    inc.meta = dict(ev.meta)
                    inc.cx, inc.cy = cx, cy
                    inc.consecutive_frames = ev.consecutive_frames
                # Carry smoke evidence onto the incident so severity can boost
                # the alert: a collision + growing dust cloud is more serious.
                if ev.kind == "smoke" or had_smoke:
                    if had_smoke:
                        inc.meta["has_smoke"] = True
                        inc.meta["smoke_area"] = max(smoke_area,
                                                     ev.meta.get("area", 0.0),
                                                     ev.meta.get("smoke_area", 0.0))
                    else:
                        inc.meta["has_smoke"] = True
                        inc.meta["smoke_area"] = max(ev.meta.get("area", 0.0),
                                                     ev.meta.get("smoke_area", 0.0))

        # Forget incidents once they can no longer merge with new events.
        cutoff = t - self.window_s * 2
        self._incidents = [i for i in self._incidents if i.t > cutoff]
        return merged

    def _find_incident(self, t: float, ev: ConfirmedEvent, cx: float, cy: float) -> _Incident:
        best = None
        for inc in self._incidents:
            if t - inc.t > self.window_s:
                continue
            if set(inc.track_ids) & set(ev.track_ids):
                return inc  # shared vehicles — same incident
            if cx and cy and inc.cx and inc.cy:
                dist = ((inc.cx - cx) ** 2 + (inc.cy - cy) ** 2) ** 0.5
                if dist <= self.radius_px and best is None:
                    best = inc  # same spot — merge into oldest nearby incident
        return best
