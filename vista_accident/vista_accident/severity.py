"""
Dynamic severity assessment for accident events.

Moves beyond a static kind -> severity map. Each event kind gets a score
(0.0-1.0) from scene signals: pre-impact speed, overlap depth, pedestrian
proximity, and post-impact behavior. The score determines which emergency
channels get notified, so a minor fender-bender only reaches traffic police
while a high-speed multi-vehicle crash also alerts EMS and police control.

Severity -> channels:
  low       (0.00-0.35) -> traffic police
  medium    (0.35-0.60) -> traffic police
  high      (0.60-0.85) -> traffic police + hospital/EMS
  critical  (0.85-1.00) -> traffic police + hospital/EMS + police control
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import PERSON_CLASSES
from .track_history import TrackHistory
from .verification import ConfirmedEvent


@dataclass
class SeverityConfig:
    speed_drop_window_s: float = 0.5
    low_speed_threshold: float = 30.0
    high_speed_threshold: float = 120.0
    low_iou: float = 0.15
    high_iou: float = 0.55
    person_proximity_radius: float = 200.0
    kind_baseline: Dict[str, float] = field(default_factory=lambda: {
        "hit_and_run": 0.65,
        "collision": 0.55,
        "jerk": 0.45,
        "smoke": 0.40,
        "speed_drop": 0.30,
        "anomaly_stop": 0.15,
    })


SEVERITY_LABELS = ["low", "medium", "high", "critical"]

CHANNELS_BY_SEVERITY = {
    "low": ("traffic_police",),
    "medium": ("traffic_police",),
    "high": ("traffic_police", "hospital_ems"),
    "critical": ("traffic_police", "hospital_ems", "police_control_room"),
}


def _label(score: float) -> str:
    if score < 0.35:
        return "low"
    if score < 0.60:
        return "medium"
    if score < 0.85:
        return "high"
    return "critical"


class SeverityAssessor:
    def __init__(self, cfg: Optional[SeverityConfig] = None):
        self.cfg = cfg or SeverityConfig()

    def assess(self, event: ConfirmedEvent, history: TrackHistory) -> str:
        fn = {
            "collision": self._score_collision,
            "hit_and_run": self._score_hit_and_run,
            "jerk": self._score_jerk,
            "smoke": self._score_smoke,
            "speed_drop": self._score_speed_drop,
            "anomaly_stop": self._score_anomaly_stop,
        }.get(event.kind)
        if fn is None:
            return "medium"
        return _label(fn(event, history))

    # ------------------------------------------------------------------
    # kind-specific scorers
    # ------------------------------------------------------------------

    def _score_collision(self, event: ConfirmedEvent, history: TrackHistory) -> float:
        meta = event.meta
        iou = meta.get("iou", 0.0)
        v_a = meta.get("v_a", 0.0)
        v_b = meta.get("v_b", 0.0)

        tid_a, tid_b = event.track_ids[:2]
        prior_a = history.velocity(tid_a, self.cfg.speed_drop_window_s) or 0.0
        prior_b = history.velocity(tid_b, self.cfg.speed_drop_window_s) or 0.0
        max_prior = max(prior_a, prior_b)

        speed_score = np.clip(max_prior / self.cfg.high_speed_threshold, 0.0, 1.0)
        impact_score = np.clip(
            (iou - self.cfg.low_iou) / (self.cfg.high_iou - self.cfg.low_iou + 1e-9),
            0.0, 1.0,
        )
        both_stopped = 1.0 if (v_a < 10.0 and v_b < 10.0) else 0.0
        persons = self._persons_near(event, history)
        person_factor = min(persons / 3.0, 1.0) * 0.15
        smoke_factor = 0.10 if (event.meta.get("has_smoke") or event.meta.get("smoke_area")) else 0.0

        return np.clip(
            0.30 * speed_score + 0.25 * impact_score + 0.15 * both_stopped
            + 0.15 * person_factor + 0.15 + smoke_factor,
            0.0, 1.0,
        )

    def _score_hit_and_run(self, event: ConfirmedEvent, history: TrackHistory) -> float:
        meta = event.meta
        ped_drop = meta.get("ped_drop", 0.0)
        vehicle_v = meta.get("vehicle_v", 0.0)

        drop_score = np.clip((ped_drop - 0.5) / 0.5, 0.0, 1.0)
        flee_score = np.clip(vehicle_v / 80.0, 0.0, 1.0)
        persons = self._persons_near(event, history)
        person_factor = min(persons / 3.0, 1.0) * 0.10

        return np.clip(
            0.40 * drop_score + 0.25 * flee_score + 0.10 * person_factor + 0.25,
            0.0, 1.0,
        )

    def _score_jerk(self, event: ConfirmedEvent, history: TrackHistory) -> float:
        meta = event.meta
        decel = meta.get("decel", 0.0)
        prior_v = meta.get("prior_v", 0.0)

        # Hard impact = very high deceleration + the vehicle ended near-stopped.
        decel_score = np.clip((decel - 5.0) / 15.0, 0.0, 1.0)  # 5→20 m/s²
        speed_score = np.clip(prior_v / self.cfg.high_speed_threshold, 0.0, 1.0)
        persons = self._persons_near(event, history)
        person_factor = min(persons / 3.0, 1.0) * 0.10
        smoke_factor = 0.10 if (meta.get("has_smoke") or meta.get("smoke_area")) else 0.0

        return np.clip(
            0.40 * decel_score + 0.30 * speed_score + 0.10 * person_factor
            + 0.10 + smoke_factor,
            0.0, 1.0,
        )

    def _score_smoke(self, event: ConfirmedEvent, history: TrackHistory) -> float:
        meta = event.meta
        area = meta.get("area", 0.0)

        # Bigger cloud = more violent crash (more debris/dust thrown up).
        area_score = np.clip(area / 20000.0, 0.0, 1.0)
        growth = meta.get("growth", 1.0)
        growth_score = np.clip((growth - 1.0) / 2.0, 0.0, 1.0)
        persons = self._persons_near(event, history)
        person_factor = min(persons / 3.0, 1.0) * 0.10

        return np.clip(
            0.35 * area_score + 0.25 * growth_score + 0.10 * person_factor + 0.20,
            0.0, 1.0,
        )

    def _score_speed_drop(self, event: ConfirmedEvent, history: TrackHistory) -> float:
        meta = event.meta
        prior_v = meta.get("prior_v", 0.0)
        drop_ratio = meta.get("drop_ratio", 0.0)

        speed_score = np.clip(prior_v / self.cfg.high_speed_threshold, 0.0, 1.0)
        drop_score = np.clip((drop_ratio - 0.5) / 0.5, 0.0, 1.0)
        persons = self._persons_near(event, history)
        person_factor = min(persons / 2.0, 1.0) * 0.10

        return np.clip(
            0.35 * speed_score + 0.35 * drop_score + 0.10 * person_factor + 0.20,
            0.0, 1.0,
        )

    def _score_anomaly_stop(self, event: ConfirmedEvent, history: TrackHistory) -> float:
        duration = event.meta.get("duration", 2.0)
        dur_score = np.clip((duration - 2.0) / 10.0, 0.0, 1.0)
        persons = self._persons_near(event, history)
        person_factor = min(persons / 2.0, 1.0) * 0.10

        return np.clip(0.40 * dur_score + 0.10 * person_factor + 0.05, 0.0, 1.0)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _persons_near(self, event: ConfirmedEvent, history: TrackHistory) -> int:
        pts = [history.latest(tid) for tid in event.track_ids]
        pts = [p for p in pts if p is not None]
        if not pts:
            return 0
        center_x = sum(p.cx for p in pts) / len(pts)
        center_y = sum(p.cy for p in pts) / len(pts)

        count = 0
        for pid in history.active_ids(cls_filter=PERSON_CLASSES):
            p = history.latest(pid)
            if p is None:
                continue
            dist = ((p.cx - center_x) ** 2 + (p.cy - center_y) ** 2) ** 0.5
            if dist < self.cfg.person_proximity_radius:
                count += 1
        return count