"""
The four heuristic signals described in VISTA.md, each operating purely on
TrackHistory (no model inference here — this is the ~50-lines-of-custom-logic
layer sitting on top of detection + tracking).

Each function returns a list of RawTrigger — one per frame call. These feed
into verification.Verifier, which requires a signal to persist across
consecutive frames before it becomes a confirmed alert.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

from .config import HeuristicConfig, PERSON_CLASSES, VEHICLE_CLASSES
from .track_history import TrackHistory


@dataclass
class RawTrigger:
    kind: str                  # "speed_drop" | "collision" | "anomaly_stop" | "hit_and_run"
    track_ids: Tuple[int, ...]  # 1 id for speed_drop/anomaly_stop, 2 for collision/hit_and_run
    t: float
    meta: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Stable dedup/verification key for this event instance."""
        return f"{self.kind}:{'-'.join(str(i) for i in sorted(self.track_ids))}"


def check_speed_drop(history: TrackHistory, t: float, cfg: HeuristicConfig,
                     stop_zones=None) -> List[RawTrigger]:
    triggers = []
    stop_zones = stop_zones or []
    for tid in history.active_ids(cls_filter=VEHICLE_CLASSES):
        # Both readings are windowed averages (previous window vs. current
        # window) instead of raw instantaneous velocity, so a single jittery
        # tracker frame cannot fake a "drop".
        now_v = history.velocity_between(tid, t - cfg.speed_drop_window_s, t)
        prior_v = history.velocity_between(
            tid, t - 2 * cfg.speed_drop_window_s, t - cfg.speed_drop_window_s)
        if prior_v is None or now_v is None:
            continue
        if prior_v < cfg.speed_drop_min_prior_speed:
            continue  # was already slow/stationary — not a meaningful "drop"
        p = history.latest(tid)
        if p and _in_any_zone((p.cx, p.cy), stop_zones):
            continue  # legitimate braking at an intersection/bus stop — suppress
        drop_ratio = (prior_v - now_v) / prior_v
        if drop_ratio > cfg.speed_drop_ratio:
            triggers.append(RawTrigger(
                kind="speed_drop", track_ids=(tid,), t=t,
                meta={"prior_v": prior_v, "now_v": now_v, "drop_ratio": drop_ratio,
                      "cx": p.cx, "cy": p.cy},
            ))
    return triggers


def check_collision(history: TrackHistory, t: float, cfg: HeuristicConfig) -> List[RawTrigger]:
    triggers = []
    vehicle_ids = history.active_ids(cls_filter=VEHICLE_CLASSES)
    for i in range(len(vehicle_ids)):
        for j in range(i + 1, len(vehicle_ids)):
            id_a, id_b = vehicle_ids[i], vehicle_ids[j]
            pa, pb = history.latest(id_a), history.latest(id_b)
            if not pa or not pb:
                continue
            iou = TrackHistory.iou(pa.bbox, pb.bbox)
            if iou < cfg.collision_iou_threshold:
                continue
            va = history.instantaneous_velocity(id_a)
            vb = history.instantaneous_velocity(id_b)
            if va is None or vb is None:
                continue
            if va <= cfg.collision_max_velocity and vb <= cfg.collision_max_velocity:
                triggers.append(RawTrigger(
                    kind="collision", track_ids=(id_a, id_b), t=t,
                    meta={"iou": iou, "v_a": va, "v_b": vb},
                ))
    return triggers


def check_anomaly_stop(history: TrackHistory, t: float, cfg: HeuristicConfig, stop_zones=None) -> List[RawTrigger]:
    triggers = []
    stop_zones = stop_zones or []
    for tid in history.active_ids(cls_filter=VEHICLE_CLASSES):
        duration = history.stationary_duration(tid, cfg.anomaly_stop_max_velocity)
        if duration < cfg.anomaly_stop_duration_s:
            continue
        p = history.latest(tid)
        if p and _in_any_zone((p.cx, p.cy), stop_zones):
            continue  # legitimate stop (intersection/bus stop) — suppress
        triggers.append(RawTrigger(
            kind="anomaly_stop", track_ids=(tid,), t=t,
            meta={"duration": duration, "cx": p.cx, "cy": p.cy},
        ))
    return triggers


def check_hit_and_run(history: TrackHistory, t: float, cfg: HeuristicConfig) -> List[RawTrigger]:
    triggers = []
    vehicle_ids = history.active_ids(cls_filter=VEHICLE_CLASSES)
    person_ids = history.active_ids(cls_filter=PERSON_CLASSES)
    for vid in vehicle_ids:
        pv = history.latest(vid)
        v_now = history.instantaneous_velocity(vid)
        if not pv or v_now is None:
            continue
        for pid in person_ids:
            pp = history.latest(pid)
            if not pp:
                continue
            iou = TrackHistory.iou(pv.bbox, pp.bbox)
            if iou < cfg.hitrun_iou_threshold:
                continue
            prior_ped_v = history.velocity(pid, cfg.speed_drop_window_s)
            now_ped_v = history.instantaneous_velocity(pid)
            if prior_ped_v is None or now_ped_v is None or prior_ped_v <= 0:
                continue
            ped_drop = (prior_ped_v - now_ped_v) / prior_ped_v
            if ped_drop > cfg.hitrun_ped_velocity_drop and v_now > cfg.hitrun_vehicle_continues_min_speed:
                triggers.append(RawTrigger(
                    kind="hit_and_run", track_ids=(vid, pid), t=t,
                    meta={"iou": iou, "ped_drop": ped_drop, "vehicle_v": v_now,
                          "cx": pv.cx, "cy": pv.cy},
                ))
    return triggers


def _in_any_zone(point, zones) -> bool:
    """zones: list of polygons [(x,y), ...]; point-in-polygon via ray casting."""
    x, y = point
    for poly in zones:
        n = len(poly)
        if n < 3:
            continue
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
                inside = not inside
            j = i
        if inside:
            return True
    return False


def run_all_heuristics(history: TrackHistory, t: float, cfg: HeuristicConfig, stop_zones=None) -> List[RawTrigger]:
    triggers = []
    triggers += check_speed_drop(history, t, cfg, stop_zones=stop_zones)
    triggers += check_collision(history, t, cfg)
    triggers += check_anomaly_stop(history, t, cfg, stop_zones=stop_zones)
    triggers += check_hit_and_run(history, t, cfg)
    return triggers
