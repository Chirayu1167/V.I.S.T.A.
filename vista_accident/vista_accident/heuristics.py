"""
The four heuristic signals described in VISTA.md, each operating purely on
TrackHistory (no model inference here — this is the ~50-lines-of-custom-logic
layer sitting on top of detection + tracking).

Each function returns a list of RawTrigger — one per frame call. These feed
into verification.Verifier, which requires a signal to persist across
consecutive frames before it becomes a confirmed alert.

Now supports ML-based speed estimation for real-world speeds (m/s, km/h).
"""

from dataclasses import dataclass, field
from typing import List, Tuple

from .config import HeuristicConfig, PERSON_CLASSES, VEHICLE_CLASSES
from .track_history import TrackHistory
from .speed_estimator import TrackSpeed


@dataclass
class RawTrigger:
    kind: str                  # "speed_drop" | "collision" | "anomaly_stop" | "hit_and_run" | "jerk" | "smoke"
    track_ids: Tuple[int, ...]  # 1 id for speed_drop/anomaly_stop/jerk, 2 for collision/hit_and_run, () for smoke
    t: float
    meta: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Stable dedup/verification key for this event instance."""
        if self.track_ids:
            return f"{self.kind}:{'-'.join(str(i) for i in sorted(self.track_ids))}"
        # Trackless kinds (smoke) key by location so different clouds don't
        # share one verification streak.
        cx, cy = self.meta.get("cx", 0.0), self.meta.get("cy", 0.0)
        return f"{self.kind}:{int(cx // 40)},{int(cy // 40)}"


def check_speed_drop(history: TrackHistory, t: float, cfg: HeuristicConfig,
                     stop_zones=None) -> List[RawTrigger]:
    triggers = []
    stop_zones = stop_zones or []
    # A queue braking together at a light/congestion is a normal stop, not an
    # accident — reuse the same jam suppression as anomaly_stop.
    jammed = _traffic_jam_track_ids(history, cfg)
    for tid in history.active_ids(cls_filter=VEHICLE_CLASSES):
        if tid in jammed:
            continue  # part of a stationary queue — normal braking
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
            # Include ML speed details if available
            ml_speed = history.get_ml_speed(tid)
            meta = {"prior_v": prior_v, "now_v": now_v, "drop_ratio": drop_ratio,
                    "cx": p.cx, "cy": p.cy}
            if ml_speed:
                meta["ml_speed_mps"] = ml_speed.speed_mps
                meta["ml_speed_kmph"] = ml_speed.speed_kmph
                meta["ml_world_pos"] = ml_speed.world_pos
            triggers.append(RawTrigger(
                kind="speed_drop", track_ids=(tid,), t=t,
                meta=meta,
            ))
    return triggers


def _impacted(prior_v, now_v, cfg: HeuristicConfig) -> bool:
    """Impact signature for one vehicle: it was moving meaningfully before the
    overlap and its speed collapsed at the moment of overlap. A car that was
    already slow/stationary before (parked, queued) does NOT count."""
    if prior_v is None or now_v is None:
        return False
    if prior_v < cfg.collision_min_prior_speed:
        return False  # was already slow/stationary — parked cars don't crash
    stopped = now_v <= cfg.collision_max_velocity
    collapsed = now_v < prior_v * (1.0 - cfg.collision_decel_ratio)
    return stopped or collapsed


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
            # "Prior" speed from a pre-impact window that ends BEFORE the
            # overlap moment; the overlap frames already contain the
            # deceleration, which would drag the average down.
            prior_a = history.velocity_between(
                id_a, t - cfg.collision_prior_lookback_s, t - cfg.collision_prior_end_s)
            prior_b = history.velocity_between(
                id_b, t - cfg.collision_prior_lookback_s, t - cfg.collision_prior_end_s)
            now_a = history.instantaneous_velocity(id_a)
            now_b = history.instantaneous_velocity(id_b)
            if not (_impacted(prior_a, now_a, cfg) or _impacted(prior_b, now_b, cfg)):
                continue
            # Include ML speed details if available
            ml_a = history.get_ml_speed(id_a)
            ml_b = history.get_ml_speed(id_b)
            meta = {"iou": iou, "v_a": now_a, "v_b": now_b,
                    "prior_v_a": prior_a, "prior_v_b": prior_b,
                    "cx": (pa.cx + pb.cx) / 2.0, "cy": (pa.cy + pb.cy) / 2.0}
            if ml_a:
                meta["ml_a_speed_mps"] = ml_a.speed_mps
                meta["ml_a_speed_kmph"] = ml_a.speed_kmph
            if ml_b:
                meta["ml_b_speed_mps"] = ml_b.speed_mps
                meta["ml_b_speed_kmph"] = ml_b.speed_kmph
            triggers.append(RawTrigger(
                kind="collision", track_ids=(id_a, id_b), t=t,
                meta=meta,
            ))
    return triggers


def _traffic_jam_track_ids(history: TrackHistory, cfg: HeuristicConfig) -> set:
    """Tracks that are part of a stationary queue (3+ stationary vehicles
    close together). A queue is a traffic jam or a red light, not an
    incident, so anomaly_stop is suppressed for them. Uses world-space
    positions when the ML estimator is available, otherwise falls back to
    pixel-space distance (same suppression, just coarser)."""
    stationary = {}
    use_world = history.speed_estimator is not None
    for tid in history.active_ids(cls_filter=VEHICLE_CLASSES):
        v = history.instantaneous_velocity(tid)
        if v is not None and v <= cfg.anomaly_stop_max_velocity:
            ml = history.get_ml_speed(tid)
            if use_world and ml is not None and ml.world_pos:
                stationary[tid] = ml.world_pos
            else:
                p = history.latest(tid)
                if p:
                    stationary[tid] = (p.cx, p.cy)
    max_gap = cfg.traffic_jam_max_gap_m if use_world else cfg.traffic_jam_max_gap_px
    ids = list(stationary.keys())
    parent = list(range(len(ids)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            xi, yi = stationary[ids[i]]
            xj, yj = stationary[ids[j]]
            gap = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5
            if gap <= max_gap:
                union(i, j)

    sizes = {}
    for i in range(len(ids)):
        r = find(i)
        sizes[r] = sizes.get(r, 0) + 1
    return {tid for i, tid in enumerate(ids) if sizes[find(i)] >= cfg.traffic_jam_min_vehicles}


def check_anomaly_stop(history: TrackHistory, t: float, cfg: HeuristicConfig, stop_zones=None) -> List[RawTrigger]:
    triggers = []
    stop_zones = stop_zones or []
    jammed = _traffic_jam_track_ids(history, cfg)
    for tid in history.active_ids(cls_filter=VEHICLE_CLASSES):
        if tid in jammed:
            continue  # stationary queue — jam or red light, not an incident
        duration = history.stationary_duration(tid, cfg.anomaly_stop_max_velocity)
        if duration < cfg.anomaly_stop_duration_s:
            continue
        # A normal stop (red light / turning / crawling traffic) is preceded
        # by gradual braking; a crashed car was moving meaningfully right up
        # to the impact. Require prior motion so parked cars and cars that
        # were never moving fast enough don't alert.
        prior_v = history.velocity_between(
            tid, t - duration - cfg.anomaly_stop_prior_window_s,
            t - duration)
        if prior_v is None or prior_v < cfg.anomaly_stop_min_prior_speed:
            continue
        p = history.latest(tid)
        if p and _in_any_zone((p.cx, p.cy), stop_zones):
            continue  # legitimate stop (intersection/bus stop) — suppress
        ml_speed = history.get_ml_speed(tid)
        meta = {"duration": duration, "prior_v": prior_v, "cx": p.cx, "cy": p.cy}
        if ml_speed:
            meta["ml_speed_mps"] = ml_speed.speed_mps
            meta["ml_speed_kmph"] = ml_speed.speed_kmph
            meta["ml_world_pos"] = ml_speed.world_pos
        triggers.append(RawTrigger(
            kind="anomaly_stop", track_ids=(tid,), t=t,
            meta=meta,
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
                ml_v = history.get_ml_speed(vid)
                ml_p = history.get_ml_speed(pid)
                meta = {"iou": iou, "ped_drop": ped_drop, "vehicle_v": v_now,
                        "cx": pv.cx, "cy": pv.cy}
                if ml_v:
                    meta["ml_vehicle_speed_mps"] = ml_v.speed_mps
                    meta["ml_vehicle_speed_kmph"] = ml_v.speed_kmph
                if ml_p:
                    meta["ml_ped_speed_mps"] = ml_p.speed_mps
                    meta["ml_ped_speed_kmph"] = ml_p.speed_kmph
                triggers.append(RawTrigger(
                    kind="hit_and_run", track_ids=(vid, pid), t=t,
                    meta=meta,
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
