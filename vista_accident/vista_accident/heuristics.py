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
from typing import List, Optional, Tuple

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
        fired = drop_ratio > cfg.speed_drop_ratio

        # FAST-DROP branch: speed readings from a tracker on distant CCTV are
        # noisy, so a real wall/barrier crash-stop can collapse in only a few
        # frames and be diluted by the windowed average below the >80% bar —
        # exactly the accidents we must not miss. The Kalman instantaneous
        # velocity (innovation gate re-inits on hard stops) snaps toward zero
        # immediately, so near-stopped instantaneous + a big deceleration over
        # a SHORT recent span is the crash signature. Thresholds are set loose
        # on purpose (they only fire when the speed really collapsed AND the
        # vehicle is effectively stopped), so normal gradual braking at a light
        # doesn't qualify.
        now_inst = None
        decel = None
        if not fired and cfg.speed_drop_fast_decel_mps2 > 0:
            fw = cfg.speed_drop_fast_window_s
            prior_recent_v = history.velocity_between(tid, t - 2 * fw, t - fw)
            recent_v = history.velocity_between(tid, t - fw, t)
            now_inst = history.instantaneous_velocity(tid)
            if (prior_recent_v is not None and recent_v is not None
                    and now_inst is not None and now_inst < prior_recent_v
                    and prior_recent_v >= cfg.speed_drop_min_prior_speed):
                decel = (prior_recent_v - recent_v) / fw
                fired = (
                    now_inst <= cfg.speed_drop_fast_end_max_velocity
                    and decel >= cfg.speed_drop_fast_decel_mps2
                )

        if not fired:
            continue
        # Include ML speed details if available
        ml_speed = history.get_ml_speed(tid)
        # Score the fast branch with the instantaneous collapse ratio (the
        # windowed ratio understates a fast crash-stop — that's why the
        # windowed branch missed it).
        if now_inst is not None and now_inst < prior_v:
            meta_ratio = (prior_v - now_inst) / prior_v
        else:
            meta_ratio = drop_ratio
        meta = {"prior_v": prior_v,
                "now_v": now_inst if now_inst is not None else now_v,
                "drop_ratio": meta_ratio, "cx": p.cx, "cy": p.cy}
        if decel is not None:
            meta["decel_mps2"] = round(decel, 2)
            meta["fast"] = True
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


class CollisionEvidence:
    """Causal memory for impact detection.

    A real impact leaves TWO signatures that are usually NOT simultaneous:
    (1) two moving vehicles overlap at high IoU for a frame or two, then
    (2) one of them collapses to near-zero speed. Post-impact trajectories
    can separate the boxes faster than the verifier window, so requiring
    "overlap AND collapse in the SAME frame" (and for N consecutive frames)
    misses real crashes (clip_07: overlap IoU 0.54 for ~1 frame while the
    struck car still reads ~8 m/s, stops ~0.2s later).

    This keeps the pre-impact (prior) speeds of recent high-speed overlaps;
    if either vehicle's speed collapses within collision_collapse_window_s
    of the overlap, the collision fires — same-frame (clip_03-style) or
    shortly after (clip_07-style), and while the collapse persists the
    verifier gets its consecutive frames.
    """

    def __init__(self, window_s: float = 0.8):
        self.window_s = window_s
        self._entries = {}  # sorted pair -> {"t":, "prior_a":, "prior_b":, "iou":, "cx":, "cy":}

    @staticmethod
    def _key(a: int, b: int):
        return (a, b) if a < b else (b, a)

    def remember(self, id_a: int, id_b: int, prior_a, prior_b, iou, cx, cy, t: float,
                 now_a=None, now_b=None) -> None:
        key = self._key(id_a, id_b)
        entry = self._entries.get(key)
        if entry is None:
            entry = {"t": t, "prior_a": prior_a, "prior_b": prior_b,
                     "iou": iou, "peak_iou": iou, "cx": cx, "cy": cy,
                     "now_a": now_a, "now_b": now_b}
            self._entries[key] = entry
        else:
            entry.update({"t": t, "iou": iou, "cx": cx, "cy": cy,
                          "now_a": now_a, "now_b": now_b})
            entry["peak_iou"] = max(entry["peak_iou"], iou)

    def active_entries(self, t: float):
        cutoff = t - self.window_s
        stale = [k for k, e in self._entries.items() if e["t"] < cutoff]
        for k in stale:
            del self._entries[k]
        return self._entries


def check_collision(history: TrackHistory, t: float, cfg: HeuristicConfig,
                    evidence: Optional[CollisionEvidence] = None) -> List[RawTrigger]:
    """Impact detection. `evidence` carries recent high-speed overlaps across
    frames (see CollisionEvidence) so the collapse can be confirmed shortly
    after a brief overlap instead of needing both at the same instant."""
    triggers = []
    evidence = evidence or CollisionEvidence(window_s=cfg.collision_collapse_window_s)
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
            if (prior_a is None or prior_a < cfg.collision_min_prior_speed) and \
               (prior_b is None or prior_b < cfg.collision_min_prior_speed):
                continue  # neither was moving meaningfully — not an impact
            evidence.remember(id_a, id_b, prior_a, prior_b, iou,
                              (pa.cx + pb.cx) / 2.0, (pa.cy + pb.cy) / 2.0, t,
                              now_a=history.instantaneous_velocity(id_a),
                              now_b=history.instantaneous_velocity(id_b))

    for (id_a, id_b), e in evidence.active_entries(t).items():
        now_a = history.instantaneous_velocity(id_a)
        now_b = history.instantaneous_velocity(id_b)
        hit_a = _impacted(e["prior_a"], now_a, cfg)
        hit_b = _impacted(e["prior_b"], now_b, cfg)
        if not (hit_a or hit_b):
            continue
        # Discriminate a real impact from a queue-tail panic stop (clip_04 FP):
        # either the velocity was ALREADY collapsing at the overlap instant
        # (ratio ≤ collision_overlap_collapse_factor — crash physics started at
        # contact, clip_07) or the contact itself was deep (peak IoU ≥ the
        # buffer threshold — struck at speed, clip_03-style).
        ratio_a = (e["now_a"] / e["prior_a"]) if (e["now_a"] is not None and e["prior_a"]) else None
        ratio_b = (e["now_b"] / e["prior_b"]) if (e["now_b"] is not None and e["prior_b"]) else None
        collapse_at_overlap = (
            (hit_a and ratio_a is not None and ratio_a <= cfg.collision_overlap_collapse_factor)
            or (hit_b and ratio_b is not None and ratio_b <= cfg.collision_overlap_collapse_factor))
        deep_contact = e["peak_iou"] >= cfg.collision_buffer_iou_threshold
        if not (collapse_at_overlap or deep_contact):
            continue
        # Include ML speed details if available
        ml_a = history.get_ml_speed(id_a)
        ml_b = history.get_ml_speed(id_b)
        meta = {"iou": e["iou"], "v_a": now_a, "v_b": now_b,
                "prior_v_a": e["prior_a"], "prior_v_b": e["prior_b"],
                "cx": e["cx"], "cy": e["cy"]}
        if collapse_at_overlap:
            meta["collapse_at_overlap"] = True
        if deep_contact:
            meta["peak_iou"] = round(e["peak_iou"], 2)
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


def run_all_heuristics(history: TrackHistory, t: float, cfg: HeuristicConfig, stop_zones=None,
                       collision_evidence: Optional[CollisionEvidence] = None) -> List[RawTrigger]:
    triggers = []
    triggers += check_speed_drop(history, t, cfg, stop_zones=stop_zones)
    triggers += check_collision(history, t, cfg, evidence=collision_evidence)
    triggers += check_anomaly_stop(history, t, cfg, stop_zones=stop_zones)
    triggers += check_hit_and_run(history, t, cfg)
    return triggers
