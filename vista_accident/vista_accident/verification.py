"""
Per-branch verification: a heuristic must fire on `verify_window_frames`
consecutive checks (i.e. persist across a short window of frames) before it
becomes a confirmed event. This is what cuts false positives from a single
noisy frame (a flickering detection box, a momentary tracker ID swap) without
adding real latency — verification only spans ~5 frames (~0.15-0.3s), not a
buffered replay.

A per-event cooldown then prevents the same track/pair from re-firing every
frame once it's already been alerted.
"""

import time
from dataclasses import dataclass
from typing import Dict, List

from .config import HeuristicConfig
from .heuristics import RawTrigger


@dataclass
class ConfirmedEvent:
    kind: str
    track_ids: tuple
    t: float
    consecutive_frames: int
    meta: dict


class _EventState:
    __slots__ = ("streak", "last_t", "confirmed_at", "cooldown_until")

    def __init__(self):
        self.streak = 0
        self.last_t = None
        self.confirmed_at = None
        self.cooldown_until = -1.0


class Verifier:
    def __init__(self, cfg: HeuristicConfig):
        self.cfg = cfg
        self._state: Dict[str, _EventState] = {}
        # Recently confirmed alerts (t, kind, cx, cy, track_ids) used for
        # spatial dedup: tracker ID swaps at impact re-key the same physical
        # incident, so a same-kind re-confirmation near the same spot is
        # treated as a repeat rather than a brand-new accident.
        self._recent_confirmed = []

    def process(self, t: float, raw_triggers: List[RawTrigger]) -> List[ConfirmedEvent]:
        confirmed = []
        fired_keys = set()

        for trig in raw_triggers:
            key = trig.key
            fired_keys.add(key)
            st = self._state.setdefault(key, _EventState())

            if t < st.cooldown_until:
                continue  # suppressed — recently alerted for this same track/pair

            st.streak += 1
            st.last_t = t

            window = self.cfg.verify_window_frames_by_kind.get(trig.kind,
                                                               self.cfg.verify_window_frames)
            if st.streak < window:
                continue

            if self._recently_confirmed(trig, t):
                st.streak = 0  # same physical incident re-keyed by tracker — suppress
                continue

            confirmed.append(ConfirmedEvent(
                kind=trig.kind, track_ids=trig.track_ids, t=t,
                consecutive_frames=st.streak, meta=trig.meta,
            ))
            st.confirmed_at = t
            st.cooldown_until = t + self.cfg.verify_cooldown_s
            st.streak = 0  # reset so it must re-persist for a fresh alert later
            self._recent_confirmed.append(
                (t, trig.kind, trig.meta.get("cx"), trig.meta.get("cy"), trig.track_ids))

        # Any event key that didn't fire this frame resets its streak — the
        # signal must persist on *consecutive* checks, not just cumulatively.
        for key, st in self._state.items():
            if key not in fired_keys:
                st.streak = 0

        # Forget old confirmations once their cooldown has lapsed.
        cutoff = t - self.cfg.verify_cooldown_s
        self._recent_confirmed = [r for r in self._recent_confirmed if r[0] > cutoff]

        return confirmed

    def _recently_confirmed(self, trig: RawTrigger, t: float) -> bool:
        """Dedup re-confirmations of the SAME physical incident — per-kind
        only (same tracker IDs or same spot within the cooldown)."""
        for r_t, r_kind, r_cx, r_cy, r_ids in self._recent_confirmed:
            if r_kind != trig.kind:
                continue
            if t - r_t > self.cfg.verify_cooldown_s:
                continue
            if set(r_ids) & set(trig.track_ids):
                return True  # same tracker IDs — direct repeat
            cx, cy = trig.meta.get("cx"), trig.meta.get("cy")
            if r_cx is not None and cx is not None:
                dist = ((r_cx - cx) ** 2 + (r_cy - cy) ** 2) ** 0.5
                if dist <= self.cfg.verify_dedup_radius_px:
                    return True  # same spot, new IDs (tracker ID churn at impact)
        return False
