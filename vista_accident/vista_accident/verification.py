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

            if st.streak >= self.cfg.verify_window_frames:
                confirmed.append(ConfirmedEvent(
                    kind=trig.kind, track_ids=trig.track_ids, t=t,
                    consecutive_frames=st.streak, meta=trig.meta,
                ))
                st.confirmed_at = t
                st.cooldown_until = t + self.cfg.verify_cooldown_s
                st.streak = 0  # reset so it must re-persist for a fresh alert later

        # Any event key that didn't fire this frame resets its streak — the
        # signal must persist on *consecutive* checks, not just cumulatively.
        for key, st in self._state.items():
            if key not in fired_keys:
                st.streak = 0

        return confirmed
