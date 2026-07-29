"""
AccidentPipeline: ties detector + tracker + heuristics + verification +
secondary confirmation + alert dispatch into a single per-frame call.

This is ONE of the two independent VISTA branches (accident). It has zero
dependency on the violence branch — run it in its own thread/process/CUDA
stream alongside a ViolencePipeline so neither blocks the other. See
demo.py for a minimal single-branch runner, and README.md for the
concurrency wiring pattern.
"""

from collections import deque
from typing import Optional

import numpy as np

from .alert import AlertDispatcher
from .config import CameraConfig, DispatchConfig, HeuristicConfig
from .confirmation import SecondaryConfirmation
from .detector import Detector
from .heuristics import run_all_heuristics
from .severity import SeverityAssessor, SeverityConfig
from .track_history import TrackHistory
from .tracker import Tracker
from .verification import Verifier


class AccidentPipeline:
    def __init__(self,
                 detector: Optional[Detector] = None,
                 tracker: Optional[Tracker] = None,
                 heuristic_cfg: Optional[HeuristicConfig] = None,
                 camera_cfg: Optional[CameraConfig] = None,
                 dispatch_cfg: Optional[DispatchConfig] = None,
                 secondary: Optional[SecondaryConfirmation] = None,
                 severity_cfg: Optional[SeverityConfig] = None,
                 clip_buffer_seconds: float = 4.0,
                 fps_hint: float = 25.0):
        self.cfg = heuristic_cfg or HeuristicConfig()
        self.camera_cfg = camera_cfg or CameraConfig()
        self.dispatch_cfg = dispatch_cfg or DispatchConfig()

        self.detector = detector or Detector()
        self.tracker = tracker or Tracker(frame_rate=int(fps_hint))
        self.history = TrackHistory(history_seconds=self.cfg.history_seconds, assumed_fps=fps_hint)
        self.verifier = Verifier(self.cfg)
        self.severity = SeverityAssessor(severity_cfg)
        self.secondary = secondary or SecondaryConfirmation(weights_path=None)
        self.dispatcher = AlertDispatcher(self.camera_cfg, self.dispatch_cfg)

        # Rolling clip buffer for alert packaging ("clip, location, timestamp").
        self._clip_buffer = deque(maxlen=int(clip_buffer_seconds * fps_hint))

        self.frame_count = 0
        self.confirmed_log = []  # in-memory record for the demo/CLI summary

    def process_frame(self, frame: np.ndarray, t: float) -> dict:
        """
        Runs one frame through the full accident branch.
        Returns {"tracks": [...], "confirmed_events": [...], "alerts": [...]}
        """
        self.frame_count += 1
        self._clip_buffer.append(frame.copy())

        detections = self.detector.detect(frame)                # (bbox, conf, cls)
        tracks = self.tracker.update(detections)                # (track_id, bbox, cls)
        self.history.update(t, [(tid, bbox, cls) for tid, bbox, cls in tracks])

        raw_triggers = run_all_heuristics(self.history, t, self.cfg,
                                           stop_zones=self.camera_cfg.stop_zones)
        confirmed_events = self.verifier.process(t, raw_triggers)

        alerts = []
        for event in confirmed_events:
            secondary_result = self.secondary.confirm(frame)
            # If secondary confirmation ran and explicitly rejected it, skip dispatch —
            # otherwise (not run, or confirmed) proceed on heuristic+verification alone.
            if secondary_result["ran"] and not secondary_result["confirmed"]:
                self.confirmed_log.append((event, secondary_result, "rejected_by_secondary"))
                continue

            severity = self.severity.assess(event, self.history)
            clip_path = None  # wire up e.g. self._save_clip(event) for real clip export
            payload = self.dispatcher.build_and_dispatch(event, secondary_result, clip_path, severity=severity)
            alerts.append(payload)
            self.confirmed_log.append((event, secondary_result, "dispatched"))

        return {"tracks": tracks, "confirmed_events": confirmed_events, "alerts": alerts}
