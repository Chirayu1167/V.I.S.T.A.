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
from .fusion import IncidentFuser
from .heuristics import run_all_heuristics
from .severity import SeverityAssessor, SeverityConfig
from .speed_estimator import MlSpeedEstimator, create_speed_estimator_from_config
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
                 fps_hint: float = 25.0,
                 use_ml_speed: bool = True):
        self.cfg = heuristic_cfg or HeuristicConfig()
        self.camera_cfg = camera_cfg or CameraConfig()
        self.dispatch_cfg = dispatch_cfg or DispatchConfig()
        self.fps_hint = fps_hint

        self.detector = detector or Detector()
        self.tracker = tracker or Tracker(frame_rate=int(fps_hint))
        
        # Initialize ML speed estimator if enabled
        self.speed_estimator = None
        if use_ml_speed:
            self.speed_estimator = create_speed_estimator_from_config(self.camera_cfg)
        
        self.history = TrackHistory(
            history_seconds=self.cfg.history_seconds, 
            assumed_fps=fps_hint,
            speed_estimator=self.speed_estimator
        )
        self.verifier = Verifier(self.cfg)
        self.fuser = IncidentFuser()
        self.severity = SeverityAssessor(severity_cfg)
        self.secondary = secondary or SecondaryConfirmation(weights_path=None)
        self.dispatcher = AlertDispatcher(self.camera_cfg, self.dispatch_cfg)

        # Rolling clip buffer for alert packaging ("clip, location, timestamp").
        self._clip_buffer = deque(maxlen=int(clip_buffer_seconds * fps_hint))

        self.frame_count = 0
        self.confirmed_log = []  # in-memory record for the demo/CLI summary

    def _pick_impact_frame(self, event) -> Optional[np.ndarray]:
        """The stored raw frame nearest to the moment the impact actually
        happened (the confirmation frame is a few frames too late — the
        crash may only be visible 1-2 frames after the trigger fired)."""
        t_impact = event.t - (event.consecutive_frames - 1) / self.fps_hint
        best, best_d = None, float("inf")
        for bt, bf in self._clip_buffer:
            d = abs(bt - t_impact)
            if d < best_d:
                best_d, best = d, bf
        return best

    def process_frame(self, frame: np.ndarray, t: float) -> dict:
        """
        Runs one frame through the full accident branch.
        Returns {"tracks": [...], "confirmed_events": [...], "alerts": [...], "speeds": {...}}
        """
        self.frame_count += 1
        self._clip_buffer.append((t, frame.copy()))

        detections = self.detector.detect(frame)                # (bbox, conf, cls)
        tracks = self.tracker.update(detections)                # (track_id, bbox, cls)
        # Speed estimation reuses these exact tracks (same detector, same
        # track IDs) rather than re-running detection internally.
        self.history.update(t, [(tid, bbox, cls) for tid, bbox, cls in tracks])

        raw_triggers = run_all_heuristics(self.history, t, self.cfg,
                                           stop_zones=self.camera_cfg.stop_zones)
        confirmed_events = self.fuser.process(
            t, self.verifier.process(t, raw_triggers))

        alerts = []
        for event in confirmed_events:
            # Confirm against the stored impact frame (the moment the crash
            # was first seen), not the current frame — by confirmation time
            # the scene has moved on a few frames.
            impact_frame = self._pick_impact_frame(event)
            if impact_frame is None:
                impact_frame = frame
            secondary_result = self.secondary.confirm(impact_frame)
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

        # Include ML speed estimates in output
        speeds = {}
        if self.speed_estimator:
            speeds = self.speed_estimator.get_all_speeds()

        return {
            "tracks": tracks, 
            "confirmed_events": confirmed_events, 
            "alerts": alerts,
            "speeds": speeds
        }
