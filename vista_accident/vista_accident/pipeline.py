"""
AccidentPipeline: ties detector + tracker + heuristics + verification +
secondary confirmation + alert dispatch into a single per-frame call.

This is ONE of the two independent VISTA branches (accident). It has zero
dependency on the violence branch — run it in its own thread/process/CUDA
stream alongside a ViolencePipeline so neither blocks the other. See
demo.py for a minimal single-branch runner, and README.md for the
concurrency wiring pattern.
"""

import os
from collections import deque
from typing import Optional

import cv2
import numpy as np

from .alert import AlertDispatcher
from .config import (HOMOGRAPHY_THRESHOLD_SCALE, CameraConfig, DispatchConfig,
                     HeuristicConfig, scale_thresholds)
from .confirmation import SecondaryConfirmation
from .detector import Detector
from .fusion import IncidentFuser
from .heuristics import CollisionEvidence, run_all_heuristics
from .plate_reader import PlateReader
from .severity import SeverityAssessor, SeverityConfig
from .speed_estimator import create_speed_estimator_from_config
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
                 use_ml_speed: bool = True,
                 clip_dir: Optional[str] = None,
                 clip_pre_seconds: float = 2.0,
                 clip_post_seconds: float = 1.5,
                 plate_reader: Optional[PlateReader] = None,
                 threshold_scale: Optional[float] = None):
        self.cfg = heuristic_cfg or HeuristicConfig()
        self.camera_cfg = camera_cfg or CameraConfig()
        self.dispatch_cfg = dispatch_cfg or DispatchConfig()
        self.fps_hint = fps_hint

        # Re-anchor thresholds: homography speeds read systematically lower
        # than the flat 0.05 m/px scale the defaults were tuned on, so when a
        # fitted homography is present the physical-velocity thresholds are
        # multiplied by HOMOGRAPHY_THRESHOLD_SCALE (see config.py docstring).
        if threshold_scale is None:
            has_homography = bool(getattr(self.camera_cfg, "homography_src_points", None)
                                  and getattr(self.camera_cfg, "homography_dst_points", None))
            threshold_scale = HOMOGRAPHY_THRESHOLD_SCALE if has_homography else 1.0
        if threshold_scale != 1.0:
            scale_thresholds(self.cfg, threshold_scale)

        self.detector = detector or Detector()
        self.tracker = tracker or Tracker(frame_rate=int(fps_hint))
        
        # Initialize ML speed estimator if enabled
        self.speed_estimator = None
        if use_ml_speed:
            self.speed_estimator = create_speed_estimator_from_config(self.camera_cfg)
        
        self.history = TrackHistory(
            history_seconds=self.cfg.history_seconds, 
            assumed_fps=fps_hint,
            speed_estimator=self.speed_estimator,
            meter_per_pixel=getattr(self.camera_cfg, "meter_per_pixel", 0.05),
        )
        self.verifier = Verifier(self.cfg)
        self._collision_evidence = CollisionEvidence(
            window_s=self.cfg.collision_collapse_window_s)
        self.fuser = IncidentFuser(
            window_s=self.cfg.fusion_window_s,
            radius_px=self.cfg.fusion_radius_px,
        )
        self.severity = SeverityAssessor(severity_cfg)
        self.secondary = secondary or SecondaryConfirmation(weights_path=None)
        self.dispatcher = AlertDispatcher(self.camera_cfg, self.dispatch_cfg)
        self.plate_reader = plate_reader  # optional; only used on hit_and_run events

        # Rolling clip buffer for alert packaging ("clip, location, timestamp").
        # Sized to cover pre- AND post-impact frames for real clip export
        # (see clip_dir), not just the impact-frame lookup below.
        self.clip_dir = clip_dir
        self.clip_pre_seconds = clip_pre_seconds
        self.clip_post_seconds = clip_post_seconds
        effective_buffer_s = max(clip_buffer_seconds, clip_pre_seconds + clip_post_seconds + 1.0)
        self._clip_buffer = deque(maxlen=int(effective_buffer_s * fps_hint))
        self._pending_clips = []  # [{"alert_id":, "t_impact":, "due_t":}]
        if self.clip_dir:
            os.makedirs(self.clip_dir, exist_ok=True)

        self.frame_count = 0
        self.confirmed_log = []  # in-memory record for the demo/CLI summary

    def close(self):
        """Flush and stop the alert dispatcher's background log thread.
        Call this once after the frame loop ends (see demo.py/gui_app.py) —
        without it, alerts still queued when the process exits can be lost."""
        self.dispatcher.close()

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

    def process_frame(self, frame: np.ndarray, t: float, detections: Optional[list] = None) -> dict:
        """
        Runs one frame through the full accident branch.
        Returns {"tracks": [...], "confirmed_events": [...], "alerts": [...], "speeds": {...}}

        `detections`: optional pre-computed detector.detect() output for
        this frame — [(bbox_xyxy, conf, cls), ...]. Pass this in when a
        shared batch_inference worker already ran Detector.detect_batch()
        across cameras this cycle (see batch_inference.py); it's used as-is
        and self.detector.detect() is skipped for this frame. Leave it None
        (default) for the original self-contained behavior (single camera /
        demo.py / gui_app.py / test_scenario.py), which is unchanged.
        """
        self.frame_count += 1
        self._clip_buffer.append((t, frame.copy()))

        detections = detections if detections is not None else self.detector.detect(frame)  # (bbox, conf, cls)
        tracks = self.tracker.update(detections)                # (track_id, bbox, cls)
        # Speed estimation reuses these exact tracks (same detector, same
        # track IDs) rather than re-running detection internally.
        self.history.update(t, [(tid, bbox, cls) for tid, bbox, cls in tracks])

        raw_triggers = run_all_heuristics(self.history, t, self.cfg,
                                           stop_zones=self.camera_cfg.stop_zones,
                                           collision_evidence=self._collision_evidence)
        confirmed_events = self.fuser.process(
            t, self.verifier.process(t, raw_triggers))

        alerts = []
        for event in confirmed_events:
            try:
                dispatched = self._dispatch_event(event, frame, t, tracks)
                if dispatched is not None:
                    alerts.append(dispatched)
            except Exception as e:  # one bad event must never kill the whole feed
                self.confirmed_log.append((event, {}, f"dispatch_error: {e}"))
        saved_clips = self._process_pending_clips(t)

        # Include ML speed estimates in output
        speeds = {}
        if self.speed_estimator:
            speeds = self.speed_estimator.get_all_speeds()

        return {
            "tracks": tracks, 
            "confirmed_events": confirmed_events, 
            "alerts": alerts,
            "speeds": speeds,
            "clips_saved": saved_clips,
        }

    def _dispatch_event(self, event, frame: np.ndarray, _t: float,
                        tracks: list) -> Optional[object]:
        """Dispatch one confirmed event: secondary confirmation against the
        stored impact frame, severity assessment, optional plate OCR, clip
        path reservation, then dispatch + pending-clip scheduling. Returns
        the AlertPayload, or None if secondary confirmation rejected it.
        Kept separate from process_frame so a failing event can be skipped
        without killing the whole feed."""
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
            return None

        severity = self.severity.assess(event, self.history)

        # Plate OCR only for hit_and_run — the one event kind where "the
        # vehicle kept moving" makes identifying it the actual point.
        if self.plate_reader is not None and self.plate_reader.enabled and event.kind == "hit_and_run":
            vehicle_bbox = event.meta.get("vehicle_bbox")
            if vehicle_bbox is None:
                # heuristics.py doesn't currently store the raw bbox in
                # meta — fall back to the current tracked box for the
                # vehicle track id if still visible this frame.
                vehicle_tid = event.track_ids[0]
                vt = next((tr for tr in tracks if tr[0] == vehicle_tid), None)
                vehicle_bbox = vt[1] if vt else None
            if vehicle_bbox is not None:
                crop = PlateReader.crop_bbox(impact_frame, vehicle_bbox)
                plate_result = self.plate_reader.read(crop)
                if plate_result.get("plate_text"):
                    event.meta["plate_text"] = plate_result["plate_text"]
                    event.meta["plate_confidence"] = plate_result["confidence"]

        clip_path = None
        if self.clip_dir:
            # Reserve the path now (dispatch isn't delayed for it); the
            # actual file is written a few frames later once the buffer
            # has enough post-impact frames — see _process_pending_clips.
            clip_path = os.path.join(self.clip_dir, f"pending-{self.camera_cfg.camera_id}-{int(event.t)}-{self.frame_count}.mp4")

        payload = self.dispatcher.build_and_dispatch(event, secondary_result, clip_path, severity=severity)
        self.confirmed_log.append((event, secondary_result, "dispatched"))

        if self.clip_dir:
            self._pending_clips.append({
                "alert_id": payload.alert_id,
                "path": clip_path,
                "t_impact": event.t,
                "due_t": event.t + self.clip_post_seconds,
            })
        return payload

    def _process_pending_clips(self, t: float) -> list:
        """Writes out any pending clip whose post-impact window has now
        fully landed in the buffer. Returns [(alert_id, path), ...] for
        clips written this call."""
        if not self._pending_clips:
            return []
        ready, still_pending = [], []
        for p in self._pending_clips:
            (ready if t >= p["due_t"] else still_pending).append(p)
        self._pending_clips = still_pending

        saved = []
        for p in ready:
            frames = [f for (bt, f) in self._clip_buffer
                      if p["t_impact"] - self.clip_pre_seconds <= bt <= p["t_impact"] + self.clip_post_seconds]
            if not frames:
                continue
            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(p["path"], cv2.VideoWriter_fourcc(*"mp4v"), self.fps_hint, (w, h))
            for f in frames:
                writer.write(f)
            writer.release()
            saved.append((p["alert_id"], p["path"]))
        return saved
