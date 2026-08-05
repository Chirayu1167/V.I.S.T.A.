"""
ViolencePipeline: the second independent VISTA branch (violence/road rage).

Pose-based (yolo11n-pose): person keypoints are tracked and a violence signal
fires when two people are close together with aggressive limb motion (see
violence_heuristics.py). The chain mirrors the accident branch — Verifier ->
IncidentFuser -> SeverityAssessor -> AlertDispatcher — so both branches share
the same alert log, clip packaging, and dashboard wiring.

Like the accident branch it owns zero rendering; callers (demo.py, gui_app.py)
draw with render.draw_overlay / render.draw_skeletons. Runs at reduced cadence
(motion prefilter per design: the accident branch needs every frame, the
violence branch may skip) — pose detection runs every `pose_cadence_frames`.
"""

import os
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .alert import AlertDispatcher, VIOLENCE_CHANNELS_BY_SEVERITY
from .config import CameraConfig, DispatchConfig, HeuristicConfig, ViolenceConfig
from .fusion import IncidentFuser
from .severity import SeverityAssessor, SeverityConfig
from .tracker import Tracker
from .verification import Verifier
from .violence_heuristics import KEYPOINT_CONF_THRESHOLD, PoseHistory, check_violence


class PoseDetector:
    """Thin wrapper around Ultralytics YOLO pose (yolo11n-pose by default).
    Returns persons as (bbox_xyxy, confidence, kpts_xy) with low-visibility
    keypoints masked to NaN. GPU with CPU fallback, matching Detector."""

    def __init__(self, cfg: ViolenceConfig, device: str = "cuda", half: bool = True):
        from ultralytics import YOLO
        self.cfg = cfg
        self.model = YOLO(cfg.pose_weights)
        self.device = device
        self.half = half
        try:
            import torch
            if device == "cuda" and not torch.cuda.is_available():
                import warnings
                warnings.warn("CUDA not available for violence branch. Falling back to CPU.")
                self.device = "cpu"
        except ImportError:
            self.device = "cpu"
        if self.device != "cuda":
            self.half = False

    def detect(self, frame: np.ndarray) -> List[Tuple[Tuple[float, float, float, float], float, np.ndarray]]:
        """Returns [(bbox_xyxy, confidence, kpts_xy (17,2)), ...]; missing
        keypoints are NaN so heuristics can skip them."""
        results = self.model.predict(
            frame,
            device=self.device,
            half=self.half,
            conf=self.cfg.pose_conf_threshold,
            imgsz=self.cfg.pose_imgsz,
            verbose=False,
        )
        out = []
        if not results:
            return out
        boxes = results[0].boxes
        kps = getattr(results[0], "keypoints", None)
        if boxes is None or kps is None:
            return out
        data = kps.data.cpu().numpy()  # (N, 17, 3): x, y, conf
        for i, box in enumerate(boxes):
            if i >= len(data):
                break
            xyxy = tuple(box.xyxy[0].tolist())
            conf = float(box.conf[0])
            kpt_xy = data[i, :, :2].astype(np.float32)
            kpt_xy[data[i, :, 2] < KEYPOINT_CONF_THRESHOLD] = np.nan
            out.append((xyxy, conf, kpt_xy))
        return out


class ViolencePipeline:
    def __init__(self,
                 cfg: Optional[ViolenceConfig] = None,
                 camera_cfg: Optional[CameraConfig] = None,
                 dispatch_cfg: Optional[DispatchConfig] = None,
                 device: str = "cuda",
                 fps_hint: float = 25.0,
                 clip_dir: Optional[str] = None,
                 clip_pre_seconds: float = 2.0,
                 clip_post_seconds: float = 1.5,
                 clip_buffer_seconds: float = 4.0,
                 detector: Optional[PoseDetector] = None,
                 tracker: Optional[Tracker] = None):
        self.cfg = cfg or ViolenceConfig()
        self.camera_cfg = camera_cfg or CameraConfig()
        self.dispatch_cfg = dispatch_cfg or DispatchConfig()
        self.fps_hint = fps_hint

        self.detector = detector or PoseDetector(self.cfg, device=device)
        self.tracker = tracker or Tracker(frame_rate=int(fps_hint))
        check_rate = fps_hint / max(1, self.cfg.pose_cadence_frames)
        self.pose_history = PoseHistory(history_seconds=2.0, check_rate_hz=check_rate)

        vcfg = HeuristicConfig(verify_cooldown_s=self.cfg.verify_cooldown_s)
        vcfg.verify_window_frames_by_kind = {"violence": self.cfg.verify_window_frames}
        self.verifier = Verifier(vcfg)
        self.fuser = IncidentFuser(window_s=2.0, radius_px=max(120.0, self.cfg.pair_max_distance_px * 3))
        self.severity = SeverityAssessor()
        self.dispatcher = AlertDispatcher(self.camera_cfg, self.dispatch_cfg,
                                          channels_map=VIOLENCE_CHANNELS_BY_SEVERITY,
                                          alert_prefix="V")

        self.clip_dir = clip_dir
        self.clip_pre_seconds = clip_pre_seconds
        self.clip_post_seconds = clip_post_seconds
        effective_buffer_s = max(clip_buffer_seconds, clip_pre_seconds + clip_post_seconds + 1.0)
        self._clip_buffer = deque(maxlen=int(effective_buffer_s * fps_hint))
        self._pending_clips = []
        if self.clip_dir:
            os.makedirs(self.clip_dir, exist_ok=True)

        self.frame_count = 0
        self.confirmed_log = []           # in-memory record for CLI/GUI summaries
        self.latest_persons: List[Tuple[int, Tuple[float, float, float, float], np.ndarray]] = []
        check_rate = fps_hint / max(1, self.cfg.pose_cadence_frames)
        self._person_count_window = deque(
            maxlen=max(4, int(self.cfg.pair_max_persons_window_s * check_rate)))
        self._recent_person_count = 0

    def close(self):
        self.dispatcher.close()

    def process_frame(self, frame: np.ndarray, t: float) -> dict:
        """
        Runs one frame through the violence branch. Returns
        {"tracks": [...], "confirmed_events": [...], "alerts": [...],
         "clips_saved": [...], "persons": [...]}.
        """
        self.frame_count += 1
        self._clip_buffer.append((t, frame.copy()))

        alerts = []
        if self.frame_count % self.cfg.pose_cadence_frames == 0:
            persons = self.detector.detect(frame)
            tracked = self.tracker.update([(b, c, 0) for (b, c, _) in persons])
            pose_input = []
            for tid, bbox, _cls in tracked:
                kpts = self._match_keypoints(bbox, persons)
                if kpts is not None:
                    pose_input.append((tid, bbox, kpts))
            self.pose_history.update(t, pose_input)
            self.latest_persons = pose_input
            self._person_count_window.append(len(pose_input))
            self._recent_person_count = max(self._person_count_window)

            raw_triggers = check_violence(self.pose_history, t, self.cfg,
                                          person_count=self._recent_person_count)
            confirmed_events = self.fuser.process(t, self.verifier.process(t, raw_triggers))
            for event in confirmed_events:
                try:
                    dispatched = self._dispatch_event(event, t)
                    if dispatched is not None:
                        alerts.append(dispatched)
                except Exception as e:
                    self.confirmed_log.append((event, {}, f"dispatch_error: {e}"))

        saved_clips = self._process_pending_clips(t)
        return {
            "tracks": [(tid, bbox, 0) for (tid, bbox, _) in self.latest_persons],
            "confirmed_events": [],  # filled only on cadence frames; keep key for callers
            "alerts": alerts,
            "clips_saved": saved_clips,
            "persons": self.latest_persons,
        }

    @staticmethod
    def _match_keypoints(bbox, persons) -> Optional[np.ndarray]:
        """Associate a tracked bbox back to its pose keypoints via IoU
        (ByteTrack may reorder detections vs. the tracker's output)."""
        best, best_iou = None, 0.5
        for pb, _conf, kpts in persons:
            iou = _box_iou(bbox, pb)
            if iou > best_iou:
                best_iou, best = iou, kpts
        return best

    def _dispatch_event(self, event, t: float) -> Optional[object]:
        severity = self.severity.assess(event, None)
        clip_path = None
        if self.clip_dir:
            clip_path = os.path.join(self.clip_dir,
                                     f"pending-{self.camera_cfg.camera_id}-{int(event.t)}-{self.frame_count}.mp4")
        payload = self.dispatcher.build_and_dispatch(
            event, {"ran": False, "confidence": None}, clip_path, severity=severity)
        self.confirmed_log.append((event, {}, "dispatched"))
        if self.clip_dir:
            self._pending_clips.append({
                "alert_id": payload.alert_id,
                "path": clip_path,
                "t_impact": event.t,
                "due_t": event.t + self.clip_post_seconds,
            })
        return payload

    def _process_pending_clips(self, t: float) -> list:
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
            writer = cv2.VideoWriter(p["path"], cv2.VideoWriter_fourcc(*"mp4v"),
                                     self.fps_hint, (w, h))
            for f in frames:
                writer.write(f)
            writer.release()
            saved.append((p["alert_id"], p["path"]))
        return saved


def _box_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0
