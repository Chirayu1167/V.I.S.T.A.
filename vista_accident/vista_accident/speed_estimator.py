"""
Real-world speed estimator for a FIXED camera.

Takes (track_id, bbox, cls) triples from the pipeline's own Detector +
Tracker (the same detections/track IDs that heuristics.py reads from) and
converts each box's ground-contact point into real-world meters via a
calibrated homography, then fits velocity with least-squares regression
over a rolling time window.

Why this replaced the previous version
---------------------------------------
The old implementation ran its own internal YOLO(yolov8n) + ByteTrack pass
purely for speed, separate from the main Detector/Tracker used everywhere
else in the pipeline. That meant:

  1. GPU inference ran twice per frame for no benefit.
  2. Its track IDs came from a *different* tracker instance than the IDs
     heuristics.py uses, so `history.get_ml_speed(tid)` was looking up the
     wrong ID space and essentially never matched the right track.
  3. It used the bounding-box *centroid*, which is not on the ground plane
     for an angled/overhead camera -- its pixel position drifts as box
     height changes (partial occlusion, detector jitter), which injects
     noise straight into the homography-transformed world position.
  4. Speed was computed from only the first and last sample in the
     window, so a single noisy detection could swing the whole estimate.
  5. `CameraCalibration` silently fell back to a guessed
     `meter_per_pixel` scale if no homography was supplied -- i.e. speeds
     were not actually calibrated to the camera at all unless someone
     filled in homography points.

This version fixes all five: it consumes the pipeline's own tracks (IDs
always line up and detection only runs once), uses the ground-contact
point (bottom-center of the bbox), fits velocity with least-squares over
the requested time window, and warns loudly if you haven't calibrated the
homography yet (see tools/calibrate_camera.py).
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings

import cv2
import numpy as np


@dataclass
class CameraCalibration:
    """Camera calibration for pixel -> real-world-meter conversion.

    For a fixed camera, ALWAYS calibrate with src_points/dst_points
    (a homography fit from 4+ known ground-plane correspondences) rather
    than relying on meter_per_pixel. meter_per_pixel assumes uniform scale
    across the whole frame, which is wrong for anything but a perfectly
    top-down camera -- for an angled camera, near-camera pixels cover far
    less real-world distance than far-camera pixels, so a single scale
    factor will be significantly off for at least part of the frame.

    Use tools/calibrate_camera.py to generate src_points/dst_points by
    clicking known reference points (e.g. lane markings, crosswalk
    corners, a measured rectangle) in a frame from your actual camera.
    """
    src_points: List[Tuple[float, float]] = None
    dst_points: List[Tuple[float, float]] = None
    meter_per_pixel: float = 0.05
    camera_height_m: float = 8.0
    camera_pitch_deg: float = 45.0

    def __post_init__(self):
        self.homography = None
        self.inv_homography = None

        if self.src_points and self.dst_points and len(self.src_points) >= 4 and len(self.dst_points) >= 4:
            src_np = np.array(self.src_points, dtype=np.float32)
            dst_np = np.array(self.dst_points, dtype=np.float32)
            self.homography, _ = cv2.findHomography(src_np, dst_np)
            self.inv_homography, _ = cv2.findHomography(dst_np, src_np)
        else:
            warnings.warn(
                "CameraCalibration has no homography (src_points/dst_points not set). "
                "Falling back to a flat meter_per_pixel scale -- speeds will NOT be "
                "accurate for an angled/perspective camera. Run tools/calibrate_camera.py "
                "against a frame from your actual camera and set homography_src_points / "
                "homography_dst_points in CameraConfig.",
                stacklevel=2,
            )


@dataclass
class TrackSpeed:
    """Speed information for a single track."""
    track_id: int
    speed_mps: float
    speed_kmph: float
    world_pos: Tuple[float, float]
    is_locked: bool
    history_length: int


class MlSpeedEstimator:
    """
    Real-world speed estimator that rides on top of the pipeline's own
    Detector + Tracker output (no separate detection/tracking pass).

    Usage: call update(t, tracks) once per frame with the SAME
    (track_id, bbox, cls) tuples produced by Tracker.update(), using the
    same clock `t` passed to the rest of the pipeline.
    """

    def __init__(
        self,
        fps: float = 25.0,
        calibration: Optional[CameraCalibration] = None,
        history_seconds: float = 2.0,
        min_history: int = 5,
        max_speed_kmph: float = 150.0,
        lock_after_seconds: Optional[float] = None,
    ):
        """
        Args:
            fps: nominal video frame rate (used only to size history buffers)
            calibration: CameraCalibration (homography strongly recommended)
            history_seconds: how much world-position history to retain per track
            min_history: minimum samples before a speed can be computed
            max_speed_kmph: clamp/outlier-filter ceiling
            lock_after_seconds: if set, speed estimate freezes (and history is
                freed) once a track has this many seconds of history -- use for
                the "how fast was this vehicle going" style outputs where you
                want one stable number rather than a constantly-updating one.
                Leave None to keep re-estimating every frame (recommended for
                heuristics like sudden-deceleration detection).
        """
        self.fps = fps
        self.min_history = min_history
        self.max_speed_kmph = max_speed_kmph
        self.history_seconds = history_seconds
        self.lock_after_seconds = lock_after_seconds

        self.calibration = calibration or CameraCalibration()

        self._maxlen = max(30, int(history_seconds * fps * 2))
        # track_id -> deque[(t, world_x, world_y, bbox, cls)]
        self.track_histories: Dict[int, deque] = {}
        self.track_speeds: Dict[int, TrackSpeed] = {}
        self.locked_tracks: set = set()
        self._last_t: float = 0.0

    # -- coordinate transforms ------------------------------------------------

    @staticmethod
    def _ground_point(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Bottom-center of the bbox: the point where the object contacts the
        ground plane. This is what a ground-plane homography is calibrated
        against -- the box centroid is NOT on the ground plane for anything
        but a perfectly overhead camera."""
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, y2

    def _pixel_to_world(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        if self.calibration.homography is not None:
            pt = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
            world_pt = cv2.perspectiveTransform(pt, self.calibration.homography)
            return float(world_pt[0, 0, 0]), float(world_pt[0, 0, 1])
        return pixel_x * self.calibration.meter_per_pixel, pixel_y * self.calibration.meter_per_pixel

    # -- main update ------------------------------------------------------------

    def update(
        self,
        t: float,
        tracks: List[Tuple[int, Tuple[float, float, float, float], int]],
    ) -> Dict[int, TrackSpeed]:
        """
        Args:
            t: current timestamp (seconds), same clock used elsewhere in the pipeline
            tracks: list of (track_id, bbox_xyxy, class_id) -- pass the exact
                output of Tracker.update() so IDs match the rest of the pipeline
        """
        self._last_t = t
        current_ids = set()

        for track_id, bbox, cls in tracks:
            current_ids.add(track_id)
            if track_id in self.locked_tracks:
                continue

            px, py = self._ground_point(bbox)
            wx, wy = self._pixel_to_world(px, py)

            hist = self.track_histories.setdefault(track_id, deque(maxlen=self._maxlen))
            hist.append((t, wx, wy, bbox, cls))

            if len(hist) >= self.min_history:
                self._compute_speed(track_id, window_s=self.history_seconds)

        self._age_tracks(current_ids, t)
        return self.track_speeds

    def _fit_velocity(self, points) -> Optional[Tuple[float, float, float]]:
        """Least-squares linear fit of (world_x, world_y) vs t over `points`.
        Returns (speed_mps, world_x_now, world_y_now) or None if degenerate.
        Using a full-window regression (vs. just first/last sample) makes the
        estimate robust to a single noisy detection or a brief partial
        occlusion."""
        if len(points) < 2:
            return None
        times = np.array([p[0] for p in points], dtype=np.float64)
        xs = np.array([p[1] for p in points], dtype=np.float64)
        ys = np.array([p[2] for p in points], dtype=np.float64)

        span = times[-1] - times[0]
        if span <= 0:
            return None

        t0 = times - times[0]
        A = np.vstack([t0, np.ones_like(t0)]).T
        vx, _ = np.linalg.lstsq(A, xs, rcond=None)[0]
        vy, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
        speed_mps = float(np.hypot(vx, vy))
        return speed_mps, float(xs[-1]), float(ys[-1])

    def _compute_speed(self, track_id: int, window_s: float) -> None:
        hist = self.track_histories.get(track_id)
        if not hist:
            return
        latest_t = hist[-1][0]
        window_pts = [p for p in hist if latest_t - p[0] <= window_s]

        fit = self._fit_velocity(window_pts)
        if fit is None:
            return
        speed_mps, wx, wy = fit

        speed_kmph = speed_mps * 3.6
        if speed_kmph > self.max_speed_kmph:
            speed_kmph = self.max_speed_kmph
            speed_mps = speed_kmph / 3.6

        is_locked = (
            self.lock_after_seconds is not None
            and (latest_t - hist[0][0]) >= self.lock_after_seconds
        )

        self.track_speeds[track_id] = TrackSpeed(
            track_id=track_id,
            speed_mps=speed_mps,
            speed_kmph=speed_kmph,
            world_pos=(wx, wy),
            is_locked=is_locked,
            history_length=len(hist),
        )

        if is_locked:
            self.locked_tracks.add(track_id)
            self.track_histories.pop(track_id, None)

    def _age_tracks(self, current_ids: set, now: float) -> None:
        stale_ids = [
            tid for tid, hist in self.track_histories.items()
            if tid not in current_ids and hist and now - hist[-1][0] > self.history_seconds
        ]
        for tid in stale_ids:
            self.track_histories.pop(tid, None)
            self.track_speeds.pop(tid, None)
            self.locked_tracks.discard(tid)
        for tid in list(self.track_speeds.keys()):
            if tid not in current_ids and tid not in self.track_histories and tid not in self.locked_tracks:
                self.track_speeds.pop(tid, None)

    # -- query interface (compatible with TrackHistory's pixel-based API) -----

    def get_speed(self, track_id: int) -> Optional[TrackSpeed]:
        return self.track_speeds.get(track_id)

    def get_all_speeds(self) -> Dict[int, TrackSpeed]:
        return self.track_speeds.copy()

    def velocity(self, track_id: int, window_s: float) -> Optional[float]:
        """Average speed (m/s) over the last `window_s` seconds, fit fresh
        over exactly that window (independent of the estimator's default
        history_seconds window used for the "current" TrackSpeed)."""
        hist = self.track_histories.get(track_id)
        if not hist:
            speed = self.track_speeds.get(track_id)
            return speed.speed_mps if speed else None
        latest_t = hist[-1][0]
        window_pts = [p for p in hist if latest_t - p[0] <= window_s]
        fit = self._fit_velocity(window_pts)
        return fit[0] if fit else None

    def velocity_between(self, track_id: int, t0: float, t1: float) -> Optional[float]:
        """Average speed (m/s) using samples with t in [t0, t1]."""
        hist = self.track_histories.get(track_id)
        if not hist:
            speed = self.track_speeds.get(track_id)
            return speed.speed_mps if speed else None
        lo, hi = min(t0, t1), max(t0, t1)
        window_pts = [p for p in hist if lo <= p[0] <= hi]
        fit = self._fit_velocity(window_pts)
        return fit[0] if fit else None

    def instantaneous_velocity(self, track_id: int) -> Optional[float]:
        hist = self.track_histories.get(track_id)
        if not hist or len(hist) < 2:
            speed = self.track_speeds.get(track_id)
            return speed.speed_mps if speed else None
        fit = self._fit_velocity(list(hist)[-min(len(hist), self.min_history):])
        return fit[0] if fit else None

    def stationary_duration(self, track_id: int, max_velocity: float) -> float:
        hist = self.track_histories.get(track_id)
        if not hist or len(hist) < 2:
            return 0.0
        pts = list(hist)
        end_t = pts[-1][0]
        start_t = end_t
        for i in range(len(pts) - 1, 0, -1):
            t2, x2, y2 = pts[i][0], pts[i][1], pts[i][2]
            t1, x1, y1 = pts[i - 1][0], pts[i - 1][1], pts[i - 1][2]
            dt = t2 - t1
            if dt <= 0:
                continue
            v = np.hypot(x2 - x1, y2 - y1) / dt
            if v <= max_velocity:
                start_t = t1
            else:
                break
        return end_t - start_t

    def draw_speeds(self, frame: np.ndarray, current_tracks=None) -> np.ndarray:
        """Draw speed annotations on frame. Pass the current frame's
        (track_id, bbox, cls) tracks so boxes are drawn even for tracks
        already locked/freed from internal history."""
        annotated = frame.copy()
        boxes_by_id = {}
        if current_tracks:
            for tid, bbox, cls in current_tracks:
                boxes_by_id[tid] = bbox
        for track_id, speed in self.track_speeds.items():
            bbox = boxes_by_id.get(track_id)
            if bbox is None and track_id in self.track_histories:
                bbox = self.track_histories[track_id][-1][3]
            if bbox is None:
                continue
            x1, y1, x2, y2 = map(int, bbox)
            label = f"ID:{track_id} {speed.speed_kmph:.1f} km/h"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return annotated


def create_speed_estimator_from_config(config) -> MlSpeedEstimator:
    """Factory function to create MlSpeedEstimator from CameraConfig."""
    calibration = CameraCalibration(
        meter_per_pixel=getattr(config, 'meter_per_pixel', 0.05),
        camera_height_m=getattr(config, 'camera_height_m', 8.0),
        camera_pitch_deg=getattr(config, 'camera_pitch_deg', 45.0),
        src_points=getattr(config, 'homography_src_points', None) or None,
        dst_points=getattr(config, 'homography_dst_points', None) or None,
    )

    return MlSpeedEstimator(
        fps=getattr(config, 'fps', 25.0),
        calibration=calibration,
        history_seconds=getattr(config, 'speed_history_seconds', 2.0),
        min_history=getattr(config, 'speed_min_history', 5),
        max_speed_kmph=getattr(config, 'speed_max_kmph', 150.0),
        lock_after_seconds=getattr(config, 'speed_lock_after_seconds', None),
    )
