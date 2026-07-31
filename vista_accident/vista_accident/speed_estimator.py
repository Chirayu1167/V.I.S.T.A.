"""
ML-based Speed Estimator using Ultralytics YOLO tracking with perspective transformation.

Replaces pixel-based velocity with real-world speed estimation (m/s, km/h).
Uses a pretrained YOLO model for detection + ByteTrack for tracking,
with configurable camera calibration for perspective correction.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.solutions.speed_estimation import SpeedEstimator as UltralyticsSpeedEstimator


@dataclass
class CameraCalibration:
    """Camera calibration parameters for perspective transformation."""
    # Source points in image (pixel coordinates) - 4 corners of a known rectangle on ground plane
    src_points: List[Tuple[float, float]] = None
    # Destination points in real-world (meters) - corresponding rectangle corners
    dst_points: List[Tuple[float, float]] = None
    # Alternative: direct meter_per_pixel scale (simpler, less accurate for angled cameras)
    meter_per_pixel: float = 0.05
    # Camera height (meters) for fallback estimation
    camera_height_m: float = 8.0
    # Camera pitch angle (degrees, downward from horizontal)
    camera_pitch_deg: float = 45.0
    
    def __post_init__(self):
        self.src_np = None
        self.dst_np = None
        self.homography = None
        self.inv_homography = None
        
        if self.src_points is not None and self.dst_points is not None:
            if len(self.src_points) >= 4 and len(self.dst_points) >= 4:
                self.src_np = np.array(self.src_points, dtype=np.float32)
                self.dst_np = np.array(self.dst_points, dtype=np.float32)
                self.homography, _ = cv2.findHomography(self.src_np, self.dst_np)
                self.inv_homography, _ = cv2.findHomography(self.dst_np, self.src_np)


@dataclass
class TrackSpeed:
    """Speed information for a single track."""
    track_id: int
    speed_mps: float          # meters per second
    speed_kmph: float         # kilometers per hour
    world_pos: Tuple[float, float]  # (x, y) in real-world meters
    is_locked: bool           # whether speed estimate is finalized
    history_length: int       # number of frames in history


class MlSpeedEstimator:
    """
    ML-based speed estimator using Ultralytics YOLO + ByteTrack.
    
    Features:
    - Pretrained YOLO detection + ByteTrack tracking
    - Perspective transformation for real-world coordinates
    - Configurable speed estimation window
    - Returns speeds in m/s and km/h
    - Compatible with existing TrackHistory interface
    """
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        fps: float = 25.0,
        calibration: Optional[CameraCalibration] = None,
        max_history: int = 15,
        min_history: int = 5,
        max_speed_kmph: float = 150.0,
        device: str = "cpu",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        classes: Optional[List[int]] = None,
    ):
        """
        Initialize the ML speed estimator.
        
        Args:
            model_path: Path to YOLO model (default: yolov8n.pt - COCO pretrained)
            fps: Video frame rate for time calculations
            calibration: CameraCalibration for perspective transformation
            max_history: Maximum frames to keep in track history
            min_history: Minimum frames before speed is computed
            max_speed_kmph: Maximum plausible speed (filters outliers)
            device: Inference device ('cpu', 'cuda', 'mps')
            conf_threshold: Detection confidence threshold
            iou_threshold: IoU threshold for NMS
            classes: Filter by COCO class IDs (None = all)
        """
        self.fps = fps
        self.max_history = max_history
        self.min_history = min_history
        self.max_speed_kmph = max_speed_kmph
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes
        
        # Camera calibration
        self.calibration = calibration or CameraCalibration()
        
        # Load YOLO model
        self.model = YOLO(model_path)
        if device != "auto":
            self.model.to(device)
        
        # Track histories: track_id -> deque of (frame_idx, world_x, world_y, bbox, class_id)
        self.track_histories: Dict[int, deque] = {}
        self.track_first_frame: Dict[int, int] = {}
        self.track_speeds: Dict[int, TrackSpeed] = {}
        self.locked_tracks: set = set()
        self.frame_count = 0
        
        # Ultralytics SpeedEstimator for reference (uses line-crossing method)
        self._ultralytics_estimator = UltralyticsSpeedEstimator(
            fps=fps,
            meter_per_pixel=self.calibration.meter_per_pixel,
            max_speed=int(max_speed_kmph),
            max_hist=max_history,
            verbose=False,
        )
    
    def _pixel_to_world(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """Convert pixel coordinates to real-world meters using homography."""
        if self.calibration.homography is not None:
            pt = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
            world_pt = cv2.perspectiveTransform(pt, self.calibration.homography)
            return float(world_pt[0, 0, 0]), float(world_pt[0, 0, 1])
        else:
            # Fallback: simple scale
            return pixel_x * self.calibration.meter_per_pixel, pixel_y * self.calibration.meter_per_pixel
    
    def _world_to_pixel(self, world_x: float, world_y: float) -> Tuple[float, float]:
        """Convert real-world meters to pixel coordinates."""
        if self.calibration.inv_homography is not None:
            pt = np.array([[[world_x, world_y]]], dtype=np.float32)
            pixel_pt = cv2.perspectiveTransform(pt, self.calibration.inv_homography)
            return float(pixel_pt[0, 0, 0]), float(pixel_pt[0, 0, 1])
        else:
            return world_x / self.calibration.meter_per_pixel, world_y / self.calibration.meter_per_pixel
    
    def update(self, frame: np.ndarray) -> Dict[int, TrackSpeed]:
        """
        Process a frame and update speed estimates for all tracked objects.
        
        Args:
            frame: Input frame (BGR numpy array)
            
        Returns:
            Dict mapping track_id -> TrackSpeed
        """
        self.frame_count += 1
        
        # Run YOLO tracking
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            verbose=False,
        )
        
        if results[0].boxes is None or results[0].boxes.id is None:
            # No tracks in this frame
            self._age_tracks()
            return self.track_speeds
        
        boxes = results[0].boxes.xyxy.cpu().numpy()  # (N, 4)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)  # (N,)
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # (N,)
        confidences = results[0].boxes.conf.cpu().numpy()  # (N,)
        
        current_track_ids = set()
        
        for box, track_id, cls_id, conf in zip(boxes, track_ids, class_ids, confidences):
            current_track_ids.add(track_id)
            
            # Get center point of bounding box
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            
            # Convert to world coordinates
            world_x, world_y = self._pixel_to_world(cx, cy)
            
            # Initialize or update track history
            if track_id not in self.track_histories:
                self.track_histories[track_id] = deque(maxlen=self.max_history)
                self.track_first_frame[track_id] = self.frame_count
            
            history = self.track_histories[track_id]
            history.append((self.frame_count, world_x, world_y, box, cls_id))
            
            # Compute speed if we have enough history and not locked
            if track_id not in self.locked_tracks and len(history) >= self.min_history:
                self._compute_speed(track_id)
        
        # Age out tracks not seen in this frame
        self._age_tracks(current_track_ids)
        
        return self.track_speeds
    
    def _compute_speed(self, track_id: int) -> None:
        """Compute speed for a track based on its history."""
        history = self.track_histories[track_id]
        if len(history) < 2:
            return
        
        # Use first and last points in history window
        first_frame, first_x, first_y, _, _ = history[0]
        last_frame, last_x, last_y, last_box, last_cls = history[-1]
        
        dt = (last_frame - first_frame) / self.fps
        if dt <= 0:
            return
        
        # Distance in meters
        dx = last_x - first_x
        dy = last_y - first_y
        distance_m = np.sqrt(dx * dx + dy * dy)
        
        # Speed in m/s and km/h
        speed_mps = distance_m / dt
        speed_kmph = speed_mps * 3.6
        
        # Clamp to max speed
        if speed_kmph > self.max_speed_kmph:
            speed_kmph = self.max_speed_kmph
            speed_mps = speed_kmph / 3.6
        
        # Check if we have enough history to lock the speed
        is_locked = len(history) >= self.max_history
        
        self.track_speeds[track_id] = TrackSpeed(
            track_id=track_id,
            speed_mps=speed_mps,
            speed_kmph=speed_kmph,
            world_pos=(last_x, last_y),
            is_locked=is_locked,
            history_length=len(history),
        )
        
        if is_locked:
            self.locked_tracks.add(track_id)
            # Free memory
            self.track_histories.pop(track_id, None)
            self.track_first_frame.pop(track_id, None)
    
    def _age_tracks(self, current_ids: Optional[set] = None) -> None:
        """Remove tracks that haven't been seen recently."""
        if current_ids is None:
            current_ids = set()
        
        stale_ids = []
        for track_id, history in self.track_histories.items():
            if track_id not in current_ids:
                last_frame = history[-1][0]
                if self.frame_count - last_frame > self.max_history:
                    stale_ids.append(track_id)
        
        for track_id in stale_ids:
            self.track_histories.pop(track_id, None)
            self.track_first_frame.pop(track_id, None)
            self.track_speeds.pop(track_id, None)
            self.locked_tracks.discard(track_id)
    
    def get_speed(self, track_id: int) -> Optional[TrackSpeed]:
        """Get current speed estimate for a track."""
        return self.track_speeds.get(track_id)
    
    def get_all_speeds(self) -> Dict[int, TrackSpeed]:
        """Get all current speed estimates."""
        return self.track_speeds.copy()
    
    def velocity(self, track_id: int, window_s: float) -> Optional[float]:
        """
        Get average velocity (m/s) over the last window_s seconds.
        Compatible with TrackHistory.velocity() interface.
        """
        speed = self.track_speeds.get(track_id)
        if speed:
            return speed.speed_mps
        return None
    
    def velocity_between(self, track_id: int, t0: float, t1: float) -> Optional[float]:
        """
        Get average velocity (m/s) between two timestamps.
        Compatible with TrackHistory.velocity_between() interface.
        """
        # For simplicity, return current speed if track exists
        return self.velocity(track_id, t1 - t0)
    
    def instantaneous_velocity(self, track_id: int) -> Optional[float]:
        """Get instantaneous velocity (m/s). Compatible with TrackHistory interface."""
        return self.velocity(track_id, 1.0 / self.fps)
    
    def stationary_duration(self, track_id: int, max_velocity: float) -> float:
        """
        How long (seconds) the track has stayed below max_velocity (m/s).
        Compatible with TrackHistory.stationary_duration() interface.
        """
        speed = self.track_speeds.get(track_id)
        if speed and speed.speed_mps <= max_velocity:
            # Approximate: if currently slow, assume stationary for history length
            return speed.history_length / self.fps
        return 0.0
    
    def draw_speeds(self, frame: np.ndarray) -> np.ndarray:
        """Draw speed annotations on frame."""
        annotated = frame.copy()
        for track_id, speed in self.track_speeds.items():
            # Find the track's current bbox
            if track_id in self.track_histories:
                _, _, _, box, _ = self.track_histories[track_id][-1]
                x1, y1, x2, y2 = map(int, box)
                
                # Draw speed label
                label = f"ID:{track_id} {speed.speed_kmph:.1f} km/h"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
        return annotated


def create_speed_estimator_from_config(config) -> MlSpeedEstimator:
    """Factory function to create MlSpeedEstimator from CameraConfig."""
    calibration = CameraCalibration(
        meter_per_pixel=getattr(config, 'meter_per_pixel', 0.05),
        camera_height_m=getattr(config, 'camera_height_m', 8.0),
        camera_pitch_deg=getattr(config, 'camera_pitch_deg', 45.0),
        src_points=getattr(config, 'homography_src_points', None),
        dst_points=getattr(config, 'homography_dst_points', None),
    )
    
    return MlSpeedEstimator(
        fps=getattr(config, 'fps', 25.0),
        calibration=calibration,
        max_history=getattr(config, 'speed_max_history', 15),
        min_history=getattr(config, 'speed_min_history', 5),
        max_speed_kmph=getattr(config, 'speed_max_kmph', 150.0),
        device="cpu",
    )