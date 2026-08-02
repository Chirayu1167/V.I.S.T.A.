"""
VISTA — Accident Detection Pipeline
Central configuration: COCO class ids, heuristic thresholds, alert routing.

Tune these numbers against your actual camera height/angle and frame rate —
pixel-based velocity thresholds are resolution- and camera-distance-dependent.
For the hackathon demo, defaults are tuned assuming a ~1280x720, ~15-30fps
CCTV-style overhead/angled feed.
"""

import json
import os
import threading
import time
import warnings
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# COCO class ids (from YOLOv8n COCO-pretrained weights)
# ---------------------------------------------------------------------------
COCO_PERSON = 0
COCO_BICYCLE = 1
COCO_CAR = 2
COCO_MOTORCYCLE = 3
COCO_BUS = 5
COCO_TRUCK = 7

VEHICLE_CLASSES = {COCO_CAR, COCO_MOTORCYCLE, COCO_BUS, COCO_TRUCK}
PERSON_CLASSES = {COCO_PERSON, COCO_BICYCLE}
ALL_TRACKED_CLASSES = VEHICLE_CLASSES | PERSON_CLASSES


@dataclass
class HeuristicConfig:
    # --- track history buffer ---
    history_seconds: float = 3.0          # how much per-track history to retain

    # --- Signal 1: speed drop ---
    speed_drop_window_s: float = 0.5      # look-back window for velocity comparison
    speed_drop_ratio: float = 0.8         # >80% velocity drop within window
    # m/s equivalents of the tuned px/s thresholds (old: 40 px/s @ 0.05 m/px).
    speed_drop_min_prior_speed: float = 2.0  # m/s — ignore tracks that were already ~stationary

    # --- Signal 2: collision (two tracks overlap + impact signature) ---
    # "Impact signature" = at least one vehicle was moving meaningfully before
    # the impact and its speed collapsed at the moment of overlap (either
    # stopped outright or lost a large share of its speed). This replaces the
    # old "both tracks near-zero velocity" rule, which only fired AFTER both
    # vehicles had already stopped (and fired on parked cars touching).
    # Values = old px/s thresholds × 0.05 m/px (old: 15 / 70 px/s).
    collision_iou_threshold: float = 0.45
    collision_max_velocity: float = 0.75  # m/s (~15 px/s), "stopped at overlap" cap
    collision_min_prior_speed: float = 3.5  # m/s (~70 px/s), must have been moving this fast before overlap
    collision_decel_ratio: float = 0.65  # must lose >65% of pre-overlap speed
    # "prior" speed is measured over this pre-impact window, deliberately
    # ending BEFORE the impact moment — a window that includes the impact
    # frames drags the average down with the deceleration and hides the
    # signature. Window: [t - collision_prior_lookback_s, t - collision_prior_end_s].
    collision_prior_lookback_s: float = 1.2
    collision_prior_end_s: float = 0.2

    # --- Signal 3: anomaly stop (stopped mid-road, not at a known stop zone) ---
    anomaly_stop_duration_s: float = 2.0
    anomaly_stop_max_velocity: float = 0.5  # m/s (~10 px/s)
    # A normal stop (red light, turning, congestion) is usually preceded by
    # *gradual* braking; require the vehicle was actually moving meaningfully
    # before the stop so parked cars and creeping turn-stops don't alert.
    anomaly_stop_min_prior_speed: float = 1.5  # m/s — must have been moving this fast before stopping
    anomaly_stop_prior_window_s: float = 1.5   # look-back window for the "before" speed
    # Traffic-jam suppression: if this many stationary vehicles are within
    # traffic_jam_max_gap_m of each other (world space, needs the ML speed
    # estimator), it's a queue at a light/jam, not an incident — no alert.
    # Skipped (no suppression) when world positions are unavailable.
    traffic_jam_min_vehicles: int = 3
    traffic_jam_max_gap_m: float = 2.5
    # Pixel-space fallback jam suppression when the ML estimator (world
    # coordinates) is unavailable: 3+ stationary vehicles within this pixel
    # radius are treated as a queue too.
    traffic_jam_max_gap_px: float = 90.0

    # --- Signal 5: jerk / impact shock (crash into wall, median, object) ---
    # A vehicle hitting a fixed obstacle produces a violent deceleration spike
    # (jerk) that a collision (needs 2 tracks) or a slow windowed speed_drop
    # cannot see. Thresholds in m/s² — normal braking ~2-4 m/s², impacts >8.
    jerk_min_decel: float = 8.0         # m/s² — peak deceleration in the window
    jerk_min_prior_speed: float = 2.0   # m/s — must have been moving meaningfully
    jerk_window_s: float = 0.3          # window over which the spike must occur

    # --- Signal 6: smoke/dust cloud (crash aftermath signature) ---
    # A hard impact (wall, tree, another car) throws up a dust/smoke cloud that
    # grows over a second or two. Purely CV (no extra model): smoke is a large,
    # grayish (low saturation), low-texture, growing blob.
    smoke_min_area_px: float = 900.0       # smallest blob worth considering
    smoke_max_area_px: float = 40000.0     # ignore blobs that cover the whole frame (fog)
    smoke_max_saturation: int = 50         # HSV S — smoke/dust is gray, not colored
    smoke_min_value: int = 110             # HSV V — smoke is mid-bright
    smoke_max_value: int = 245
    smoke_max_local_std: int = 22          # low texture (blurred haze), not a busy object
    smoke_growth_ratio: float = 1.25       # blob must grow >25% vs first-seen area
    smoke_persist_frames: int = 5          # blob must persist this many frames
    # Off by default — restored to yesterday's tuning. Smoke/dust as its own
    # alert kind was splitting one crash into extra cards and masking the real
    # collision classification (A/B: clip_03 collision -> jerk+smoke, no
    # collision). Can be re-enabled with --heuristics smoke_detector_enabled=true.
    smoke_detector_enabled: bool = False

    # --- Signal 4: hit-and-run (vehicle-pedestrian intersection + ped velocity crash) ---
    hitrun_iou_threshold: float = 0.15
    hitrun_ped_velocity_drop: float = 0.9   # >90% drop
    hitrun_vehicle_continues_min_speed: float = 1.0  # m/s (~20 px/s), vehicle keeps moving after

    # --- Verification (per-branch, across consecutive frames) ---
    verify_window_frames: int = 5           # default: must trigger in this many consecutive checks
    verify_window_frames_by_kind: Dict[str, int] = field(default_factory=lambda: {
        # Fast impacts overlap for only a few frames — confirm quicker.
        "collision": 3,
        "jerk": 3,          # impact spikes are 3-5 frames long
        "smoke": 4,         # dust cloud must persist a moment before alerting
    })
    verify_cooldown_s: float = 15.0         # suppress repeated alerts for same track/pair
    verify_dedup_radius_px: float = 90.0    # suppress re-confirmations of the same physical
                                            # incident even if tracker re-assigned track IDs

    # --- Incident fusion (one alert per crash) ---
    # Sub-events of the SAME physical crash (collision -> smoke -> speed_drop
    # -> anomaly_stop) can arrive seconds apart. Fusion window must cover the
    # whole cascade, otherwise one crash is split into several incidents and
    # each produces its own clip + screenshots. Alerts at the same spot within
    # this window are merged into a single incident (see fusion.py).
    fusion_window_s: float = 6.0
    fusion_radius_px: float = 120.0         # how close two spots must be to fuse


@dataclass
class CameraConfig:
    camera_id: str = "CAM-01"
    location_name: str = "MG Road & 2nd Cross Junction"
    lat: float = 22.7196
    lon: float = 75.8577
    # Optional: polygon of known "stop zones" (intersections, bus stops) in pixel
    # coords, where a stationary vehicle should NOT trigger anomaly_stop.
    stop_zones: list = field(default_factory=list)
    
    # --- Camera calibration for real-world speed estimation ---
    # REQUIRED for accurate speeds on a fixed/angled camera: run
    # tools/calibrate_camera.py against a frame from your actual camera to
    # generate homography_src_points/homography_dst_points below. Without
    # these, speed falls back to a flat meter_per_pixel scale, which is
    # only correct for a perfectly top-down camera and will be
    # significantly wrong for anything shot at an angle.
    #
    # meter_per_pixel: flat scale factor, fallback ONLY if no homography is set
    meter_per_pixel: float = 0.05
    # camera_height_m / camera_pitch_deg: informational, not currently used
    # in the homography math (kept for future 3D-geometry-based calibration).
    camera_height_m: float = 8.0
    camera_pitch_deg: float = 45.0
    # homography_src_points: 4+ points in image pixel coords, e.g. corners of
    # a lane, crosswalk, or any rectangle with KNOWN real-world dimensions.
    # Format: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    homography_src_points: list = field(default_factory=list)
    # homography_dst_points: the SAME points' real-world coordinates in
    # meters (e.g. measured with a tape measure / known lane width).
    # Format: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    homography_dst_points: list = field(default_factory=list)

    # --- Speed estimator settings ---
    # fps: video frame rate (used for time-window sizing)
    fps: float = 25.0
    # speed_history_seconds: how much position history to keep per track;
    # speed is fit (least-squares) over this whole window each update
    speed_history_seconds: float = 2.0
    # speed_min_history: minimum samples before a speed can be computed
    speed_min_history: int = 5
    # speed_max_kmph: maximum plausible speed, used to clamp/filter outliers
    speed_max_kmph: float = 150.0
    # speed_lock_after_seconds: if set, freeze each track's speed once it has
    # this many seconds of history (stable "how fast was it going" number).
    # Leave None to keep re-estimating every frame (recommended default —
    # heuristics like sudden-deceleration detection need a live value).
    speed_lock_after_seconds: Optional[float] = None


@dataclass
class DispatchConfig:
    # Mock endpoints for the hackathon demo — swap for real webhook/SMS/API URLs.
    traffic_police_webhook: str = "mock://traffic-police/alert"
    hospital_ems_webhook: str = "mock://hospital-ems/alert"
    police_control_room_webhook: str = "mock://police-control-room/alert"
    dashboard_log_path: str = "alerts.jsonl"

    # --- Global rate limiting (per camera) ---
    # If more than rate_limit_max_alerts dispatch within rate_limit_window_s,
    # further alerts in that window are bundled: channels collapse to
    # traffic_police only instead of re-paging EMS/police-control per event.
    # Set rate_limit_max_alerts=0 to disable.
    rate_limit_window_s: float = 10.0
    rate_limit_max_alerts: int = 4

    # --- Outbound payload signing (real endpoints only — mock dispatch just
    # logs the signature) ---
    # Real webhook/API endpoints should verify this before trusting a
    # payload. Set to a real secret (env var, secrets manager, etc.) before
    # swapping _send_mock for a live requests.post(). Do NOT commit a real
    # secret here.
    hmac_secret: Optional[str] = None


class ConfigWatcher:
    """Polls a JSON file for changes and mutates an existing HeuristicConfig
    / CameraConfig IN PLACE (same object references AccidentPipeline already
    holds), so tuning thresholds or redrawing stop zones during a live demo
    takes effect on the next frame without restarting the process.

    Expected JSON shape (all keys optional):
        {
          "heuristics": {"speed_drop_ratio": 0.75, "collision_iou_threshold": 0.4},
          "stop_zones": [[[x1,y1],[x2,y2],[x3,y3]], ...]
        }

    Unknown keys are ignored with a warning rather than raising, so a typo
    in the file doesn't crash a running demo.
    """

    def __init__(self, path: str, heuristic_cfg: Optional["HeuristicConfig"] = None,
                 camera_cfg: Optional["CameraConfig"] = None, interval_s: float = 2.0):
        self.path = path
        self.heuristic_cfg = heuristic_cfg
        self.camera_cfg = camera_cfg
        self.interval_s = interval_s
        self._last_mtime = None
        self._stop_flag = False
        self._thread: Optional[threading.Thread] = None

    def reload_once(self) -> bool:
        """Reload immediately if the file changed since the last check.
        Returns True if a reload happened. Safe to call directly (e.g. from
        a single-threaded CLI loop) instead of using start()/stop()."""
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            return False
        if mtime == self._last_mtime:
            return False
        self._last_mtime = mtime
        try:
            with open(self.path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            warnings.warn(f"ConfigWatcher: failed to read/parse {self.path}: {e}")
            return False

        if self.heuristic_cfg is not None:
            self._apply(self.heuristic_cfg, data.get("heuristics", {}))
        if self.camera_cfg is not None and "stop_zones" in data:
            self.camera_cfg.stop_zones = data["stop_zones"]
        return True

    @staticmethod
    def _apply(cfg_obj, updates: dict):
        valid = {f.name for f in fields(cfg_obj)}
        for key, value in updates.items():
            if key not in valid:
                warnings.warn(f"ConfigWatcher: unknown field '{key}', ignoring")
                continue
            setattr(cfg_obj, key, value)

    def start(self):
        """Run reload_once() on a background daemon thread every interval_s."""
        if self._thread is not None:
            return
        self._stop_flag = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop_flag:
            self.reload_once()
            time.sleep(self.interval_s)

    def stop(self):
        self._stop_flag = True
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 1)
            self._thread = None
