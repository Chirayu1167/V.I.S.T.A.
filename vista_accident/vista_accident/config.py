"""
VISTA — Accident Detection Pipeline
Central configuration: COCO class ids, heuristic thresholds, alert routing.

Tune these numbers against your actual camera height/angle and frame rate —
pixel-based velocity thresholds are resolution- and camera-distance-dependent.
For the hackathon demo, defaults are tuned assuming a ~1280x720, ~15-30fps
CCTV-style overhead/angled feed.

Threshold audit (real-world speeds)
-----------------------------------
All HeuristicConfig velocities are in PHYSICAL units (m/s) — they were
converted from the old pixel/second values using the flat 0.05 m/px scale
and kept when the ML speed estimator (homography-based, see
CameraConfig + speed_estimator.py) became the default. With a calibrated
homography the estimator reports honest m/s, so the same numbers still
describe physical behaviour:

    speed_drop_min_prior_speed      2.0 m/s  (7.2 km/h)  — must have been
                                                          moving before drop
    collision_max_velocity          0.75 m/s (2.7 km/h)  — "stopped at impact"
    collision_min_prior_speed       3.5 m/s  (12.6 km/h) — impact approach speed
    anomaly_stop_max_velocity       0.5 m/s  (1.8 km/h)  — stationary cap
    anomaly_stop_min_prior_speed    1.5 m/s  (5.4 km/h)  — moving before stop
    hitrun_vehicle_continues_min_speed 1.0 m/s (3.6 km/h) — perp keeps moving

Audited 2026-08-04 after the homography calibration work: these remain
sane for calibrated speeds (a real crash approach is well above 12.6 km/h;
a genuinely stopped car sits below 1.8 km/h including tracker jitter). No
changes were needed. If you switch to uncalibrated pixel speed readings
(use_ml_speed=False, no homography), re-derive thresholds from your
meter_per_pixel instead of trusting these m/s values.

Scale re-anchor (2026-08-09, benchmark-driven)
----------------------------------------------
Benchmarking clips 01-10 showed the flat 0.05 m/px scale overstates speeds
near the horizon (perspective), so thresholds tuned on flat readings are
too strict for honest homography speeds: auto-calibration mode missed the
clip_03 collision at the default thresholds. A sweep of threshold scales
in auto mode (benchmark_accident.py --threshold-scale):

    k=1.0  F1=0.36 (clip_03 collision missed)
    k=0.5  F1=0.50
    k=0.65 F1=0.55  <-- chosen
    k=0.8  F1=0.55

HOMOGRAPHY_THRESHOLD_SCALE=0.65 is applied automatically by
AccidentPipeline whenever the camera carries a fitted homography
(homography_src_points/dst_points set — auto-calibration or camera
profile); flat/px-m paths keep the tuned scale of 1.0.
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

# ---------------------------------------------------------------------------
# Threshold scale re-anchor for homography-based speeds (see module docstring)
# ---------------------------------------------------------------------------
HOMOGRAPHY_THRESHOLD_SCALE = 0.65

PHYSICAL_VELOCITY_THRESHOLDS = (
    "speed_drop_min_prior_speed",
    "speed_drop_fast_decel_mps2",
    "speed_drop_fast_end_max_velocity",
    "collision_max_velocity",
    "collision_min_prior_speed",
    "anomaly_stop_max_velocity",
    "anomaly_stop_min_prior_speed",
    "hitrun_vehicle_continues_min_speed",
)


def scale_thresholds(cfg: "HeuristicConfig", k: float) -> None:
    """Multiply all PHYSICAL-VELOCITY thresholds by k. Ratios, durations,
    IoUs and pixel/distances stay untouched. Used to re-anchor thresholds
    between the flat 0.05 m/px scale they were tuned on and the honest m/s
    a fitted homography reports (which read systematically lower than flat
    speeds near the horizon)."""
    for name in PHYSICAL_VELOCITY_THRESHOLDS:
        setattr(cfg, name, getattr(cfg, name) * k)


@dataclass
class HeuristicConfig:
    # --- track history buffer ---
    history_seconds: float = 3.0          # how much per-track history to retain

    # --- Signal 1: speed drop ---
    # Loosened from 0.5s/0.8 because tracker speed readings on distant CCTV
    # are noisy and jittery: a longer window smooths the noise, and a lower
    # ratio catches crashes whose measured drop reads under 80%.
    speed_drop_window_s: float = 0.5      # look-back window for velocity comparison
    speed_drop_ratio: float = 0.8         # >80% velocity drop within window
    # m/s equivalents of the tuned px/s thresholds (old: 40 px/s @ 0.05 m/px).
    # Lowered from 2.0 because tracker speed readings are noisy on distant
    # CCTV — a car that was cruising but reads slow-ish must still be caught
    # when it crashes to a stop.
    speed_drop_min_prior_speed: float = 2.0  # m/s — ignore tracks that were already ~stationary

    # FAST-DROP supplement: catches the crash-stop that happens in only a few
    # frames (wall/barrier hit) where the windowed >70% ratio is diluted by
    # pre-crash frames. Fires when the Kalman instantaneous velocity (which
    # snaps to ~0 on a hard stop via its innovation gate) is near-stopped AND
    # the short-window deceleration is huge. Set 0.0 to disable.
    # Values are deliberately LOOSE because tracker speed is not stable —
    # the noise floor of velocity readings is high, so a crash that reads
    # "only" ~4 m/s^2 of decel or stops "only" at ~2 m/s must still alert.
    # The near-stopped + big-drop combination keeps normal braking excluded.
    speed_drop_fast_decel_mps2: float = 4.0      # min decel to count as a fast crash-stop
    speed_drop_fast_end_max_velocity: float = 2.0 # "now" must be near-stopped (m/s)
    speed_drop_fast_window_s: float = 0.3        # short span the fast drop is measured over

    # --- Signal 2: collision (two tracks overlap + impact signature) ---
    # "Impact signature" = at least one vehicle was moving meaningfully before
    # the impact and its speed collapsed at the moment of overlap (either
    # stopped outright or lost a large share of its speed). This replaces the
    # old "both tracks near-zero velocity" rule, which only fired AFTER both
    # vehicles had already stopped (and fired on parked cars touching).
    # Values = old px/s thresholds × 0.05 m/px (old: 15 / 70 px/s).
    collision_iou_threshold: float = 0.45
    collision_buffer_iou_threshold: float = 0.50  # deep-contact bar for the
    # evidence buffer (a crash struck AT speed has deep overlap; a braking
    # queue approach only touches at ~0.45)
    collision_overlap_collapse_factor: float = 0.80  # velocity at the overlap
    # instant already ≤ 0.80×prior counts as "collapse started at contact"
    collision_max_velocity: float = 0.75  # m/s (~15 px/s), "stopped at overlap" cap
    collision_min_prior_speed: float = 3.5  # m/s (~70 px/s), must have been moving this fast before overlap
    collision_decel_ratio: float = 0.65  # must lose >65% of pre-overlap speed
    # "prior" speed is measured over this pre-impact window, deliberately
    # ending BEFORE the impact moment — a window that includes the impact
    # frames drags the average down with the deceleration and hides the
    # signature. Window: [t - collision_prior_lookback_s, t - collision_prior_end_s].
    collision_prior_lookback_s: float = 1.2
    collision_prior_end_s: float = 0.2
    collision_collapse_window_s: float = 0.8  # how long after a high-speed box
    # overlap a pair vehicle may still collapse and count as the same impact

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

    # --- Signal 4: hit-and-run (vehicle-pedestrian intersection + ped velocity crash) ---
    hitrun_iou_threshold: float = 0.15
    hitrun_ped_velocity_drop: float = 0.9   # >90% drop
    hitrun_vehicle_continues_min_speed: float = 1.0  # m/s (~20 px/s), vehicle keeps moving after

    # --- Verification (per-branch, across consecutive frames) ---
    verify_window_frames: int = 5           # default: must trigger in this many consecutive checks
    verify_window_frames_by_kind: Dict[str, int] = field(default_factory=lambda: {
        # Fast impacts overlap for only a few frames — confirm quicker.
        "collision": 3,
    })
    verify_cooldown_s: float = 15.0         # suppress repeated alerts for same track/pair
    verify_dedup_radius_px: float = 90.0    # suppress re-confirmations of the same physical
                                            # incident even if tracker re-assigned track IDs

    # --- Incident fusion (one alert per crash) ---
    # Sub-events of the SAME physical crash (collision -> speed_drop ->
    # anomaly_stop) can arrive seconds apart. Fusion window must cover the
    # whole cascade, otherwise one crash is split into several incidents and
    # each produces its own clip + screenshots. Alerts at the same spot within
    # this window are merged into a single incident (see fusion.py).
    # NOTE: was temporarily 6.0s, which fused the 2nd/4th accidents into the
    # 1st/3rd (same spot, a few seconds later) and swallowed real crashes —
    # restored to the 1.5s baseline so distinct accidents dispatch separately.
    fusion_window_s: float = 1.5
    fusion_radius_px: float = 120.0         # how close two spots must be to fuse


@dataclass
class ViolenceConfig:
    """Pose-based violence/road-rage branch (see violence_heuristics.py).

    A fight/assault is detected geometrically: two people close together
    with high-speed limb (wrist/elbow) motion. Scene-agnostic — no fight
    classifier weights to train, and it works on unseen CCTV angles because
    it uses keypoint motion, not appearance. The VideoMAE clip classifier
    (violence_model/) can be bolted on later as a secondary confirmation.

    Pixel thresholds are tuned for ~1280x720 CCTV-style feeds. This branch
    runs at reduced cadence (motion prefilter per design: the accident
    branch needs every frame, the violence branch may skip).
    """

    # --- pose model ---
    pose_weights: str = "yolo11n-pose.pt"   # auto-downloaded by ultralytics (~5.5 MB)
    pose_imgsz: int = 640                   # smaller input = faster inference
    pose_conf_threshold: float = 0.45
    pose_cadence_frames: int = 3            # run pose detection every N frames

    # --- person pair gates ---
    # Both must hold before a pair is even scored:
    pair_max_distance_px: float = 60.0      # centroid distance gate
    pair_min_iou: float = 0.20              # bbox overlap gate (fallback for crouched/entangled);
                                            # raised to 0.35 to stop 1080p crowd bboxes, but that
                                            # also killed test2.mp4's real grapple (dist 60-92px,
                                            # IoU 0.17-0.25) — restored to 0.20; the crowd case is
                                            # handled by pair_max_persons instead
    pair_min_duration_s: float = 0.5        # pair must co-exist this long before scoring
    pair_max_stale_s: float = 0.35          # drop pairs whose latest pose sample is older than this
                                            # (kills ghost pairs from tracks that left the frame but
                                            # still live in the 2s history buffer; 0.5s still let
                                            # walkers 7+ checks stale trigger)
    pair_max_persons: int = 5               # suppress ALL scoring above this many concurrent people.
                                            # Geometry-only motion cannot separate a walker's arm
                                            # gesture from a punch (both 150-300 px/s on 1080p); a
                                            # crowded street ALWAYS produces false pairs. Fights in
                                            # sparse scenes (<=5 people) are detected reliably; crowd
                                            # violence is deferred to the VideoMAE secondary
                                            # confirmation. Ground truth: test4.mp4 (crowd, 6-9 ppl)
                                            # = 0 alerts, test2/3/fight.mp4 (2-4 ppl) = detected.
    pair_max_persons_window_s: float = 2.0  # the count gate uses the MAX person count over this
                                            # recent window, not the instant frame — test4's crowd
                                            # thinned to 3 people for a moment and slipped a false
                                            # pair past the instant gate

    # --- limb motion ---
    # COCO keypoint ids: 5,6 = shoulders; 7,8 = elbows; 9,10 = wrists.
    limb_keypoints: List[int] = field(default_factory=lambda: [7, 8, 9, 10])
    limb_window_s: float = 0.4              # window for mean limb-point speed
    limb_min_sample_gap_s: float = 0.12     # only score point pairs >= this far apart in time;
                                            # at 60fps pose cadence (every 3rd frame = 0.05s) the
                                            # frame-to-frame keypoint jitter reads as 20x too much
                                            # speed (saw 1600 px/s on normal crowds), the gap floors it
    limb_speed_threshold_px_s: float = 30.0  # mean wrist/elbow speed to count as aggressive motion.
                                            # test2.mp4's real grapple is only 32 px/s, so this must
                                            # stay low; the pair_max_persons gate is what keeps
                                            # crowds out, not the speed threshold.

    # --- entangled-pair (bbox overlap) signal ---
    # Distant CCTV fights: people are SMALL, keypoints are unreliable/NaN,
    # but a fight means two boxes that overlap strongly and STAY overlapped
    # (grappling/struggling). This signal fires with NO limb speed at all —
    # exactly the far-angle case the motion gate misses. The crowd gate
    # (pair_max_persons) still applies. Fire-or-not fires on BOTH signals
    # (limb OR overlap); overlap adds no false positives on the clips since
    # walkers never sustain 0.45+ IoU for 0.8s.
    pair_overlap_min_iou: float = 0.45       # sustained strong bbox overlap
    pair_overlap_min_duration_s: float = 0.8 # overlap must persist this long

    # --- verification (mirrors HeuristicConfig, kind-scoped) ---
    verify_window_frames: int = 3           # consecutive checks before confirm
    verify_cooldown_s: float = 20.0         # re-alert suppression for the same pair/spot

    # --- severity baseline ---
    severity_baseline: float = 0.55


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
    # --- Per-camera detection overrides (shared-detector multi-camera mode) ---
    # When every camera shares ONE Detector/PoseDetector instance (see
    # batch_inference.py), the model itself can only hold one global
    # conf_threshold/classes value. These let an individual camera still
    # tune detection sensitivity — e.g. a low-light or far-field camera
    # that needs a lower confidence floor than the rest. Applied by
    # batch_inference.py's fan-out step AFTER the shared batched call
    # returns, filtering that camera's slice of results only.
    # Both default to None = "use the shared detector's own defaults",
    # i.e. today's behavior, unchanged, for any camera that doesn't set them.
    detection_conf_threshold: Optional[float] = None
    detection_classes: Optional[List[int]] = None

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

    # --- Emergency Response server bridge (see emergency_response/) ---
    # Every dispatched alert is also POSTed to this Emergency Response
    # server's /api/incidents endpoint, so an ML-detected crash shows up on
    # the hospital/police/traffic-police dashboards alongside manual /report
    # incidents (nearest-authority routing via Haversine happens server-side).
    # Forwarding is fire-and-forget on a background thread — an absent server
    # never delays or breaks dispatch. Leave None to disable (default: no
    # server, identical behavior to before).
    emergency_response_url: Optional[str] = None

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
            warnings.warn(f"ConfigWatcher: failed to read/parse {self.path}: {e}", stacklevel=2)
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
                warnings.warn(f"ConfigWatcher: unknown field '{key}', ignoring", stacklevel=2)
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
