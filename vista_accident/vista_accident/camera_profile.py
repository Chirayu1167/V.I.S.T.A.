"""
Camera calibration profiles: persist a camera's homography + calibration
metadata as a single JSON file, and load it back into a CameraConfig.

A profile captures everything needed to run the pipeline with REAL speeds
on a specific fixed camera:

    - homography_src_points / homography_dst_points (from
      tools/calibrate_camera.py or the GUI calibration flow)
    - meter_per_pixel (fallback only when no homography is set)
    - stop_zones, camera id/location/lat/lon
    - fps and a free-text calibration_note documenting what real-world
      features / assumptions the homography was built from (e.g. "lane
      width 3.5 m assumed per IRC" or "measured 7.0 m road width").

CLI:  python demo.py --source video.mp4 --camera-profile camera_profiles/CAM-01.json
GUI:  Camera Profile -> Load ... in gui_app.py
"""

import json
import os

from .config import CameraConfig

# Fields a profile JSON may contain, mapped to CameraConfig attributes.
# Only these are persisted/serialized — unknown keys are preserved on
# round-trip but not applied to CameraConfig.
_PROFILE_FIELDS = [
    "camera_id",
    "location_name",
    "lat",
    "lon",
    "stop_zones",
    "meter_per_pixel",
    "camera_height_m",
    "camera_pitch_deg",
    "homography_src_points",
    "homography_dst_points",
    "fps",
    "speed_history_seconds",
    "speed_min_history",
    "speed_max_kmph",
    "speed_lock_after_seconds",
    "calibration_note",
]


def load_profile(path: str) -> CameraConfig:
    """Load a camera profile JSON file into a CameraConfig.

    Unknown keys are ignored with a warning (mirrors ConfigWatcher's
    leniency) so a profile written by a newer version still loads.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Camera profile not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Camera profile {path} must be a JSON object")

    # calibration_note is free-text metadata, not a CameraConfig field —
    # attach it as a dynamic attribute so it survives a save/load round-trip.
    valid_fields = {f.name for f in CameraConfig.__dataclass_fields__.values()}
    kwargs = {key: data[key] for key in data if key in valid_fields}
    cfg = CameraConfig(**kwargs)
    cfg.calibration_note = data.get("calibration_note")
    return cfg


def save_profile(path: str, camera_cfg: CameraConfig, calibration_note: str = None) -> None:
    """Write a CameraConfig (plus optional calibration_note) to a JSON file.

    The file is written atomically-ish (write to a temp file then rename)
    so a crash mid-write can't corrupt an existing profile.
    """
    data = {}
    for key in _PROFILE_FIELDS:
        if key == "calibration_note":
            value = calibration_note or getattr(camera_cfg, "calibration_note", None)
        else:
            value = getattr(camera_cfg, key, None)
        if value is not None:
            data[key] = value

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def world_distance_m(camera_cfg: CameraConfig, p1, p2) -> float:
    """Real-world distance (meters) between two pixel points, using the
    profile's homography when present, else the flat meter_per_pixel scale.

    Used by the speed-validation tool to turn two on-road markers into a
    known-distance pair for crossing-time ground truth.
    """
    import cv2
    import numpy as np

    from .speed_estimator import CameraCalibration

    calib = CameraCalibration(
        meter_per_pixel=camera_cfg.meter_per_pixel,
        src_points=getattr(camera_cfg, "homography_src_points", None) or None,
        dst_points=getattr(camera_cfg, "homography_dst_points", None) or None,
    )
    if calib.homography is not None:
        pts = np.array([p1, p2], dtype=np.float32).reshape(1, -1, 2)
        world = cv2.perspectiveTransform(pts, calib.homography)[0]
        dx, dy = world[1] - world[0]
        return float(np.hypot(dx, dy))
    return float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]) * calib.meter_per_pixel)


def find_profiles(directory: str = None) -> list:
    """List available camera profile JSON files (sorted by name)."""
    if directory is None:
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "camera_profiles")
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".json")
    )
