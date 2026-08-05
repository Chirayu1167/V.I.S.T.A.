"""
Shared frame-annotation logic: track boxes, per-track speed readout, and the
optional on-video accident-alert panel. Used by both the CLI runner
(demo.py) and the desktop GUI (gui_app.py) so the two stay visually
consistent.
"""

import cv2
import numpy as np

from .config import COCO_BICYCLE, COCO_BUS, COCO_CAR, COCO_MOTORCYCLE, COCO_PERSON, COCO_TRUCK
from .violence_heuristics import SKELETON_PAIRS

COLORS = {
    "vehicle": (60, 180, 255),
    "person": (80, 220, 120),
    "alert": (40, 40, 255),
    "violence": (255, 0, 255),
    "speed_text": (255, 255, 255),
    "panel_bg": (20, 20, 20),
}
VEHICLE_CLASS_IDS = {2, 3, 5, 7}

# Severity -> BGR color, low to critical.
SEVERITY_COLORS = {
    "low": (60, 200, 60),
    "medium": (0, 200, 255),
    "high": (0, 130, 255),
    "critical": (0, 0, 255),
}
SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}

# How long a confirmed alert stays visible in the on-screen panel by default.
ALERT_DISPLAY_SECONDS = 4.0
# Speed look-back window (seconds) — short enough to feel responsive,
# long enough to smooth out per-frame tracker jitter.
SPEED_WINDOW_S = 0.4

# Typical real-world object widths (meters), used to auto-estimate a
# pixels-per-meter scale per object from its own bounding-box width when no
# manual camera calibration is supplied. This is an approximation (assumes a
# roughly front/side-on view, flat road, no strong lens distortion) but is
# far closer to reality than an uncalibrated pixel/second number.
REAL_WORLD_WIDTH_M = {
    COCO_CAR: 1.8,
    COCO_MOTORCYCLE: 0.8,
    COCO_BUS: 2.5,
    COCO_TRUCK: 2.5,
    COCO_PERSON: 0.5,
    COCO_BICYCLE: 0.6,
}
DEFAULT_REAL_WIDTH_M = 1.8

MAX_PLAUSIBLE_KMH = 180.0  # clamp for outlier jitter, not a real physical cap


class SpeedEstimator:
    """
    Converts a track's raw pixel velocity into a smoothed km/h estimate.

    - If `manual_px_per_meter` is supplied (from real camera calibration),
      that fixed scale is used for every object — most accurate option.
    - Otherwise, scale is auto-estimated per object per frame from its own
      bounding-box width vs. its class's typical real-world width, which
      self-corrects for distance from the camera (near objects are wider in
      pixels, matching a wider real box) far better than a single global
      pixel/second value.
    - An exponential moving average per track smooths out detector/tracker
      jitter frame to frame; call reset() when starting a new video.
    """

    def __init__(self, manual_px_per_meter=None, smoothing=0.3):
        self.manual_px_per_meter = manual_px_per_meter
        self.smoothing = smoothing
        self._ema = {}

    def reset(self):
        self._ema.clear()

    def estimate_kmh(self, track_id, cls, bbox, history):
        raw_v = history.velocity(track_id, SPEED_WINDOW_S)
        if raw_v is None:
            raw_v = history.instantaneous_velocity(track_id)
        if raw_v is None:
            return self._ema.get(track_id)

        if getattr(history, "speed_estimator", None) is not None:
            kmh = raw_v * 3.6
        elif self.manual_px_per_meter:
            px_per_m = self.manual_px_per_meter
            kmh = (raw_v / px_per_m) * 3.6
        else:
            x1, _, x2, _ = bbox
            width_px = max(1.0, x2 - x1)
            real_w_m = REAL_WORLD_WIDTH_M.get(cls, DEFAULT_REAL_WIDTH_M)
            px_per_m = width_px / real_w_m
            kmh = (raw_v / px_per_m) * 3.6
        kmh = max(0.0, min(kmh, MAX_PLAUSIBLE_KMH))

        prev = self._ema.get(track_id)
        smoothed = kmh if prev is None else (self.smoothing * kmh + (1 - self.smoothing) * prev)
        self._ema[track_id] = smoothed
        return smoothed


def draw_overlay(frame, tracks, history, active_alerts, t, speed_estimator,
                  show_alert_panel=True):
    """
    Draws in-place onto `frame`:
      - a box + id + live km/h speed readout (below the box) per tracked object
      - boxes belonging to a currently-active alert get a thicker red outline
      - (optional) a persistent accident-info panel (top-left) listing recent
        alerts, color-coded by severity — disable when a separate UI panel
        (e.g. the desktop app's side report) already shows this, to avoid
        showing the same information twice.

    active_alerts: list of {"payload": AlertPayload, "fired_t": float}
    """
    alert_track_ids = set()
    for a in active_alerts:
        alert_track_ids.update(a["payload"].track_ids)

    for tid, bbox, cls in tracks:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        is_alerting = tid in alert_track_ids
        base_color = COLORS["vehicle"] if cls in VEHICLE_CLASS_IDS else COLORS["person"]
        color = COLORS["alert"] if is_alerting else base_color
        thickness = 3 if is_alerting else 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        kmh = speed_estimator.estimate_kmh(tid, cls, bbox, history)
        id_label = f"#{tid}" + (f"  {kmh:.0f} km/h" if kmh is not None else "")
        (tw, th), _ = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1 - 2), (0, 0, 0), -1)
        cv2.putText(frame, id_label, (x1 + 3, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    if show_alert_panel:
        draw_alert_panel(frame, active_alerts, t)
    return frame


def draw_alert_panel(frame, active_alerts, t, display_seconds=ALERT_DISPLAY_SECONDS,
                      max_rows=3):
    if not active_alerts:
        return

    shown = active_alerts[-max_rows:]
    overflow = len(active_alerts) - len(shown)

    pad = 8
    line_h = 22
    rows = len(shown) + (1 if overflow > 0 else 0)
    panel_w = 360
    panel_h = pad * 2 + line_h * (rows + 1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), COLORS["panel_bg"], -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = 10 + pad + 16
    cv2.putText(frame, "ACCIDENT ALERTS", (10 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    y += line_h

    for a in shown:
        payload = a["payload"]
        color = (COLORS["violence"] if payload.kind == "violence"
                 else SEVERITY_COLORS.get(payload.severity, COLORS["alert"]))
        cv2.rectangle(frame, (10 + pad, y - 12), (10 + pad + 12, y), color, -1)
        label = f"{payload.kind.replace('_', ' ')} [{payload.severity}]  tracks={payload.track_ids}"
        cv2.putText(frame, label, (10 + pad + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    color, 1, cv2.LINE_AA)
        y += line_h

    if overflow > 0:
        cv2.putText(frame, f"+ {overflow} more", (10 + pad + 18, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


def draw_skeletons(frame, persons, violent_ids=frozenset()):
    """Draws COCO skeleton lines + joints for each (track_id, bbox, kpts_xy)
    person; joints with NaN visibility are skipped. Persons in violent_ids
    (track ids of an active violence alert) get magenta limbs, everyone else
    gets green."""
    for tid, _bbox, kpts in persons:
        if kpts is None or len(kpts) < 17:
            continue
        color = COLORS["violence"] if tid in violent_ids else (120, 220, 120)
        for a, b in SKELETON_PAIRS:
            if (not np.isfinite(kpts[a][0])) or (not np.isfinite(kpts[b][0])):
                continue
            pt_a = (int(kpts[a][0]), int(kpts[a][1]))
            pt_b = (int(kpts[b][0]), int(kpts[b][1]))
            cv2.line(frame, pt_a, pt_b, color, 2, cv2.LINE_AA)
        for k in range(17):
            if np.isfinite(kpts[k][0]):
                pt = (int(kpts[k][0]), int(kpts[k][1]))
                cv2.circle(frame, pt, 3, color, -1, cv2.LINE_AA)
    return frame
