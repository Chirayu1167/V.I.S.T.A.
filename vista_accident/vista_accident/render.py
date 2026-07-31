"""
Shared frame-annotation logic: track boxes, per-track speed readout, and the
persistent accident-alert panel. Used by both the CLI runner (demo.py) and
the desktop GUI (gui_app.py) so the two stay visually consistent.
"""

import cv2

COLORS = {
    "vehicle": (60, 180, 255),
    "person": (80, 220, 120),
    "alert": (40, 40, 255),
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

# How long a confirmed alert stays visible in the on-screen panel by default.
ALERT_DISPLAY_SECONDS = 4.0
# Speed look-back window (seconds) — short enough to feel responsive,
# long enough to smooth out per-frame tracker jitter.
SPEED_WINDOW_S = 0.4


def format_speed(px_per_s, px_per_meter=None):
    """Render a speed value either as raw px/s or, if a calibration factor
    is supplied, converted to km/h."""
    if px_per_s is None:
        return None
    if px_per_meter:
        kmh = (px_per_s / px_per_meter) * 3.6
        return f"{kmh:.0f} km/h"
    return f"{px_per_s:.0f} px/s"


def draw_overlay(frame, tracks, history, active_alerts, t, px_per_meter=None,
                  show_alert_panel=True):
    """
    Draws in-place onto `frame`:
      - a box + id + live speed readout (below the box) per tracked object
      - boxes belonging to a currently-active alert get a thicker red outline
      - (optional) a persistent accident-info panel (top-left) listing recent
        alerts, color-coded by severity

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
        thickness = 3 if is_alerting else 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, f"#{tid}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # --- speed readout, below the box ---
        speed = history.velocity(tid, SPEED_WINDOW_S)
        if speed is None:
            speed = history.instantaneous_velocity(tid)
        speed_label = format_speed(speed, px_per_meter)
        if speed_label:
            text_y = y2 + 18
            (tw, th), _ = cv2.getTextSize(speed_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, text_y - th - 4), (x1 + tw + 6, text_y + 4),
                          (0, 0, 0), -1)
            cv2.putText(frame, speed_label, (x1 + 3, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["speed_text"], 1, cv2.LINE_AA)

    if show_alert_panel:
        draw_alert_panel(frame, active_alerts, t)
    return frame


def draw_alert_panel(frame, active_alerts, t, display_seconds=ALERT_DISPLAY_SECONDS):
    if not active_alerts:
        return

    pad = 10
    line_h = 26
    header = "ACCIDENT ALERTS"
    panel_w = 430
    panel_h = pad * 2 + line_h * (len(active_alerts) + 1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), COLORS["panel_bg"], -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    y = 10 + pad + 18
    cv2.putText(frame, header, (10 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    y += line_h

    for a in active_alerts:
        payload = a["payload"]
        severity = payload.severity
        color = SEVERITY_COLORS.get(severity, COLORS["alert"])
        age = t - a["fired_t"]
        remaining = max(0.0, display_seconds - age)

        cv2.rectangle(frame, (10 + pad, y - 14), (10 + pad + 14, y), color, -1)

        channels = "+".join(c.replace("_", " ") for c in payload.channels)
        label = (f"{payload.kind.upper()} [{severity.upper()}]  "
                 f"tracks={payload.track_ids}  -> {channels}  ({remaining:.1f}s)")
        cv2.putText(frame, label, (10 + pad + 22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)
        y += line_h
