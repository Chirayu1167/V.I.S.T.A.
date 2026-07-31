#!/usr/bin/env python3
"""
VISTA — Accident Detection demo runner.

Runs the accident branch on a local video file (stand-in for an RTSP CCTV
feed), overlays track boxes + per-track speed + heuristic triggers, writes
an annotated output video, and logs confirmed alerts to alerts.jsonl (the
"dashboard" log).

Usage:
    python demo.py --source path/to/video.mp4 --output out.mp4

    # If you know the camera's real-world scale (e.g. from a calibration
    # pass / known lane width), pass --px-per-meter to show speed in km/h
    # instead of raw px/s:
    python demo.py --source video.mp4 --px-per-meter 18.5

For a live camera / RTSP feed, just pass the RTSP URL as --source — cv2
handles both identically.
"""

import argparse
import time
from collections import deque

import cv2

from vista_accident import AccidentPipeline, CameraConfig, DispatchConfig, HeuristicConfig

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

# How long a confirmed alert stays visible on screen after it fires.
ALERT_DISPLAY_SECONDS = 4.0
# Speed look-back window (seconds) — short enough to feel responsive,
# long enough to smooth out per-frame tracker jitter.
SPEED_WINDOW_S = 0.4


def _format_speed(px_per_s, px_per_meter):
    """Render a speed value either as raw px/s or, if a calibration factor
    is supplied, converted to km/h."""
    if px_per_s is None:
        return None
    if px_per_meter:
        kmh = (px_per_s / px_per_meter) * 3.6
        return f"{kmh:.0f} km/h"
    return f"{px_per_s:.0f} px/s"


def draw_overlay(frame, tracks, history, active_alerts, t, px_per_meter=None):
    """
    Draws:
      - a box + id + live speed readout (below the box) per tracked object
      - boxes belonging to a currently-active alert get a thicker red outline
      - a persistent accident-info panel (top-left) listing recent alerts,
        color-coded by severity, for ALERT_DISPLAY_SECONDS after they fire
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
        speed_label = _format_speed(speed, px_per_meter)
        if speed_label:
            text_y = y2 + 18
            (tw, th), _ = cv2.getTextSize(speed_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, text_y - th - 4), (x1 + tw + 6, text_y + 4),
                          (0, 0, 0), -1)
            cv2.putText(frame, speed_label, (x1 + 3, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["speed_text"], 1, cv2.LINE_AA)

    _draw_alert_panel(frame, active_alerts, t)
    return frame


def _draw_alert_panel(frame, active_alerts, t):
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
        remaining = max(0.0, ALERT_DISPLAY_SECONDS - age)

        # Severity swatch
        cv2.rectangle(frame, (10 + pad, y - 14), (10 + pad + 14, y), color, -1)

        channels = "+".join(c.replace("_", " ") for c in payload.channels)
        label = (f"{payload.kind.upper()} [{severity.upper()}]  "
                 f"tracks={payload.track_ids}  -> {channels}  ({remaining:.1f}s)")
        cv2.putText(frame, label, (10 + pad + 22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)
        y += line_h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Video file path or RTSP URL")
    ap.add_argument("--output", default="vista_output.mp4", help="Annotated output video path")
    ap.add_argument("--camera-id", default="CAM-01")
    ap.add_argument("--location", default="MG Road & 2nd Cross Junction")
    ap.add_argument("--max-frames", type=int, default=None, help="Limit frames processed (debug)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--secondary-weights", default=None,
                     help="Optional path to YOLO11x accident-confirmation weights")
    ap.add_argument("--px-per-meter", type=float, default=None,
                     help="Optional calibration factor (pixels per real-world meter) to "
                          "display speed in km/h instead of raw px/s.")
    ap.add_argument("--alert-display-seconds", type=float, default=ALERT_DISPLAY_SECONDS,
                     help="How long a confirmed alert stays shown in the on-screen panel.")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    from vista_accident.detector import Detector
    from vista_accident.confirmation import SecondaryConfirmation

    pipeline = AccidentPipeline(
        detector=Detector(device=args.device),
        heuristic_cfg=HeuristicConfig(),
        camera_cfg=CameraConfig(camera_id=args.camera_id, location_name=args.location),
        dispatch_cfg=DispatchConfig(dashboard_log_path="alerts.jsonl"),
        secondary=SecondaryConfirmation(weights_path=args.secondary_weights, device=args.device),
        fps_hint=fps,
    )

    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Rolling window of alerts still worth showing on screen.
    active_alerts = deque()

    frame_idx = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and frame_idx >= args.max_frames:
            break

        t = frame_idx / fps  # use video-relative timestamps for reproducible demo runs
        result = pipeline.process_frame(frame, t)

        # New dispatched alerts (carry severity + channels) join the on-screen panel.
        for payload in result["alerts"]:
            active_alerts.append({"payload": payload, "fired_t": t})

        # Drop alerts that have aged out of the display window.
        while active_alerts and (t - active_alerts[0]["fired_t"]) > args.alert_display_seconds:
            active_alerts.popleft()

        annotated = draw_overlay(frame, result["tracks"], pipeline.history,
                                  list(active_alerts), t, px_per_meter=args.px_per_meter)
        writer.write(annotated)

        if result["confirmed_events"]:
            for ev in result["confirmed_events"]:
                print(f"[t={t:.2f}s] CONFIRMED {ev.kind} tracks={ev.track_ids} "
                      f"streak={ev.consecutive_frames} meta={ev.meta}")
        if result["alerts"]:
            for payload in result["alerts"]:
                print(f"[t={t:.2f}s] DISPATCHED {payload.kind} severity={payload.severity} "
                      f"channels={payload.channels}")

        frame_idx += 1

    cap.release()
    writer.release()

    elapsed = time.time() - t0
    n_alerts = len(pipeline.confirmed_log)
    n_dispatched = sum(1 for _, _, status in pipeline.confirmed_log if status == "dispatched")
    print(f"\nProcessed {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx / elapsed:.1f} fps effective).")
    print(f"Confirmed events: {n_alerts} | Dispatched alerts: {n_dispatched}")
    print(f"Annotated video -> {args.output}")
    print(f"Alert log        -> alerts.jsonl")


if __name__ == "__main__":
    main()
