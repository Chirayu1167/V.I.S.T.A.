#!/usr/bin/env python3
"""
VISTA — Accident Detection demo runner (CLI).

Runs the accident branch on a local video file (stand-in for an RTSP CCTV
feed), overlays track boxes + per-track speed + heuristic triggers, writes
an annotated output video, and logs confirmed alerts to alerts.jsonl (the
"dashboard" log).

For an interactive local desktop UI (upload button, live preview, side-panel
report, impact screenshots) use gui_app.py instead — see:
    python gui_app.py

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
import json
import time
from collections import deque

import cv2

from vista_accident import AccidentPipeline, CameraConfig, DispatchConfig, HeuristicConfig
from vista_accident.render import ALERT_DISPLAY_SECONDS, SpeedEstimator, draw_overlay


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
                     help="Optional manual camera calibration (pixels per real-world meter). "
                          "If omitted, speed is auto-estimated per object from its own "
                          "bounding-box width vs. its class's typical real-world size — "
                          "less precise than a real calibration, but always shown in km/h.")
    ap.add_argument("--alert-display-seconds", type=float, default=ALERT_DISPLAY_SECONDS,
                     help="How long a confirmed alert stays shown in the on-screen panel.")
    ap.add_argument("--stop-zones-json", default=None,
                     help="Optional JSON file with a list of stop-zone polygons "
                          "(each: [[x,y],...]) where vehicles legitimately stop "
                          "(intersections, bus stops) — suppresses speed_drop/anomaly_stop there.")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    from vista_accident.detector import Detector
    from vista_accident.confirmation import SecondaryConfirmation

    stop_zones = []
    if args.stop_zones_json:
        with open(args.stop_zones_json) as f:
            stop_zones = json.load(f)

    pipeline = AccidentPipeline(
        detector=Detector(device=args.device),
        heuristic_cfg=HeuristicConfig(),
        camera_cfg=CameraConfig(camera_id=args.camera_id, location_name=args.location,
                                stop_zones=stop_zones),
        dispatch_cfg=DispatchConfig(dashboard_log_path="alerts.jsonl"),
        secondary=SecondaryConfirmation(weights_path=args.secondary_weights, device=args.device),
        fps_hint=fps,
    )

    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Rolling window of alerts still worth showing on screen.
    active_alerts = deque()
    speed_estimator = SpeedEstimator(manual_px_per_meter=args.px_per_meter)

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

        for payload in result["alerts"]:
            active_alerts.append({"payload": payload, "fired_t": t})
        while active_alerts and (t - active_alerts[0]["fired_t"]) > args.alert_display_seconds:
            active_alerts.popleft()

        annotated = draw_overlay(frame, result["tracks"], pipeline.history,
                                  list(active_alerts), t, speed_estimator)
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
