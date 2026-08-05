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
from vista_accident.render import ALERT_DISPLAY_SECONDS, SpeedEstimator, draw_overlay, draw_skeletons


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
                          "(intersections, bus stops) — suppresses speed_drop/anomaly_stop there. "
                          "Draw one with: python -m vista_accident.tools.draw_stop_zones")
    ap.add_argument("--clip-dir", default=None,
                     help="If set, saves a short .mp4 clip (pre/post impact) per dispatched "
                          "alert into this directory instead of just a single impact frame.")
    ap.add_argument("--enable-plate-ocr", action="store_true",
                     help="Run license-plate OCR (requires `pip install easyocr`) on "
                          "hit_and_run alerts and include plate_text in the payload.")
    ap.add_argument("--violence", action="store_true",
                     help="Also run the pose-based violence/road-rage branch (yolo11n-pose, "
                          "auto-downloaded on first run) — pair proximity + limb motion "
                          "signals route to the police control room.")
    ap.add_argument("--no-accident", action="store_true",
                     help="Run ONLY the violence branch (skip the yolo11m accident model). "
                          "Use with --violence for a smooth violence-only test — the "
                          "accident model is the performance bottleneck when both run.")
    ap.add_argument("--watch-config", default=None,
                     help="Optional JSON file (see vista_accident.config.ConfigWatcher) to "
                          "hot-reload heuristic thresholds / stop_zones from during the run, "
                          "without restarting — useful for live threshold tuning.")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    from vista_accident.detector import Detector
    from vista_accident.confirmation import SecondaryConfirmation
    from vista_accident.plate_reader import PlateReader
    from vista_accident.config import ConfigWatcher

    stop_zones = []
    if args.stop_zones_json:
        with open(args.stop_zones_json) as f:
            stop_zones = json.load(f)

    heuristic_cfg = HeuristicConfig()
    camera_cfg = CameraConfig(camera_id=args.camera_id, location_name=args.location,
                               stop_zones=stop_zones)

    watcher = None
    if args.watch_config:
        watcher = ConfigWatcher(args.watch_config, heuristic_cfg=heuristic_cfg, camera_cfg=camera_cfg)
        watcher.start()
        print(f"Hot-reloading thresholds/stop_zones from {args.watch_config} every "
              f"{watcher.interval_s:.0f}s.")

    pipeline = None
    if not args.no_accident:
        pipeline = AccidentPipeline(
            detector=Detector(device=args.device),
            heuristic_cfg=heuristic_cfg,
            camera_cfg=camera_cfg,
            dispatch_cfg=DispatchConfig(dashboard_log_path="alerts.jsonl"),
            secondary=SecondaryConfirmation(weights_path=args.secondary_weights, device=args.device),
            fps_hint=fps,
            clip_dir=args.clip_dir,
            plate_reader=PlateReader() if args.enable_plate_ocr else None,
        )

    violence_pipeline = None
    if args.violence:
        from vista_accident.violence_pipeline import ViolencePipeline
        violence_pipeline = ViolencePipeline(
            camera_cfg=camera_cfg,
            dispatch_cfg=DispatchConfig(dashboard_log_path="alerts.jsonl"),
            device=args.device,
            fps_hint=fps,
            clip_dir=args.clip_dir,
        )
        print("Violence branch enabled (yolo11n-pose, runs every 3rd frame).")
    if pipeline is None and violence_pipeline is None:
        raise SystemExit("Nothing to run: pass --violence (and/or drop --no-accident).")

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
        result = {"tracks": [], "alerts": [], "confirmed_events": [], "clips_saved": []}
        if pipeline is not None:
            result = pipeline.process_frame(frame, t)

        violence_alerts = []
        if violence_pipeline is not None:
            vres = violence_pipeline.process_frame(frame, t)
            violence_alerts = vres["alerts"]
            for vid, cpath in vres.get("clips_saved", []):
                print(f"[t={t:.2f}s] VIOLENCE CLIP saved -> {cpath}")
            for payload in violence_alerts:
                print(f"[t={t:.2f}s] DISPATCHED {payload.kind} severity={payload.severity} "
                      f"channels={payload.channels}")

        all_alerts = result["alerts"] + violence_alerts
        for payload in all_alerts:
            active_alerts.append({"payload": payload, "fired_t": t})
        while active_alerts and (t - active_alerts[0]["fired_t"]) > args.alert_display_seconds:
            active_alerts.popleft()

        annotated = frame.copy()
        if violence_pipeline is not None:
            violent_ids = {tid for a in active_alerts
                           if a["payload"].kind == "violence"
                           for tid in a["payload"].track_ids}
            draw_skeletons(annotated, violence_pipeline.latest_persons, violent_ids=violent_ids)
        if pipeline is not None:
            draw_overlay(annotated, result["tracks"], pipeline.history,
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
    if watcher:
        watcher.stop()
    if pipeline is not None:
        pipeline.close()  # flush any alerts still queued for the async log writer
    if violence_pipeline:
        violence_pipeline.close()

    elapsed = time.time() - t0
    n_alerts = n_dispatched = 0
    if pipeline is not None:
        n_alerts = len(pipeline.confirmed_log)
        n_dispatched = sum(1 for _, _, status in pipeline.confirmed_log if status == "dispatched")
    print(f"\nProcessed {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx / elapsed:.1f} fps effective).")
    print(f"Confirmed events: {n_alerts} | Dispatched alerts: {n_dispatched}")
    if violence_pipeline:
        n_v = len(violence_pipeline.confirmed_log)
        n_vd = sum(1 for _, _, status in violence_pipeline.confirmed_log if status == "dispatched")
        print(f"Violence branch: {n_v} confirmed | {n_vd} dispatched")
    print(f"Annotated video -> {args.output}")
    print(f"Alert log        -> alerts.jsonl")


if __name__ == "__main__":
    main()
