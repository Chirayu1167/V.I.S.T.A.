#!/usr/bin/env python3
"""
VISTA — Accident Detection demo runner.

Runs the accident branch on a local video file (stand-in for an RTSP CCTV
feed), overlays track boxes + heuristic triggers, writes an annotated output
video, and logs confirmed alerts to alerts.jsonl (the "dashboard" log).

Usage:
    python demo.py --source path/to/video.mp4 --output out.mp4

For a live camera / RTSP feed, just pass the RTSP URL as --source — cv2
handles both identically.
"""

import argparse
import time

import cv2

from vista_accident import AccidentPipeline, CameraConfig, DispatchConfig, HeuristicConfig

COLORS = {
    "vehicle": (60, 180, 255),
    "person": (80, 220, 120),
    "alert": (40, 40, 255),
}
VEHICLE_CLASS_IDS = {2, 3, 5, 7}


def draw_overlay(frame, tracks, confirmed_events):
    for tid, bbox, cls in tracks:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        color = COLORS["vehicle"] if cls in VEHICLE_CLASS_IDS else COLORS["person"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"#{tid}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    y_off = 30
    for event in confirmed_events:
        label = f"ALERT: {event.kind.upper()} tracks={event.track_ids}"
        cv2.putText(frame, label, (15, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    COLORS["alert"], 2, cv2.LINE_AA)
        y_off += 28
    return frame


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

        annotated = draw_overlay(frame, result["tracks"], result["confirmed_events"])
        writer.write(annotated)

        if result["confirmed_events"]:
            for ev in result["confirmed_events"]:
                print(f"[t={t:.2f}s] CONFIRMED {ev.kind} tracks={ev.track_ids} "
                      f"streak={ev.consecutive_frames} meta={ev.meta}")

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
