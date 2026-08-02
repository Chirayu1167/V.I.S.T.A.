#!/usr/bin/env python3
"""
VISTA — multi-camera runner.

Runs one AccidentPipeline per configured camera, each on its own thread
(mirrors the accident+violence concurrency pattern documented in README.md —
same idea, applied across cameras instead of across branches). Every
pipeline's AlertDispatcher already logs asynchronously on its own
background thread and every AlertPayload carries its own camera_id, so
pointing them all at the SAME dashboard_log_path gives you one merged feed
for tools/dashboard.py without any extra plumbing.

Config file (--config, JSON):
    [
      {"source": "cam1.mp4", "camera_id": "CAM-01", "location": "MG Road Junction"},
      {"source": "rtsp://192.168.1.10/stream", "camera_id": "CAM-02", "location": "2nd Cross"}
    ]

Usage:
    python multi_camera.py --config cameras.json --log alerts.jsonl
    # in another terminal:
    python -m vista_accident.tools.dashboard --log alerts.jsonl

Note: each camera's AlertDispatcher opens its own file handle in append
mode on the shared log path. Individual JSONL lines are written+flushed as
single writes, which is fine for a demo's alert volume, but this is NOT a
guarantee of atomicity under heavy concurrent write load — for a real
multi-camera deployment, route through one shared writer/queue instead of
N independent file handles.
"""

import argparse
import json
import threading
import time

import cv2

from vista_accident import AccidentPipeline, CameraConfig, DispatchConfig, HeuristicConfig
from vista_accident.detector import Detector


def run_camera(cam_cfg: dict, log_path: str, device: str, stop_event: threading.Event):
    source = cam_cfg["source"]
    camera_id = cam_cfg.get("camera_id", source)
    location = cam_cfg.get("location", camera_id)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[{camera_id}] Could not open source: {source} — skipping.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    pipeline = AccidentPipeline(
        detector=Detector(device=device),
        heuristic_cfg=HeuristicConfig(),
        camera_cfg=CameraConfig(camera_id=camera_id, location_name=location),
        dispatch_cfg=DispatchConfig(dashboard_log_path=log_path),
        fps_hint=fps,
    )

    frame_idx = 0
    print(f"[{camera_id}] Started ({source}).")
    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                print(f"[{camera_id}] Source ended.")
                break
            t = frame_idx / fps
            result = pipeline.process_frame(frame, t)
            for payload in result["alerts"]:
                print(f"[{camera_id} t={t:.1f}s] DISPATCHED {payload.kind} "
                      f"severity={payload.severity} channels={payload.channels}")
            frame_idx += 1
    finally:
        cap.release()
        pipeline.close()
        print(f"[{camera_id}] Stopped after {frame_idx} frames.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON list of {source, camera_id, location}")
    ap.add_argument("--log", default="alerts.jsonl", help="Shared alerts log for all cameras")
    ap.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    args = ap.parse_args()

    with open(args.config) as f:
        cameras = json.load(f)
    if not cameras:
        raise SystemExit("Config file has no cameras.")

    stop_event = threading.Event()
    threads = []
    for cam_cfg in cameras:
        th = threading.Thread(target=run_camera, args=(cam_cfg, args.log, args.device, stop_event), daemon=True)
        th.start()
        threads.append(th)

    print(f"Running {len(threads)} camera(s). Ctrl+C to stop all.")
    try:
        while any(th.is_alive() for th in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping all cameras...")
        stop_event.set()
        for th in threads:
            th.join(timeout=5)


if __name__ == "__main__":
    main()
