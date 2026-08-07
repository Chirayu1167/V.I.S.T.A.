#!/usr/bin/env python3
"""
VISTA — multi-camera runner (shared, batched inference).

Runs one AccidentPipeline (+ optional ViolencePipeline) per configured
camera, each on its own thread — same per-camera behavior as before
(per-camera stop zones, thresholds, homography, dispatch, config
hot-reload, clip export, all unchanged). Every pipeline's AlertDispatcher
still logs asynchronously on its own background thread and every
AlertPayload still carries its own camera_id, so pointing them all at the
SAME dashboard_log_path still gives you one merged feed — and, with the
ML bridge enabled, one merged view on the Emergency Response dashboards
(emergency_response/server.py) without any extra plumbing.

What's different from the original per-camera-model version: there is now
exactly ONE Detector and (if --enable-violence) ONE PoseDetector for the
whole process, shared across every camera thread via a SharedInferenceHub.
Each camera's capture thread submits its frame to the hub instead of
building and calling its own model — this is what lets N cameras share GPU
memory instead of each holding a full private copy, and lets the hub batch
frames across cameras into one forward pass per cycle instead of firing N
unbatched calls. See vista_accident/batch_inference.py for the collector
implementation and vista_accident/config.py's CameraConfig.detection_
conf_threshold / detection_classes for per-camera detection tuning under
this shared-model setup.

Config file (--config, JSON):
    [
      {"source": "cam1.mp4", "camera_id": "CAM-01", "location": "MG Road Junction"},
      {"source": "rtsp://192.168.1.10/stream", "camera_id": "CAM-02", "location": "2nd Cross"}
    ]

Usage:
    python multi_camera.py --config cameras.json --log alerts.jsonl
    python multi_camera.py --config cameras.json --log alerts.jsonl --enable-violence
    # in another terminal (dispatch UI — siren/banner/clips/status):
    python -m vista_accident.emergency_response.server --port 8890
    python demo.py --source cam.mp4 --emergency-response-url http://127.0.0.1:8890/api/incidents

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

from vista_accident import (
    AccidentPipeline,
    CameraConfig,
    Detector,
    DispatchConfig,
    HeuristicConfig,
    PoseDetector,
    SharedInferenceHub,
    ViolenceConfig,
)
from vista_accident.violence_pipeline import ViolencePipeline


def run_camera(cam_cfg: dict, log_path: str, hub: SharedInferenceHub,
                enable_violence: bool, stop_event: threading.Event):
    source = cam_cfg["source"]
    camera_id = cam_cfg.get("camera_id", source)
    location = cam_cfg.get("location", camera_id)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[{camera_id}] Could not open source: {source} — skipping.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Per-camera config stays exactly as before — only the detector
    # instantiation moved out to the shared hub. Anything a camera needs
    # tuned individually (stop zones, homography, thresholds, and now the
    # optional detection_conf_threshold/detection_classes overrides) still
    # lives here, per camera.
    camera_cfg = CameraConfig(
        camera_id=camera_id,
        location_name=location,
        stop_zones=cam_cfg.get("stop_zones", []),
        detection_conf_threshold=cam_cfg.get("detection_conf_threshold"),
        detection_classes=cam_cfg.get("detection_classes"),
    )
    client = hub.client_for(camera_cfg)

    pipeline = AccidentPipeline(
        heuristic_cfg=HeuristicConfig(),
        camera_cfg=camera_cfg,
        dispatch_cfg=DispatchConfig(dashboard_log_path=log_path),
        fps_hint=fps,
    )

    violence_pipeline = None
    if enable_violence:
        violence_pipeline = ViolencePipeline(
            cfg=ViolenceConfig(),
            camera_cfg=camera_cfg,
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

            # Detection now goes through the shared hub (batched across all
            # camera threads submitting this cycle) instead of each camera
            # calling its own model. Everything downstream — tracking,
            # heuristics, verification, severity, dispatch, clip export —
            # is unchanged and still runs per-camera.
            detections = client.detect_accident(frame)
            result = pipeline.process_frame(frame, t, detections=detections)
            for payload in result["alerts"]:
                print(f"[{camera_id} t={t:.1f}s] DISPATCHED {payload.kind} "
                      f"severity={payload.severity} channels={payload.channels}")

            if violence_pipeline is not None:
                persons = client.detect_violence(frame)
                vresult = violence_pipeline.process_frame(frame, t, persons=persons)
                for payload in vresult["alerts"]:
                    print(f"[{camera_id} t={t:.1f}s] DISPATCHED {payload.kind} "
                          f"severity={payload.severity} channels={payload.channels}")

            frame_idx += 1
    finally:
        cap.release()
        pipeline.close()
        if violence_pipeline is not None:
            violence_pipeline.close()
        print(f"[{camera_id}] Stopped after {frame_idx} frames.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON list of {source, camera_id, location}")
    ap.add_argument("--log", default="alerts.jsonl", help="Shared alerts log for all cameras")
    ap.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    ap.add_argument("--enable-violence", action="store_true",
                     help="Also run the violence branch per camera, sharing one PoseDetector.")
    ap.add_argument("--batch-window-ms", type=float, default=8.0,
                     help="Max wait to fill the accident-detector micro-batch before firing "
                          "anyway. Bounds worst-case added latency (default 8ms).")
    args = ap.parse_args()

    with open(args.config) as f:
        cameras = json.load(f)
    if not cameras:
        raise SystemExit("Config file has no cameras.")

    # Exactly ONE Detector (and, if requested, ONE PoseDetector) for the
    # whole process — shared across every camera thread via the hub below,
    # instead of one full model copy per camera.
    accident_detector = Detector(device=args.device)
    pose_detector = PoseDetector(ViolenceConfig(), device=args.device) if args.enable_violence else None
    hub = SharedInferenceHub(
        accident_detector, pose_detector,
        accident_window_s=args.batch_window_ms / 1000.0,
    )

    stop_event = threading.Event()
    threads = []
    for cam_cfg in cameras:
        th = threading.Thread(
            target=run_camera,
            args=(cam_cfg, args.log, hub, args.enable_violence, stop_event),
            daemon=True,
        )
        th.start()
        threads.append(th)

    print(f"Running {len(threads)} camera(s) against a shared "
          f"{'accident+violence' if args.enable_violence else 'accident-only'} "
          f"inference hub. Ctrl+C to stop all.")
    try:
        while any(th.is_alive() for th in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping all cameras...")
        stop_event.set()
        for th in threads:
            th.join(timeout=5)
    finally:
        hub.close()


if __name__ == "__main__":
    main()
