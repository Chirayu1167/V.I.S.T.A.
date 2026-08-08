"""
Accident-branch accuracy benchmark: runs clips through the pipeline and
measures precision/recall against a labeled ground-truth spec.

Each clip is run once through the real AccidentPipeline (detector+tracker+
heuristics+verification+fusion+severity, no secondary confirmation unless
--secondary-weights), and every confirmed event (kind + timestamp) is
compared against the spec.

Spec file (JSON):
    {
      "clip_01.mp4": {"events": [{"kind": "collision", "t": 3.14, "tolerance": 1.5}]},
      "clip_03.mp4": {"events": [{"kind": "speed_drop", "t": 6.2, "tolerance": 1.5}]}
    }
  - A clip with no "events" key expects ZERO confirmed events (a true-negative
    clip — any event produced counts as a false positive).
  - Matching is per-kind within |t - expected.t| <= tolerance.

Modes (--mode):
  flat     - no homography: flat meter_per_pixel scale (the tuned baseline)
  auto     - auto_calibrate_video() on the clip itself (the GUI's default
             when no profile is loaded) — use to A/B whether auto-calibration
             shifts accuracy vs the flat baseline
  profile  - use --profile camera JSON (explicit calibration)

Usage:
    python -m vista_accident.tools.benchmark_accident --dir ../../Video --spec spec.json
    python -m vista_accident.tools.benchmark_accident --dir ../../Video --spec spec.json --mode auto
    python -m vista_accident.tools.benchmark_accident --dir ../../Video --spec spec.json --mode profile --profile ../camera_profiles/CAM-03.json

Exits 0 if every clip meets its expected events, 1 otherwise (CI-friendly).
"""

import argparse
import glob
import json
import os
import sys
import time
import warnings

import cv2

warnings.filterwarnings("ignore", message=".*'half'.*deprecated.*")

from vista_accident import AccidentPipeline, DispatchConfig
from vista_accident.camera_profile import load_profile
from vista_accident.config import (HOMOGRAPHY_THRESHOLD_SCALE, CameraConfig,
                                   HeuristicConfig, scale_thresholds)
from vista_accident.detector import Detector
from vista_accident.tracker import Tracker


def run_clip(path, mode, profile_path, device, threshold_scale, verbose=False):
    """Run one clip; return (fps, events) where events is a list of
    dicts {"kind", "t", "track_ids", "severity"} in confirm order."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    detector = Detector(device=device)
    camera_cfg = CameraConfig()
    calibration_note = None
    if mode == "profile":
        camera_cfg = load_profile(profile_path)
    elif mode == "auto":
        from vista_accident.auto_calibration import auto_calibrate_video
        auto_cfg = auto_calibrate_video(path, detector, camera_id="CAM-BENCH")
        if auto_cfg is not None:
            camera_cfg = auto_cfg
            calibration_note = auto_cfg.calibration_note

    heuristic_cfg = HeuristicConfig()

    pipeline = AccidentPipeline(
        detector=detector,
        tracker=Tracker(frame_rate=int(fps)),
        heuristic_cfg=heuristic_cfg,
        camera_cfg=camera_cfg,
        dispatch_cfg=DispatchConfig(dashboard_log_path="test_alerts.jsonl"),
        fps_hint=fps,
        use_ml_speed=True,
        threshold_scale=threshold_scale,
    )

    events = []
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_idx / fps
            result = pipeline.process_frame(frame, t)
            for ev in result["confirmed_events"]:
                events.append({"kind": ev.kind, "t": round(ev.t, 2),
                               "track_ids": list(ev.track_ids),
                               "meta": {k: v for k, v in ev.meta.items()
                                        if k in ("iou", "prior_v", "drop_ratio",
                                                 "ml_speed_kmph")}})
            frame_idx += 1
    finally:
        cap.release()
        pipeline.close()

    if verbose and calibration_note:
        print(f"    calibration: {calibration_note.split('.' )[0] if calibration_note else 'flat scale'}")
    return fps, events


def match_events(detected, expected, clip):
    """Match detected events to expected ones per kind+time. Returns
    (matched, fps) — every unmatched detected event is a false positive,
    every unmatched expected event is a miss."""
    matched = []
    for exp in expected.get("events", []):
        tol = exp.get("tolerance", 1.5)
        best = None
        for det in detected:
            if det["kind"] != exp["kind"]:
                continue
            if any(det is m for m in matched):
                continue
            if exp.get("t") is None:
                best = (0.0, det)  # "any time" expectation — match first
                break
            d = abs(det["t"] - exp["t"])
            if d <= tol and (best is None or d < best[0]):
                best = (d, det)
        if best is not None:
            matched.append(best[1])
    return matched


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="Video", help="Directory with clip_XX.mp4 files")
    ap.add_argument("--spec", required=True, help="Ground-truth JSON spec (see docstring)")
    ap.add_argument("--mode", default="flat", choices=["flat", "auto", "profile"])
    ap.add_argument("--profile", default=None, help="Camera profile for --mode profile")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--threshold-scale", type=float, default=None,
                    help="Multiply physical-velocity thresholds by this factor. "
                         "Default: 0.65 for auto/profile (matches the pipeline's "
                         "automatic homography re-anchor), 1.0 for flat.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    with open(args.spec, "r", encoding="utf-8") as f:
        spec = json.load(f)

    clips = sorted(glob.glob(os.path.join(args.dir, "*.mp4")))
    if not clips:
        raise SystemExit(f"No .mp4 files in {args.dir}")

    if args.threshold_scale is None:
        args.threshold_scale = (HOMOGRAPHY_THRESHOLD_SCALE if args.mode in ("auto", "profile")
                                else 1.0)

    t0 = time.time()
    total_tp = total_fp = total_miss = total_expected = 0
    all_pass = True
    print(f"Mode: {args.mode}  device: {args.device}\n")
    for path in clips:
        name = os.path.basename(path)
        if name not in spec and not args.verbose:
            continue  # only run clips present in the spec
        expected = spec.get(name, {"events": []})
        fps, detected = run_clip(path, args.mode, args.profile, args.device,
                                 threshold_scale=args.threshold_scale,
                                 verbose=args.verbose)
        matched = match_events(detected, expected, name)
        exp_events = expected.get("events", [])
        tp = len(matched)
        fp = len(detected) - tp
        miss = len(exp_events) - tp
        total_tp += tp; total_fp += fp; total_miss += miss
        total_expected += len(exp_events)
        ok = (fp == 0 and miss == 0)
        all_pass = all_pass and ok

        det_str = ", ".join(f"{e['kind']}@{e['t']}s" for e in detected) or "none"
        exp_str = ", ".join(f"{e['kind']}@{e['t']}s" for e in exp_events) or "none"
        print(f"[{'PASS' if ok else 'FAIL'}] {name:12s}  detected: {det_str:45s}"
              f"  expected: {exp_str:45s}  TP={tp} FP={fp} MISS={miss}")

    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 1.0
    rec = total_tp / total_expected if total_expected else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print(f"\nTotal: TP={total_tp} FP={total_fp} MISS={total_miss} "
          f"| precision={prec:.2f} recall={rec:.2f} F1={f1:.2f} "
          f"({time.time()-t0:.0f}s)")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
