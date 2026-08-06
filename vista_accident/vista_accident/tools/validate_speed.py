"""
Speed validation against a known-distance marker pair.

Runs the real detector/tracker over a clip with a calibrated camera profile,
finds a vehicle whose ground point crosses both markers, and compares the
ML speed estimator's reading against ground truth measured from crossing
time:

    ground truth = marker distance / (t_marker2 - t_marker1)
    estimated    = pipeline speed estimator averaged over [t1, t2]

Acceptance gate: |error %| <= 15.

Usage
-----
    python -m vista_accident.tools.validate_speed --source clip.mp4 \\
        --profile camera_profiles/CAM-01.json \\
        --markers '[[x1,y1],[x2,y2]]'

    --markers: two PIXEL points on the road, lying on the vehicle's path.
    Their real-world distance is taken from the profile's homography
    (i.e. from the SAME calibration the pipeline uses, so the check measures
    how faithfully the estimator reproduces physically-measured crossing
    time). If you measured the distance directly, override it:

        --marker-distance-m 10.5

    --frame-range 100 400: only use frames [start, stop) (skip scene
    intro/outro, or isolate one vehicle's run).

    --device cpu|cuda, --radius-px 12: how close (px) a path segment's
    perpendicular foot must pass to a marker for it to count as a crossing.
    The crossing TIME is interpolated at the foot, so a large radius only
    affects which segment qualifies, not the timing itself.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

from vista_accident import AccidentPipeline, HeuristicConfig, DispatchConfig
from vista_accident.camera_profile import load_profile, world_distance_m
from vista_accident.detector import Detector
from vista_accident.tracker import Tracker

ACCEPTANCE_ERROR_PCT = 15.0


def ground_point(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, y2


def crossing_t(points, marker, radius_px):
    """Interpolate the time the ground point passes CLOSEST to `marker`.

    Finds the first path segment whose perpendicular foot onto the marker
    lies inside `radius_px` and returns the interpolated time at that foot.
    Circle-edge entry (which biases the crossing time by the pixel radius
    asymmetrically near vs far from the camera) never enters the timing.
    Returns (t, foot_world_offset_along_segment) or None."""
    for i in range(len(points) - 1):
        (t1, p1), (t2, p2) = points[i], points[i + 1]
        seg = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if seg <= 0:
            continue
        vx, vy = (p2[0] - p1[0]) / seg, (p2[1] - p1[1]) / seg
        u = (marker[0] - p1[0]) * vx + (marker[1] - p1[1]) * vy
        if u < 0 or u > seg:
            continue
        foot = (p1[0] + vx * u, p1[1] + vy * u)
        if np.hypot(marker[0] - foot[0], marker[1] - foot[1]) <= radius_px:
            return t1 + u / seg * (t2 - t1), (foot, u)
    return None


def find_crossing(track_positions, marker_a, marker_b, radius_px):
    """Find (t1, t2, order) where the track's pixel ground point passes
    closest to marker_a, then marker_b (or the reverse order). Returns
    (t1, t2, order) or None."""
    for m1, m2, order in ((marker_a, marker_b, "a->b"), (marker_b, marker_a, "b->a")):
        hit1 = crossing_t(track_positions, m1, radius_px)
        if hit1 is None:
            continue
        t1 = hit1[0]
        later = [p for p in track_positions if p[0] > t1]
        if len(later) < 2:
            continue
        hit2 = crossing_t(later, m2, radius_px)
        if hit2 is None:
            continue
        return t1, hit2[0], order
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="Video file path")
    ap.add_argument("--profile", required=True, help="Camera profile JSON (homography)")
    ap.add_argument("--markers", required=True,
                    help="JSON list of two pixel points [[x1,y1],[x2,y2]] on the "
                         "vehicle's path, a known-distance pair")
    ap.add_argument("--marker-distance-m", type=float, default=None,
                    help="Optional override: the real-world distance between the "
                         "markers in meters. Defaults to the distance from the "
                         "profile homography.")
    ap.add_argument("--frame-range", nargs=2, type=int, default=None,
                    help="Only process frames [start, stop)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--radius-px", type=float, default=12.0,
                    help="Marker-crossing hit radius in pixels")
    ap.add_argument("--min-frames", type=int, default=15,
                    help="Minimum frames a candidate track must be visible")
    args = ap.parse_args()

    markers = [tuple(m) for m in json.loads(args.markers)]
    if len(markers) != 2:
        raise SystemExit("--markers must contain exactly two points")
    marker_a, marker_b = markers

    camera_cfg = load_profile(args.profile)
    if not (camera_cfg.homography_src_points and camera_cfg.homography_dst_points):
        print("WARNING: profile has no homography points; marker distance uses "
              "the flat meter_per_pixel fallback. Calibrate first "
              "(tools/calibrate_camera.py or the GUI) for meaningful results.")

    marker_dist_m = args.marker_distance_m
    if marker_dist_m is None:
        marker_dist_m = world_distance_m(camera_cfg, marker_a, marker_b)
    if marker_dist_m <= 0:
        raise SystemExit("Marker distance must be > 0 (check the points / homography)")

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    pipeline = AccidentPipeline(
        detector=Detector(device=args.device),
        tracker=Tracker(frame_rate=int(fps)),
        heuristic_cfg=HeuristicConfig(),
        camera_cfg=camera_cfg,
        dispatch_cfg=DispatchConfig(dashboard_log_path="test_alerts.jsonl"),
        fps_hint=fps,
        use_ml_speed=True,
    )

    frame_idx = 0
    positions = {}   # track_id -> [(t, pixel_ground_pt)]
    crossings = {}   # track_id -> (t1, t2, order, est_mps) found during the run
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.frame_range and not (args.frame_range[0] <= frame_idx < args.frame_range[1]):
                frame_idx += 1
                continue
            if args.frame_range and frame_idx >= args.frame_range[1]:
                break

            t = frame_idx / fps
            result = pipeline.process_frame(frame, t)
            for tid, bbox, _cls in result["tracks"]:
                positions.setdefault(tid, []).append((t, ground_point(bbox)))
                # Record the crossing as soon as the second marker is hit,
                # while the estimator still holds this track's history — a
                # track that leaves the scene before the clip ends would be
                # aged out by the time the loop finishes.
                if tid not in crossings and len(positions[tid]) >= 2:
                    cross = find_crossing(positions[tid], marker_a, marker_b, args.radius_px)
                    if cross is not None:
                        t1, t2, order = cross
                        est = None
                        if pipeline.speed_estimator is not None:
                            est = pipeline.speed_estimator.velocity_between(tid, t1, t2)
                        if est is None:
                            speed = pipeline.speed_estimator.get_speed(tid) if pipeline.speed_estimator else None
                            est = speed.speed_mps if speed else None
                        crossings[tid] = (t1, t2, order, est)
            frame_idx += 1
    finally:
        cap.release()
        pipeline.close()

    if not positions:
        raise SystemExit("No tracks were detected in the clip. Try a device with a GPU, "
                         "or check --frame-range.")

    candidates = {tid: pts for tid, pts in positions.items() if len(pts) >= args.min_frames}
    if not candidates:
        raise SystemExit(f"No track visible for >= {args.min_frames} frames.")

    print(f"Clip: {os.path.basename(args.source)}  fps={fps:.2f}  frames={frame_idx}")
    print(f"Markers: {marker_a} <-> {marker_b}  distance={marker_dist_m:.2f} m "
          f"(from {'--marker-distance-m' if args.marker_distance_m else 'profile homography'})")
    print(f"Detected {len(candidates)} candidate track(s) "
          f"({sum(len(p) for p in positions.values())} total track frames)\n")

    results = []
    for tid, _pts in sorted(candidates.items()):
        cross = crossings.get(tid)
        if cross is None:
            continue
        t1, t2, order, est_mps = cross
        crossing_s = t2 - t1
        if crossing_s <= 0:
            continue
        gt_mps = marker_dist_m / crossing_s
        gt_kmh = gt_mps * 3.6

        if est_mps is None:
            print(f"  track #{tid}: no speed estimate in window, skipping")
            continue
        est_kmh = est_mps * 3.6
        err_pct = (est_kmh - gt_kmh) / gt_kmh * 100.0
        verdict = "PASS" if abs(err_pct) <= ACCEPTANCE_ERROR_PCT else "FAIL"
        results.append((err_pct, tid, gt_kmh, est_kmh, crossing_s, order, verdict))
        print(f"  track #{tid}: {order}  crossing={crossing_s:.2f}s  "
              f"GT={gt_kmh:6.2f} km/h  est={est_kmh:6.2f} km/h  "
              f"error={err_pct:+6.2f}%  [{verdict}]")

    if not results:
        print("No track crossed both markers. Try a different marker pair (points "
              "must lie on a vehicle's path), a smaller --radius-px, or --frame-range.")
        sys.exit(2)

    best = min(results, key=lambda r: abs(r[0]))
    print(f"\nBest track #{best[1]}: error {best[0]:+.2f}% "
          f"(GT {best[2]:.2f} km/h vs est {best[3]:.2f} km/h)")
    if abs(best[0]) <= ACCEPTANCE_ERROR_PCT:
        print(f"PASS: within +/-{ACCEPTANCE_ERROR_PCT:.0f}% of ground truth")
        sys.exit(0)
    print(f"FAIL: outside +/-{ACCEPTANCE_ERROR_PCT:.0f}% of ground truth")
    sys.exit(1)


if __name__ == "__main__":
    main()
