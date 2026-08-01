"""
Camera calibration helper for a FIXED camera.

Speed estimates from speed_estimator.py are only as accurate as the
homography you calibrate. This is the single biggest lever for speed
accuracy -- more so than swapping detector models. This script lets you
click 4+ points on a real frame from your camera and pair them with their
known real-world coordinates (in meters), then prints a ready-to-paste
CameraConfig snippet and shows a bird's-eye-view preview so you can sanity
check the result before trusting it.

How to pick good reference points
----------------------------------
Pick 4+ points that:
  - all lie on the ground plane (road surface), not on curbs/objects
  - you can measure or already know the real-world distance between
    (e.g. lane markings of a known standard width, a crosswalk's painted
    corners, cones/tape you placed and measured yourself)
  - are spread out across the region of the frame where you actually want
    accurate speeds (don't cluster them all in one corner)
  - more points = a more robust homography fit; 4 is the minimum, 6-8 is
    better if you have good references

Usage
-----
Interactive (needs a display):
    python tools/calibrate_camera.py --video path/to/clip.mp4

    - A frame from the video appears. Click each reference point in order.
    - After each click, enter its real-world (x, y) in meters at the
      terminal prompt -- pick any consistent origin/axes on the ground
      plane (e.g. origin at one corner of a marked lane, x = across the
      road, y = along the road).
    - Press 'q' once you've placed 4+ points to finish and see the result.

Headless (no display) -- provide points directly:
    python tools/calibrate_camera.py --video path/to/clip.mp4 \\
        --points '[[412,610,0,0],[498,610,3.5,0],[412,480,0,20],[498,480,3.5,20]]'
    (each entry is [pixel_x, pixel_y, world_x_m, world_y_m])

Either mode prints:
  1. A CameraConfig snippet to paste into config.py
  2. A bird's-eye-view preview image (calibration_preview.png) so you can
     visually confirm straight lines (e.g. lane edges) stay straight and
     parallel in the transformed view -- if they look warped or bowed, the
     points were probably picked wrong (not actually on the ground plane,
     or not accurately measured) and you should redo it.
"""

import argparse
import json
import sys

import cv2
import numpy as np


def get_frame(video_path: str, frame_index: int = 0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    if frame_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read frame {frame_index} from {video_path}")
    return frame


def collect_points_interactive(frame) -> list:
    points = []
    display = frame.copy()
    window = "Calibration - click ground-plane points, 'q' to finish (min 4)"

    def on_click(event, x, y, flags, param):
        nonlocal display
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"\nClicked pixel ({x}, {y}). Enter its real-world coords in meters.")
            try:
                wx = float(input("  world_x (meters): ").strip())
                wy = float(input("  world_y (meters): ").strip())
            except ValueError:
                print("  Invalid number, skipping this point.")
                return
            points.append((float(x), float(y), wx, wy))
            cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(display, f"#{len(points)}", (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow(window, display)

    cv2.imshow(window, display)
    cv2.setMouseCallback(window, on_click)
    print(f"Click 4+ ground-plane reference points in the window, then press 'q'.")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    return points


def build_homography(points: list):
    if len(points) < 4:
        raise SystemExit(f"Need at least 4 points, got {len(points)}.")
    src = np.array([[p[0], p[1]] for p in points], dtype=np.float32)
    dst = np.array([[p[2], p[3]] for p in points], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    if H is None:
        raise SystemExit("cv2.findHomography failed -- check your points aren't collinear/degenerate.")
    return H, src, dst


def render_birdseye_preview(frame, H, out_path: str, dst_points, margin_m: float = 5.0):
    h, w = frame.shape[:2]
    # Warp the whole frame into world space, scaled to pixels-per-meter for viewing.
    px_per_m = 20
    xs = dst_points[:, 0]
    ys = dst_points[:, 1]
    min_x, max_x = xs.min() - margin_m, xs.max() + margin_m
    min_y, max_y = ys.min() - margin_m, ys.max() + margin_m
    out_w = max(50, int((max_x - min_x) * px_per_m))
    out_h = max(50, int((max_y - min_y) * px_per_m))

    # Shift homography output so (min_x, min_y) maps to (0,0), then scale to pixels.
    shift_scale = np.array([
        [px_per_m, 0, -min_x * px_per_m],
        [0, px_per_m, -min_y * px_per_m],
        [0, 0, 1],
    ], dtype=np.float64)
    H_view = shift_scale @ H

    warped = cv2.warpPerspective(frame, H_view, (out_w, out_h))
    cv2.imwrite(out_path, warped)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="Path to a video from your fixed camera")
    ap.add_argument("--frame-index", type=int, default=0, help="Which frame to calibrate against")
    ap.add_argument("--points", default=None,
                     help="JSON list of [pixel_x, pixel_y, world_x_m, world_y_m] "
                          "for headless use instead of clicking")
    ap.add_argument("--preview-out", default="calibration_preview.png")
    args = ap.parse_args()

    frame = get_frame(args.video, args.frame_index)

    if args.points:
        raw = json.loads(args.points)
        points = [tuple(p) for p in raw]
    else:
        try:
            points = collect_points_interactive(frame)
        except cv2.error as e:
            raise SystemExit(
                f"No display available for interactive mode ({e}).\n"
                f"Re-run with --points instead, e.g.:\n"
                f'  --points \'[[x1,y1,wx1,wy1],[x2,y2,wx2,wy2],...]\''
            )

    H, src, dst = build_homography(points)

    print("\n" + "=" * 70)
    print("Paste this into CameraConfig in vista_accident/config.py:")
    print("=" * 70)
    src_list = [(round(float(x), 1), round(float(y), 1)) for x, y in src]
    dst_list = [(round(float(x), 2), round(float(y), 2)) for x, y in dst]
    print(f"    homography_src_points: list = field(default_factory=lambda: {src_list})")
    print(f"    homography_dst_points: list = field(default_factory=lambda: {dst_list})")
    print("=" * 70)

    out_path = render_birdseye_preview(frame, H, args.preview_out, dst)
    print(f"\nBird's-eye-view preview saved to: {out_path}")
    print("Check it: known-straight/parallel real-world lines (lane edges, curbs) should")
    print("look straight and parallel in this preview. If they look warped or bowed,")
    print("re-pick points -- likely one wasn't really on the ground plane, or a")
    print("real-world measurement was off.")


if __name__ == "__main__":
    main()
