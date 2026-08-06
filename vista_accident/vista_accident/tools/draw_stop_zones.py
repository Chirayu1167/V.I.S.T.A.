#!/usr/bin/env python3
"""
Interactive stop-zone drawing tool.

The README calls configuring stop_zones "the single most important thing to
configure before a real demo" — without them, anomaly_stop fires at every
red light / intersection. This replaces hand-editing pixel-coordinate JSON
with click-to-draw polygons on an actual frame from your camera.

Usage:
    python -m vista_accident.tools.draw_stop_zones --source path/to/video.mp4 --output stop_zones.json
    python -m vista_accident.tools.draw_stop_zones --source rtsp://... --output stop_zones.json

Controls:
    Left click   - add a point to the current polygon
    Enter        - close the current polygon and start a new one
    u            - undo the last point
    z            - delete the last completed polygon
    s            - save to --output and continue
    q / Esc      - save and quit

Output JSON (--output) is a plain list of polygons, directly usable with
`demo.py --stop-zones-json stop_zones.json` or `CameraConfig(stop_zones=...)`.
"""

import argparse
import json

import cv2


class ZoneDrawer:
    def __init__(self, frame):
        self.base_frame = frame
        self.polygons = []       # list of completed polygons: [[(x,y), ...], ...]
        self.current = []        # points of the in-progress polygon

    def on_mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current.append((x, y))

    def close_current(self):
        if len(self.current) >= 3:
            self.polygons.append(self.current)
        elif self.current:
            print(f"[draw_stop_zones] Ignored polygon with only {len(self.current)} point(s) — need >= 3.")
        self.current = []

    def undo_point(self):
        if self.current:
            self.current.pop()

    def undo_polygon(self):
        if self.polygons:
            removed = self.polygons.pop()
            print(f"[draw_stop_zones] Removed last polygon ({len(removed)} points).")

    def render(self):
        frame = self.base_frame.copy()
        overlay = frame.copy()
        for poly in self.polygons:
            pts = [tuple(map(int, p)) for p in poly]
            cv2.fillPoly(overlay, [_np_array(pts)], (0, 200, 0))
            cv2.polylines(frame, [_np_array(pts)], True, (0, 220, 0), 2)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        for p in self.current:
            cv2.circle(frame, tuple(map(int, p)), 4, (0, 165, 255), -1)
        if len(self.current) >= 2:
            pts = [tuple(map(int, p)) for p in self.current]
            cv2.polylines(frame, [_np_array(pts)], False, (0, 165, 255), 2)

        hud = ("LClick: add point | Enter: close polygon | u: undo point | "
               "z: undo polygon | s: save | q/Esc: save+quit")
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 26), (20, 20, 20), -1)
        cv2.putText(frame, hud, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{len(self.polygons)} zone(s) saved-ready", (8, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return frame

    def to_json(self):
        return [[[float(x), float(y)] for x, y in poly] for poly in self.polygons]


def _np_array(pts):
    import numpy as np
    return np.array(pts, dtype=np.int32)


def grab_reference_frame(source: str, seek_frac: float = 0.1):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {source}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * seek_frac))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read a frame from source: {source}")
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Video file path or RTSP URL to grab a reference frame from")
    ap.add_argument("--output", default="stop_zones.json")
    args = ap.parse_args()

    frame = grab_reference_frame(args.source)
    drawer = ZoneDrawer(frame)

    win = "VISTA - Draw Stop Zones"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, drawer.on_mouse)

    while True:
        cv2.imshow(win, drawer.render())
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10):       # Enter
            drawer.close_current()
        elif key == ord("u"):
            drawer.undo_point()
        elif key == ord("z"):
            drawer.undo_polygon()
        elif key == ord("s"):
            _save(drawer, args.output)
        elif key in (27, ord("q")):  # Esc / q
            drawer.close_current()
            _save(drawer, args.output)
            break

    cv2.destroyAllWindows()


def _save(drawer: ZoneDrawer, path: str):
    with open(path, "w") as f:
        json.dump(drawer.to_json(), f, indent=2)
    print(f"[draw_stop_zones] Saved {len(drawer.polygons)} zone(s) -> {path}")


if __name__ == "__main__":
    main()
