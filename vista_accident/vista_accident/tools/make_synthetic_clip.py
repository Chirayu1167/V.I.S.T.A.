"""
Synthetic CCTV clip with EXACT ground-truth geometry, for speed-validation.

Why this exists
---------------
The team's real demo clips (Video/clip_01..10) have no measured ground-plane
reference, and speed validation needs a known-distance marker pair + a car
with a known true speed. This tool renders a camera view of a synthetic road
where every length is exact by construction:

    - two lanes, each 3.5 m wide (IRC standard lane width)
    - a one-point-perspective camera (focal fx=520/fy=440 px, horizon at
      y=120, depth offset z0=8 m on a 1280x720 frame) — the same projective
      model the homography calibration recovers
    - cars are real YOLO-detected crops from the team clips, re-rendered as
      ground-plane stickers (world footprint 4.0 m long x 2.0 m tall), so
      the detector still sees and tracks them
    - car A drives straight along the near lane at exactly 13.9 m/s
      (50 km/h); car B in the far lane at 9.72 m/s (35 km/h)
The road markings are drawn in world coordinates and projected through the
camera homography, so calibrating the generated clip (clicking lane corners)
reproduces the exact world scale — the resulting camera profile is the
reference homography for tools/validate_speed.py.

Usage:
    python -m vista_accident.tools.make_synthetic_clip --output calib_demo.mp4

Reference geometry (printed on each run, used by the profile + validation):
    - lane edges at world y = 0.25 / 3.75 / 7.25 m (lane width 3.5 m)
    - car A lane center: world y = 2.0 m; car B: y = 5.5 m
    - car A path: x in [-11, +11] m at 13.9 m/s
    - validation marker pair: (x=-8, y=2.0) <-> (x=+8, y=2.0), 16 m apart
"""

import argparse
import os

import cv2
import numpy as np

# --- camera model (one-point perspective, 1280x720) --------------------------
IMG_W, IMG_H = 1280, 720
FX, FY = 520.0, 440.0
CX, YH = IMG_W / 2.0, 120.0
Z0 = 8.0  # meters in front of camera where the near road plane sits

# --- world geometry ----------------------------------------------------------
LANE_W = 3.5          # IRC standard lane width, meters
CAR_LEN_WORLD = 4.0   # sticker world footprint along the road, meters
CAR_A_SPEED = 13.9    # m/s == 50 km/h
CAR_B_SPEED = 9.72    # m/s == 35 km/h
LANE_A_Y = 2.0        # near lane center (world y)
LANE_B_Y = 5.5        # far lane center (world y)

# world map (top-down canvas) extent, meters; s = meters per map pixel
MAP_X0, MAP_X1 = -32.0, 32.0
MAP_Y0, MAP_Y1 = -2.0, 80.0
MAP_S = 0.05

FPS = 30
FRAMES = 150

SPRITE_DIR = r"C:\Users\PCWORL~1\AppData\Local\Temp\opencode\sprites2"


# --- homographies ------------------------------------------------------------
def H_world_to_image():
    """World (x, y, 1) -> image (xi, yi, w): xi = cx + fx*x/(y+z0),
    yi = yh + fy*z0/(y+z0). World y = distance from the camera, world x =
    distance along the road."""
    return np.array([
        [FX, CX, CX * Z0],
        [0.0, YH, (YH + FY) * Z0],
        [0.0, 1.0, Z0],
    ], dtype=np.float64)


def project_world_to_image(points, H_w2i):
    """(N,2) world pts -> (N,2) image pts."""
    pts = np.array(points, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H_w2i)
    return out.reshape(-1, 2)


def build_world_map():
    """Top-down road canvas at MAP_S m/px. Road runs along world x."""
    mw = int(round((MAP_X1 - MAP_X0) / MAP_S))
    mh = int(round((MAP_Y1 - MAP_Y0) / MAP_S))
    rng = np.random.default_rng(7)

    map_img = np.zeros((mh, mw, 3), dtype=np.uint8)

    # grass beyond the road
    grass = rng.normal(105, 8, (mh, mw, 1)).clip(0, 255).astype(np.uint8)
    map_img[:] = np.concatenate([grass, grass + 14, grass - 22], axis=2).clip(0, 255)

    def world_line(yw, x0, x1, color, width_m):
        p0 = (int((x0 - MAP_X0) / MAP_S), int((yw - MAP_Y0) / MAP_S))
        p1 = (int((x1 - MAP_X0) / MAP_S), int((yw - MAP_Y0) / MAP_S))
        cv2.line(map_img, p0, p1, color, max(1, int(round(width_m / MAP_S))))

    # asphalt: road from world y 0 to 80, world x -23.5..23.5
    ay0 = int((0.0 - MAP_Y0) / MAP_S)
    ay1 = int((80.0 - MAP_Y0) / MAP_S)
    ax0 = int((-23.5 - MAP_X0) / MAP_S)
    ax1 = int((23.5 - MAP_X0) / MAP_S)
    asphalt = rng.normal(78, 5, (max(1, ay1 - ay0), max(1, ax1 - ax0), 3)).clip(0, 255).astype(np.uint8)
    map_img[ay0:ay1, ax0:ax1] = asphalt

    # edge lines at y = 0.25 / 7.25, center dashes at y = 3.75
    for ey in (0.25, 7.25):
        world_line(ey, -23.5, 23.5, (235, 235, 235), 0.15)
    for dx in np.arange(-23.5, 23.5 - 2.5, 5.0):
        world_line(3.75, dx, dx + 2.5, (225, 225, 225), 0.12)

    return map_img


def draw_car(frame, sprite, xc, yc):
    """Ground-plane sticker: sprite bottom-center placed at the projected
    ground point of (xc, yc), width = CAR_LEN_WORLD projected at that depth."""
    h, w = sprite.shape[:2]
    denom = yc + Z0
    xi = CX + FX * xc / denom
    yi = YH + FY * Z0 / denom
    width_px = FX * CAR_LEN_WORLD / denom
    height_px = width_px * h / w

    # shadow ellipse on the ground under the car
    cv2.ellipse(frame, (int(xi), int(yi)), (int(width_px * 0.55), int(width_px * 0.12)),
                0, 0, 360, (40, 40, 40), -1)

    scaled = cv2.resize(sprite, (max(2, int(width_px)), max(2, int(height_px))),
                        interpolation=cv2.INTER_LINEAR)
    x0 = int(xi - scaled.shape[1] / 2)
    y0 = int(yi - scaled.shape[0])
    x0 = max(0, min(x0, IMG_W - scaled.shape[1]))
    y0 = max(0, min(y0, IMG_H - scaled.shape[0]))
    frame[y0:y0 + scaled.shape[0], x0:x0 + scaled.shape[1]] = scaled
    return frame


def build_scene():
    H_w2i = H_world_to_image()
    world_map = build_world_map()
    mh, mw = world_map.shape[:2]
    H_map2w = np.array([
        [1.0 / MAP_S, 0.0, -MAP_X0 / MAP_S],
        [0.0, 1.0 / MAP_S, -MAP_Y0 / MAP_S],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    H_view = H_w2i @ H_map2w

    # sky gradient painted first; the map warp (BORDER_TRANSPARENT) leaves it
    # in rows above the road's horizon projection
    bg = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    horizon_row = int(YH)
    for row in range(horizon_row):
        bg[row] = np.array([int(150 - row * 0.35), int(168 - row * 0.35), int(190 - row * 0.35)])
    warped = cv2.warpPerspective(world_map, H_view, (IMG_W, IMG_H),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    mask = warped != 0
    bg[mask] = warped[mask]

    sprites = []
    for name in sorted(os.listdir(SPRITE_DIR)):
        if name.endswith(".png"):
            sprites.append(cv2.imread(os.path.join(SPRITE_DIR, name)))
    if len(sprites) < 2:
        raise SystemExit("Need at least 2 sprites (run extract_sprites.py first)")

    return bg, H_w2i, sprites


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default="calib_demo.mp4")
    ap.add_argument("--frames", type=int, default=FRAMES)
    args = ap.parse_args()

    bg, H_w2i, sprites = build_scene()
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (IMG_W, IMG_H))
    rng = np.random.default_rng(11)

    for f in range(args.frames):
        t = f / FPS
        frame = bg.copy()

        xa = -11.0 + CAR_A_SPEED * max(0.0, t - 0.5)
        if -11.0 < xa < 11.0:
            frame = draw_car(frame, sprites[0], xa, LANE_A_Y)

        xb = 11.0 - CAR_B_SPEED * max(0.0, t - 1.0)
        if -11.0 < xb < 11.0:
            frame = draw_car(frame, sprites[1 if len(sprites) > 1 else 0], xb, LANE_B_Y)

        noise = rng.normal(0, 2.5, frame.shape).astype(np.int16)
        frame = (frame.astype(np.int16) + noise).clip(0, 255).astype(np.uint8)
        writer.write(frame)

    writer.release()
    print(f"Wrote {args.frames} frames -> {args.output}")

    # print the exact reference geometry for the camera profile
    corners = []
    for y in (0.5, 7.0):
        for x in (-10.0, 10.0):
            corners.append((x, y))
    img_pts = project_world_to_image(corners, H_w2i)
    print("\nReference lane corners (world m -> image px) for the profile:")
    for (xw, yw), (xi, yi) in zip(corners, img_pts, strict=True):
        print(f"  world=({xw:6.2f},{yw:6.2f})  image=({xi:7.2f},{yi:7.2f})")

    m1 = project_world_to_image([(-8.0, LANE_A_Y)], H_w2i)[0]
    m2 = project_world_to_image([(8.0, LANE_A_Y)], H_w2i)[0]
    print(f"\nCar A: lane y={LANE_A_Y} m, speed={CAR_A_SPEED} m/s ({CAR_A_SPEED*3.6:.1f} km/h)")
    print(f"Marker pair (16 m apart): pixel {tuple(round(v,1) for v in m1)} "
          f"<-> {tuple(round(v,1) for v in m2)}")


if __name__ == "__main__":
    main()
