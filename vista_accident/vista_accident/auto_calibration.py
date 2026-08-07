"""
Automatic camera calibration from the video itself (no manual clicking).

Runs the detector over sampled frames, collects CAR detections, and fits
the same one-point-perspective ground-plane model that
camera_profiles/CAM-03.json documents:

    width_px(y) = car_width_m * px_per_m(y),   px_per_m(y) = a * (y - y_h)

where y_h is the horizon row and a = 1 / camera_height. A car's YOLO bbox
width is ~1.8 m for frontal/rear views, so each detection contributes one
(width, ground-row) sample and a least-squares line fit recovers both the
horizon and the per-row scale. The along-road scale follows from a pinhole
model with an assumed ~65 deg horizontal FOV (the same assumption the
team's manual CAM-03 profile uses, cross-checked within ~5% there).

The result is a CameraConfig carrying a fitted homography. The GUI / demo
use it automatically whenever a video is uploaded without an explicit
camera profile or px/m value ΓÇö no Calibrate button involved. Fails
gracefully (returns None) when the scene has too few usable car
detections (e.g. a fight clip), in which case the pipeline keeps its
flat meter_per_pixel fallback exactly as before.
"""

import math

import cv2
import numpy as np

from .config import COCO_CAR, CameraConfig

# Same calibration-note assumptions as camera_profiles/CAM-03.json.
CAR_WIDTH_M = 1.8            # YOLO car bbox width (frontal/rear views)
CONF_MIN = 0.5               # detection confidence floor
MAX_W_H_RATIO = 2.0          # reject side-on boxes (width != car width)
MIN_WIDTH_PX = 15.0          # tiny distant boxes are noise
MIN_SAMPLES = 12             # below this the fit is not trustworthy
R2_MIN = 0.60                # quality gate on the width-vs-row fit
HFOV_DEG = 65.0              # assumed horizontal FOV (pinhole f estimate)
ROBUST_ITERATIONS = 3        # outlier-trimming passes on the line fit
SIGMA_K = 2.5                # residual cutoff (in std dev) per pass
CAMERA_HEIGHT_RANGE_M = (1.0, 20.0)


def _sample_frames(cap, max_samples=40):
    """Frame indices spread across the whole video (evenly spaced)."""
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        return [0]
    n = min(n, 5000)  # guard against bogus frame counts from some encoders
    step = max(1, n // max_samples)
    return list(range(0, n, step))


def _collect_car_samples(source_path, detector):
    """(ground_row, width_px) samples from car detections across the video."""
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        return [], None
    samples = []
    frame_w = frame_h = None
    for idx in _sample_frames(cap):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        if frame_w is None:
            frame_h, frame_w = frame.shape[:2]
        for bbox, conf, cls in detector.detect(frame):
            if cls != COCO_CAR or conf < CONF_MIN:
                continue
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            if w < MIN_WIDTH_PX or h <= 0 or w / h > MAX_W_H_RATIO:
                continue
            samples.append((y2, w))  # ground row = bbox bottom, width in px
    cap.release()
    return samples, (frame_w, frame_h)


def _fit_perspective(samples, frame_shape):
    """Robust line fit width_px = m*y + b over (y, width) samples.

    Returns (y_h, H, m, b, r2, n) or None if the fit is unusable:
        px_per_m(y) = a*(y - y_h),  a = m / CAR_WIDTH_M,  H = 1/a.
    """
    ys = np.array([s[0] for s in samples], dtype=np.float64)
    ws = np.array([s[1] for s in samples], dtype=np.float64)
    frame_h = frame_shape[0]

    keep = np.ones(len(ys), dtype=bool)
    m = b = 0.0
    for _ in range(ROBUST_ITERATIONS):
        yy, ww = ys[keep], ws[keep]
        if len(yy) < MIN_SAMPLES:
            return None
        m, b = np.polyfit(yy, ww, 1)
        resid = ww - (m * yy + b)
        sigma = float(resid.std())
        if sigma <= 0:
            break
        keep = keep.copy()
        keep[keep] = np.abs(resid) <= SIGMA_K * sigma

    yy, ww = ys[keep], ws[keep]
    n = len(yy)
    if n < MIN_SAMPLES:
        return None
    m, b = np.polyfit(yy, ww, 1)
    ss_res = float(((ww - (m * yy + b)) ** 2).sum())
    ss_tot = float(((ww - ww.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if r2 < R2_MIN or m <= 0:
        return None
    y_h = -b / m
    H = CAR_WIDTH_M / m  # since slope m = CAR_WIDTH_M / camera_height
    if y_h >= 0.9 * frame_h or not (CAMERA_HEIGHT_RANGE_M[0] <= H <= CAMERA_HEIGHT_RANGE_M[1]):
        return None
    return y_h, H, m, b, r2, n


def _build_homography(frame_w, frame_h, y_h, H):
    """4+ pixel points -> world meters via the one-point-perspective model:
        world_x = (x - cx) * H / (y - y_h)
        world_y = f * H / (y - y_h)        (f from the assumed HFOV)
    Same formulas that reproduce CAM-03.json's fitted dst points."""
    cx = frame_w / 2.0
    f = (frame_w / 2.0) / math.tan(math.radians(HFOV_DEG / 2.0))

    margin = max(5.0, 0.05 * (frame_h - y_h))
    y_far = y_h + 0.20 * (frame_h - y_h) + margin
    y_near = y_h + 0.90 * (frame_h - y_h)
    cols = [0.2 * frame_w, 0.5 * frame_w, 0.8 * frame_w]

    src, dst = [], []
    for y in (y_far, y_near):
        for x in cols:
            denom = y - y_h
            if denom <= 1e-6:
                continue
            src.append((float(x), float(y)))
            dst.append((float((x - cx) * H / denom), float(f * H / denom)))
    if len(src) < 4:
        return None
    return src, dst


def auto_calibrate_video(source_path, detector, camera_id="CAM-AUTO") -> "CameraConfig | None":
    """Fit a homography-based CameraConfig from the video's own car traffic.

    Returns a CameraConfig with homography_src_points/dst_points + a
    calibration_note, or None when the scene has too few usable car
    detections (pipeline then keeps its flat-scale default).
    """
    samples, frame_shape = _collect_car_samples(source_path, detector)
    if frame_shape is None or not samples:
        return None
    fit = _fit_perspective(samples, frame_shape)
    if fit is None:
        return None
    y_h, H, m, b, r2, n = fit
    frame_w, frame_h = frame_shape

    homography = _build_homography(frame_w, frame_h, y_h, H)
    if homography is None:
        return None
    src, dst = homography

    cfg = CameraConfig(
        camera_id=camera_id,
        homography_src_points=[list(p) for p in src],
        homography_dst_points=[list(p) for p in dst],
    )
    cfg.calibration_note = (
        f"AUTO-Calibrated from {n} car detections (r2={r2:.3f}): "
        f"horizon y={y_h:.1f}px, camera height={H:.2f}m, "
        f"focal={((frame_w / 2.0) / math.tan(math.radians(HFOV_DEG / 2.0))):.0f}px, "
        f"HFOV={HFOV_DEG:.0f}deg assumed (reuse of the CAM-03 profile model). "
        f"Car bbox width = {CAR_WIDTH_M} m assumption per IRC. "
        f"Redo with a real field measurement before production use."
    )
    return cfg
