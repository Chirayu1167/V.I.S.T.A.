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

Result quality gates (added 2026-08-08, "auto-calibration instability"):
  The naive fit was USED BLINDLY and turned out to be dangerously unstable
  on real footage — the same camera's clips fitted with scales ranging from
  0.01x to 17x the flat 0.05 m/px baseline (clip_05 "cars standing still",
  clip_01 "cars at 67 km/h"), which both broke heuristic thresholds (missed
  real crashes) and injected new false positives. A good r2 on the width
  fit is NOT enough: systematic sample bias (occlusions, side-on boxes,
  narrow row bands, frame-edge crops) fits cleanly while being garbage.

  So a calibration is only accepted if ALL of these hold:
    1. Sample hygiene: frontal/rear aspect-ratio band, away from frame
       edges, below an estimated horizon, spread over a wide row band.
    2. Independent cross-check: the SAME detections' bbox heights must
       imply a plausible car height (1.0-2.2 m) under the fitted scale —
       a second, independent scale measurement from the same boxes.
    3. Validation-split stability: fit on half the samples, predict the
       other half; reject if the held-out error blows up.
  Otherwise auto_calibrate_video() returns None and the pipeline keeps the
  flat meter_per_pixel fallback (the scale every heuristic threshold was
  tuned against), exactly as before this feature existed.

The result is a CameraConfig carrying a fitted homography. The GUI / demo
use it automatically whenever a video is uploaded without an explicit
camera profile or px/m value — no Calibrate button involved. Fails
gracefully (returns None) when the scene has too few usable car
detections (e.g. a fight clip) or when the gates above reject the fit.
"""

import math

import cv2
import numpy as np

from .config import COCO_CAR, CameraConfig

# Same calibration-note assumptions as camera_profiles/CAM-03.json.
CAR_WIDTH_M = 1.8            # YOLO car bbox width (frontal/rear views)
CONF_MIN = 0.5               # detection confidence floor
ASPECT_RATIO_MIN = 0.85      # reject near-square boxes (tall vans, angles)
ASPECT_RATIO_MAX = 2.0       # reject side-on boxes (width != car width)
MIN_WIDTH_PX = 20.0          # tiny distant boxes are noise
EDGE_MARGIN_PX = 10.0        # reject boxes touching the frame border (partial crops)
MIN_SAMPLES = 20             # below this the fit is not trustworthy
R2_MIN = 0.60                # quality gate on the width-vs-row fit
HFOV_DEG = 65.0              # assumed horizontal FOV (pinhole f estimate)
ROBUST_ITERATIONS = 3        # outlier-trimming passes on the line fit
SIGMA_K = 2.5                # residual cutoff (in std dev) per pass
CAMERA_HEIGHT_RANGE_M = (1.0, 20.0)
# Independent cross-check: under the fitted scale, the SAME boxes' heights
# must imply a plausible car height. COCO "car" boxes (frontal/rear) are
# ~1.4-1.7 m tall body, up to ~2.0 m incl. mirrors/roof racks; anything
# outside this band means the fitted scale is wrong even with a good r2.
CAR_HEIGHT_RANGE_M = (1.0, 2.2)
# Validation-split stability: fit on one half, predict the other. Reject if
# the held-out median absolute error exceeds this fraction.
VALIDATION_MAX_ERR = 0.30
# Minimum vertical span (fraction of frame height) the kept width samples
# must cover — a fit over one narrow row band has a junk slope no matter
# how well it fits.
MIN_ROW_SPAN_FRAC = 0.25


def _sample_frames(cap, max_samples=40):
    """Frame indices spread across the whole video (evenly spaced)."""
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        return [0]
    n = min(n, 5000)  # guard against bogus frame counts from some encoders
    step = max(1, n // max_samples)
    return list(range(0, n, step))


def _collect_car_samples(source_path, detector):
    """(ground_row, width_px, height_px) samples from car detections."""
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
            if w < MIN_WIDTH_PX or h <= 0:
                continue
            if w / h < ASPECT_RATIO_MIN or w / h > ASPECT_RATIO_MAX:
                continue  # side-on or square-ish boxes don't model car width
            # Partial crops (frame edge) give bogus widths/heights.
            if (x1 < EDGE_MARGIN_PX or y1 < EDGE_MARGIN_PX
                    or x2 > frame_w - EDGE_MARGIN_PX or y2 > frame_h - EDGE_MARGIN_PX):
                continue
            samples.append((y2, w, h))  # ground row = bbox bottom
    cap.release()
    return samples, (frame_w, frame_h)


def _fit_line(ys, ws):
    """Least-squares slope/intercept + r2, guarding degenerate inputs."""
    n = len(ys)
    if n < 2:
        return None
    m, b = np.polyfit(ys, ws, 1)
    pred = m * ys + b
    ss_res = float(((ws - pred) ** 2).sum())
    ss_tot = float(((ws - ws.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return m, b, r2


def _fit_perspective(samples, frame_shape):
    """Robust line fit width_px = m*y + b over (y, width) samples.

    Returns (y_h, H, m, b, r2, n, kept) or None if the fit is unusable:
        px_per_m(y) = a*(y - y_h),  a = m / CAR_WIDTH_M,  H = 1/a.
    """
    frame_h = frame_shape[0]
    ys0 = np.array([s[0] for s in samples], dtype=np.float64)
    ws0 = np.array([s[1] for s in samples], dtype=np.float64)
    hs0 = np.array([s[2] for s in samples], dtype=np.float64)

    keep = np.ones(len(ys0), dtype=bool)
    fit = None
    for _ in range(ROBUST_ITERATIONS):
        ys, ws = ys0[keep], ws0[keep]
        if len(ys) < MIN_SAMPLES:
            return None
        fit = _fit_line(ys, ws)
        if fit is None:
            return None
        m, b, _r2 = fit
        resid = ws - (m * ys + b)
        sigma = float(resid.std())
        if sigma <= 0:
            break
        keep = keep.copy()
        keep[keep] = np.abs(resid) <= SIGMA_K * sigma

    ys, ws = ys0[keep], ws0[keep]
    n = len(ys)
    if n < MIN_SAMPLES:
        return None
    m, b, r2 = _fit_line(ys, ws)
    if m is None:
        return None
    if r2 < R2_MIN or m <= 0:
        return None

    y_h = -b / m
    H = CAR_WIDTH_M / m  # since slope m = CAR_WIDTH_M / camera_height
    if y_h >= 0.9 * frame_h or y_h < 0.0 or not (CAMERA_HEIGHT_RANGE_M[0] <= H <= CAMERA_HEIGHT_RANGE_M[1]):
        return None

    # Row-band coverage: a fit anchored on a thin strip has a junk slope.
    row_span = ys.max() - ys.min()
    if row_span < MIN_ROW_SPAN_FRAC * frame_h:
        return None

    # Independent cross-check via bbox HEIGHT: the same boxes' heights must
    # imply a plausible car height under the fitted scale. This catches
    # systematic sample bias that r2 cannot see.
    heights_m = [h_px * H / (y - y_h) for y, h_px in zip(ys, hs0[keep])
                 if y - y_h > 0]
    if len(heights_m) >= 8:
        median_h = float(np.median(heights_m))
        if not (CAR_HEIGHT_RANGE_M[0] <= median_h <= CAR_HEIGHT_RANGE_M[1]):
            return None
    else:
        return None  # too few height samples to cross-check

    # Validation-split stability: fit on half the samples, predict the
    # other half; reject if held-out median relative error is too large.
    idx = np.random.RandomState(42).permutation(len(ys))
    half = len(ys) // 2
    fit_ys, fit_ws = ys[idx[:half]], ws[idx[:half]]
    val_ys, val_ws = ys[idx[half:]], ws[idx[half:]]
    vfit = _fit_line(fit_ys, fit_ws)
    if vfit is None:
        return None
    vm, vb, _vr2 = vfit
    pred = vm * val_ys + vb
    med_err = float(np.median(np.abs(val_ws - pred) / np.maximum(val_ws, 1e-6)))
    if med_err > VALIDATION_MAX_ERR:
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
    detections OR the fitted scale fails the quality gates (unstable /
    implausible) — the pipeline then keeps its flat-scale default.
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
    heights_m = [s[2] * H / (s[0] - y_h) for s in samples if s[0] - y_h > 0]
    cfg.calibration_note = (
        f"AUTO-Calibrated from {n} car detections (r2={r2:.3f}, "
        f"median implied car height={float(np.median(heights_m)):.2f}m): "
        f"horizon y={y_h:.1f}px, camera height={H:.2f}m, "
        f"focal={((frame_w / 2.0) / math.tan(math.radians(HFOV_DEG / 2.0))):.0f}px, "
        f"HFOV={HFOV_DEG:.0f}deg assumed (reuse of the CAM-03 profile model). "
        f"Car bbox width = {CAR_WIDTH_M} m assumption per IRC; scale sanity "
        f"cross-checked via bbox height ({CAR_HEIGHT_RANGE_M[0]:.1f}-"
        f"{CAR_HEIGHT_RANGE_M[1]:.1f} m band). "
        f"Redo with a real field measurement before production use."
    )
    return cfg
