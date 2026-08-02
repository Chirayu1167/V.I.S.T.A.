# VISTA — Accident Detection Pipeline

Working implementation of the accident branch from `VISTA.md`:
**YOLO11m detection → ByteTrack tracking → real-world speed estimation →
6 heuristic signals → per-branch verification → incident fusion → dynamic
severity → optional YOLO11x secondary confirmation → multi-channel mock
dispatch**, with an async-logged dashboard feed.

## Install

```bash
pip install -r requirements.txt
```

Tested with `ultralytics==8.4.107`, `supervision==0.29.1`. First run downloads
`yolo11m.pt` (~39MB) automatically from the Ultralytics GitHub release.
`yolo11m` was picked over `yolov8n` after benchmarking on demo footage
(+79% more detections/frame for small/distant vehicles, ~66 fps on RTX 3050).

## Quick start

```bash
# On a video file (or swap --source for an rtsp:// URL — cv2 treats them identically)
python demo.py --source path/to/video.mp4 --output out.mp4

# Force CPU (no GPU in this environment)
python demo.py --source path/to/video.mp4 --device cpu
```

This writes:
- `out.mp4` — annotated video (track boxes + per-track km/h + severity-colored alert panel)
- `alerts.jsonl` — the dashboard log; one JSON record per dispatched alert

Optional flags: `--px-per-meter 18.5` (manual calibration for km/h),
`--stop-zones-json zones.json` (polygons where stopping is legal),
`--secondary-weights model.pt` (enable YOLO11x confirmation),
`--alert-display-seconds 4`.

There is also a **PyQt5 desktop GUI** (`gui_app.py`) — upload a video, watch
the live annotated preview, and get per-alert report cards with clickable
impact screenshots (`vista_screenshots/`).

## The six detection signals

All thresholds live in `vista_accident/vista_accident/config.py`
(`HeuristicConfig`). See the root `README.md` for the full criterion tables.

| Signal | What it detects | Key criteria |
|---|---|---|
| `speed_drop` | Sudden velocity collapse | prior ≥ 1.2 m/s, ratio > 0.65, or ends near-stopped (≤ 1.0 m/s with ratio > 0.5) |
| `collision` | Two vehicles overlap with impact signature | IoU ≥ 0.45, pre-impact speed ≥ 3.5 m/s, speed collapse > 65% at impact |
| `anomaly_stop` | Stopped mid-road, not at a stop zone / queue | stationary ≥ 2 s, prior motion ≥ 1.5 m/s; suppressed in jams (3+ within 2.5 m / 90 px) |
| `hit_and_run` | Vehicle strikes pedestrian and keeps going | IoU ≥ 0.15, pedestrian velocity crash > 90%, vehicle continues ≥ 1.0 m/s |
| `jerk` | Impact shock — single-vehicle crashes (wall/median) | peak deceleration > 8 m/s² in 0.3 s window, prior speed ≥ 2.0 m/s, ends near-stopped |
| `smoke` | Dust/smoke cloud after hard impact (pure CV) | growing gray low-texture blob, 900–40000 px², growth > 1.25× |

## Validate the logic without a real video

`test_scenario.py` scripts two cars driving head-on into each other plus a
struck pedestrian, feeds those synthetic detections straight through the
full pipeline (bypassing YOLO), and asserts that `collision` fires and that
the collision + post-crash speed drops fuse into **one** dispatched alert:

```bash
python test_scenario.py
```

`test_accuracy.py` covers the accuracy pass (2026-08-02): growing dust cloud →
`smoke`, single-vehicle wall crash → `jerk`, 3-car signal queue → no alert,
70% velocity drop → `speed_drop`:

```bash
python test_accuracy.py
```

Use these as your first check after changing any heuristic threshold in
`config.py` — they're fast (no GPU/video decode) and pin down exactly what a
signal should and shouldn't fire on.

## Module map

| File | Responsibility |
|---|---|
| `config.py` | All tunable thresholds (`HeuristicConfig`), camera metadata (`CameraConfig`), mock endpoints (`DispatchConfig`) |
| `detector.py` | YOLO11m wrapper, FP16 on CUDA, restricted to vehicle/person COCO classes, CPU fallback |
| `tracker.py` | ByteTrack wrapper (via `supervision`) |
| `track_history.py` | Rolling per-track-id position buffer + velocity/IoU/stationary-duration/deceleration helpers — the shared data every heuristic reads |
| `speed_estimator.py` | Ground-point → homography world coords, least-squares velocity fit, per-track Kalman filter with innovation gate |
| `heuristics.py` | The 6 signals: `speed_drop`, `collision`, `anomaly_stop`, `hit_and_run`, `jerk` (+ jam suppression) |
| `smoke_detector.py` | CV-only smoke/dust cloud detector (no extra ML model) |
| `verification.py` | Requires a signal to persist across N consecutive frames (per-kind windows) before confirming, plus cooldown + spatial dedup for tracker ID churn |
| `fusion.py` | `IncidentFuser` — collapses related events (collision + smoke + speed_drops at one spot) into ONE alert per crash; most severe kind wins |
| `severity.py` | Dynamic 0–1 severity scoring per event (speed, IoU, decel, smoke, pedestrians) → channel routing |
| `confirmation.py` | Optional secondary YOLO11x confirmation on flagged impact frames only (pluggable — pipeline runs fine without it) |
| `alert.py` | Severity-based multi-channel mock dispatch, **async** dashboard logging (background thread, never blocks the alert path) |
| `render.py` | Shared overlay: track boxes, per-track km/h, severity-colored alert panel (used by demo + GUI) |
| `pipeline.py` | `AccidentPipeline.process_frame()` — wires the above into one call per frame |

## Severity → routing (tune against real incident data before relying on this)

Severity is scored **dynamically** per event (0.0–1.0), not from a static map:

| Score | Severity | Channels |
|---|---|---|
| 0.00 – 0.35 | `low` | traffic police |
| 0.35 – 0.60 | `medium` | traffic police |
| 0.60 – 0.85 | `high` | traffic police + hospital/EMS |
| 0.85 – 1.00 | `critical` | traffic police + hospital/EMS + police control room |

Smoke evidence on a fused incident (collision + dust cloud) boosts the score.

## Wiring this into the full VISTA system (accident + violence, concurrent)

Per `VISTA.md`'s key design decision, the violence branch must run
**independently and concurrently** — not after this one. Minimal pattern:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=2)

async def run_frame(frame, t):
    loop = asyncio.get_running_loop()
    accident_task = loop.run_in_executor(executor, accident_pipeline.process_frame, frame, t)
    violence_task = loop.run_in_executor(executor, violence_pipeline.process_frame, frame, t)
    accident_result, violence_result = await asyncio.gather(accident_task, violence_task)
    return accident_result, violence_result
```

Each `AlertDispatcher` instance already logs asynchronously on its own
background thread, so neither branch's dispatch path blocks the other or the
main frame loop.

## Known simplifications (be upfront about these with judges)

- **Camera calibration is deferred** — the demo clips run on a guessed
  `0.05 m/px` scale. Use `tools/calibrate_camera.py` to generate a homography
  (`homography_src_points`/`homography_dst_points` in `CameraConfig`) for
  accurate speeds; note that re-calibrating changes every speed, so the m/s
  heuristic thresholds must be retuned alongside.
- **`stop_zones`** (legitimate stopping areas like intersections) default to
  empty — until you draw actual zone polygons per camera, `anomaly_stop`
  relies on the jam-suppression fallback (3+ stationary vehicles). This is
  the single most important thing to configure before a real demo.
- **Secondary YOLO11x confirmation** (`Enos-123/traffic-accident-detection-yolo11x`
  on HuggingFace) is wired as a pluggable optional step but ships disabled —
  download the weights yourself and pass `--secondary-weights` to enable it.
  Report both numbers to judges: alerts from heuristics alone vs. alerts that
  also survived secondary confirmation.
- **Dispatch is mocked** (`config.DispatchConfig` URLs are `mock://...`) —
  swap `AlertDispatcher._send_mock` for a real `requests.post(url, json=...)`
  once you have real webhook/SMS/API endpoints.
