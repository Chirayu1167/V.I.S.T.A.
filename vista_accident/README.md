# VISTA — Accident Detection Pipeline

Working implementation of the accident branch from `VISTA.md`:
**YOLO11m detection → ByteTrack tracking → 4 heuristic signals → per-branch
verification → optional YOLO11x secondary confirmation → multi-channel mock
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
- `out.mp4` — annotated video (track boxes + red ALERT banner on confirmed events)
- `alerts.jsonl` — the dashboard log; one JSON record per dispatched alert

## Validate the logic without a real video

`test_scenario.py` scripts two cars driving head-on into each other plus a
struck pedestrian, feeds those synthetic detections straight through the
full pipeline (bypassing YOLO), and asserts that `speed_drop` and `collision`
both fire and get dispatched with correct severity/routing:

```bash
python test_scenario.py
```

Use this as your first check after changing any heuristic threshold in
`vista_accident/config.py` — it's fast (no GPU/video decode) and pins down
exactly what a signal should and shouldn't fire on.

## Module map

| File | Responsibility |
|---|---|
| `config.py` | All tunable thresholds (`HeuristicConfig`), camera metadata (`CameraConfig`), mock endpoints (`DispatchConfig`) |
| `detector.py` | YOLOv8n wrapper, FP16 on CUDA, restricted to vehicle/person COCO classes |
| `tracker.py` | ByteTrack wrapper (via `supervision`) |
| `track_history.py` | Rolling per-track-id position buffer + velocity/IoU/stationary-duration helpers — the shared data every heuristic reads |
| `heuristics.py` | The 4 signals: `speed_drop`, `collision`, `anomaly_stop`, `hit_and_run` |
| `verification.py` | Requires a signal to persist across N consecutive frames before confirming, plus per-event cooldown to stop repeat-firing |
| `confirmation.py` | Optional secondary YOLO11x confirmation on flagged frames only (pluggable — pipeline runs fine without it) |
| `alert.py` | Severity scoring (no-injury vs injury-flagged), multi-channel mock dispatch, **async** dashboard logging (background thread, never blocks the alert path) |
| `pipeline.py` | `AccidentPipeline.process_frame()` — wires the above into one call per frame |

## Severity → routing (tune against real incident data before relying on this)

| Event kind | Severity | Channels |
|---|---|---|
| `hit_and_run` | high | traffic police + hospital/EMS + police control room |
| `collision` | high | traffic police + hospital/EMS + police control room |
| `speed_drop` (no confirmed collision) | medium | traffic police |
| `anomaly_stop` | low | traffic police |

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

- **Pixel-based velocity thresholds** in `config.py` are tuned for a rough
  ~1280x720 CCTV angle. They need calibrating per camera height/angle, or
  better, converting to real-world speed via a homography/calibration step —
  not implemented here (documented as future work, not hidden).
- **`stop_zones`** (legitimate stopping areas like intersections) default to
  empty — until you draw actual zone polygons per camera, `anomaly_stop` will
  false-positive at every red light. This is the single most important thing
  to configure before a real demo.
- **Secondary YOLO11x confirmation** (`Enos-123/traffic-accident-detection-yolo11x`
  on HuggingFace) is wired as a pluggable optional step but ships disabled —
  download the weights yourself and pass `--secondary-weights` to enable it.
  Report both numbers to judges: alerts from heuristics alone vs. alerts that
  also survived secondary confirmation.
- **Dispatch is mocked** (`config.DispatchConfig` URLs are `mock://...`) —
  swap `AlertDispatcher._send_mock` for a real `requests.post(url, json=...)`
  once you have real webhook/SMS/API endpoints.
