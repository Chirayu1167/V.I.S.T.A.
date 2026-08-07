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
- `recipients.json` — nearby hospitals/police stations used by the
  control-room console for nearest-recipient routing display
- `acks.jsonl` — operator acknowledgements from the control-room console

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
| `config.py` | All tunable thresholds (`HeuristicConfig`), camera metadata (`CameraConfig`), mock endpoints + rate limiting + signing (`DispatchConfig`), live threshold/stop-zone hot-reload (`ConfigWatcher`) |
| `detector.py` | YOLO11m wrapper, FP16 on CUDA, restricted to vehicle/person COCO classes, brightness-gated CLAHE enhancement for low-light/night frames |
| `tracker.py` | ByteTrack wrapper (via `supervision`) |
| `track_history.py` | Rolling per-track-id position buffer + velocity/IoU/stationary-duration helpers — the shared data every heuristic reads. `active_ids()` defaults to ids seen on the current frame only (not stale/occluded tracks still sitting in the buffer) |
| `heuristics.py` | The 4 signals: `speed_drop`, `collision`, `anomaly_stop`, `hit_and_run` |
| `verification.py` | Requires a signal to persist across N consecutive frames before confirming, plus per-event cooldown to stop repeat-firing |
| `confirmation.py` | Optional secondary YOLO11x confirmation on flagged frames only (pluggable — pipeline runs fine without it) |
| `plate_reader.py` | Optional license-plate OCR (`easyocr`), run only on flagged `hit_and_run` frames |
| `alert.py` | Severity scoring (no-injury vs injury-flagged), multi-channel mock dispatch (HMAC-signed), burst rate limiting, **async** dashboard logging (background thread, never blocks the alert path — call `.close()`/`pipeline.close()` to flush it on shutdown) |
| `pipeline.py` | `AccidentPipeline.process_frame()` — wires the above into one call per frame; also owns clip export |
| `camera_profile.py` | Camera calibration profiles — save/load a `CameraConfig` (homography points, `meter_per_pixel` fallback, stop zones, camera metadata) as JSON; `world_distance_m()` for known-distance marker math |
| `tools/calibrate_camera.py` | Interactive + headless homography calibration (click 4+ ground-plane points with known real-world coords → config snippet + bird's-eye preview) |
| `tools/validate_speed.py` | Speed acceptance gate: run a clip with a profile + two on-road markers, compare estimator speed vs crossing-time ground truth, report error % (±15% gate) |
| `tools/draw_stop_zones.py` | Click-to-draw stop-zone polygons on a real frame from your camera → JSON for `--stop-zones-json` |
| `tools/dashboard.py` | Control-room console over `alerts.jsonl` + `vista_clips/` — WebAudio siren, severity banner, clip playback, nearest-recipient routing (`recipients.json`), operator ACK (`acks.jsonl`). Stdlib-only. Works with the merged multi-camera log too |
| `tools/review_alerts.py` | Walk through logged alerts, label true/false positive, get a false-positive-rate-by-kind summary to actually justify threshold changes |
| `multi_camera.py` | Runs one `AccidentPipeline` per camera concurrently, all alerts merged into one shared log/dashboard |

## Severity → routing (tune against real incident data before relying on this)

| Event kind | Severity | Channels |
|---|---|---|
| `hit_and_run` | high | traffic police + hospital/EMS + police control room |
| `collision` | high | traffic police + hospital/EMS + police control room |
| `speed_drop` (no confirmed collision) | medium | traffic police |
| `anomaly_stop` | low | traffic police |

## Additional features

**Clip export.** Pass `--clip-dir clips/` to `demo.py` (or `clip_dir=` to
`AccidentPipeline`) to save a short pre/post-impact `.mp4` per dispatched
alert instead of just a single impact frame. Export is scheduled at
confirmation time and written a couple seconds later once the rolling
buffer actually has the post-impact frames — dispatch itself is never
delayed for it.

**Stop-zone drawing tool.** `python -m vista_accident.tools.draw_stop_zones
--source video.mp4 --output stop_zones.json` opens a frame from your
camera and lets you click-draw polygons instead of hand-editing pixel
coordinates. Feed the result straight into `--stop-zones-json`.

**Control-room console.** `python -m vista_accident.tools.dashboard --log
alerts.jsonl --clips vista_clips` serves a big-screen dispatch console —
stdlib only, no extra dependency. For every NEW dispatched alert it sounds
a WebAudio siren (click ARM SIREN first — browsers block autoplay), flashes
a severity-colored banner (kind / severity / location / coordinates /
timestamp / camera / plate), auto-plays the per-alert clip once the pipeline
finishes writing it (~1.5 s after dispatch), shows the nearest
hospitals/police stations the alert is routed to (from `recipients.json`,
matched to the severity's channels), and lets the operator ACK each alert
(`acks.jsonl`). This is the dispatch-side display of the two-interface demo:
the GUI (`gui_app.py`) detects, the console receives. Works unmodified with
the merged log from `multi_camera.py` since every `AlertPayload` already
carries its own `camera_id`.

**Multi-camera.** `python multi_camera.py --config cameras.json --log
alerts.jsonl` runs one `AccidentPipeline` per camera concurrently (same
pattern as the accident+violence concurrency below, applied across
cameras) and merges every camera's alerts into one shared log/dashboard.

**License-plate OCR.** `--enable-plate-ocr` (requires `pip install
easyocr`) runs OCR on the vehicle crop for `hit_and_run` events only —
the one event kind where "the vehicle kept moving" makes identifying it
the actual point of the alert. Degrades to a no-op if `easyocr` isn't
installed rather than becoming a hard dependency.

**Night / low-light handling.** `detector.py` checks each frame's mean
brightness and applies CLAHE contrast enhancement (on the LAB L-channel)
before detection when it's below `LOW_LIGHT_BRIGHTNESS_THRESHOLD` (60/255
by default). Cheap preprocessing, not a retrained model — helps recall on
underexposed CCTV footage, doesn't fix a genuinely unlit scene.

**Config hot-reload.** `--watch-config thresholds.json` polls a JSON file
(`{"heuristics": {...}, "stop_zones": [...]}`) and applies changes to the
*live* `HeuristicConfig`/`CameraConfig` objects the running pipeline
already holds — no restart needed while tuning thresholds during a demo.
See `vista_accident.config.ConfigWatcher`.

**Alert review / feedback loop.** `python -m
vista_accident.tools.review_alerts --log alerts.jsonl` walks through
logged alerts (with full `meta`) and lets you label each true/false
positive; `--summary` then prints a false-positive rate by event kind so
threshold changes are justified by labeled data instead of guesses.

**Burst rate limiting.** `DispatchConfig.rate_limit_max_alerts` (default
4 per `rate_limit_window_s`, default 10s) bundles alerts beyond that rate
from one camera down to `traffic_police`-only routing instead of re-paging
EMS/police-control per event — covers a genuine multi-vehicle pileup
generating several distinct alerts within a few seconds. Every alert is
still logged individually; nothing is dropped, only the channel fan-out is
collapsed. Set to `0` to disable.

**Payload signing.** Every mock-dispatched payload is now HMAC-SHA256
signed (`DispatchConfig.hmac_secret`) so real endpoints can verify
authenticity/integrity from day one. Set a real secret (env var / secrets
manager — do not commit one) before swapping `_send_mock` for a real
`requests.post(url, json=asdict(payload), headers={"X-Vista-Signature": sig})`.

**Emergency Response dashboards (ML bridge).** Pass
`--emergency-response-url http://127.0.0.1:8890/api/incidents` (GUI: the
"Open Emergency Response" button does this automatically) and every
dispatched alert is POSTed to the `emergency_response/` server, where it
appears on the hospital/police/traffic police dashboards tagged **ML
AUTO-DETECTED**, routed to the nearest 3 authorities of each type via
Haversine. Same `AlertPayload -> incident` mapping as the citizen
`/report` page; fire-and-forget on a background thread, so an absent
server never affects dispatch. See `emergency_response/client.py` and its
README for the kind→incident_type table.

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

## Camera calibration (real-world speeds)

Speed readings are only as accurate as the camera calibration. The flat
`meter_per_pixel` guess in `CameraConfig` is correct only for a perfectly
top-down camera; for an angled CCTV feed you need a ground-plane homography.
Full workflow:

```bash
# CLI: click 4+ ground-plane reference points (known real-world coords)
python -m vista_accident.tools.calibrate_camera --video path/to/clip.mp4

# or the GUI flow: Controls -> Calibrate… -> load frame, click points,
# enter meters, preview bird's-eye, Save profile JSON
python gui_app.py

# run with the saved profile
python demo.py --source clip.mp4 --camera-profile camera_profiles/CAM-01.json
```

A profile (`camera_profiles/<id>.json`) is a plain JSON object that
`vista_accident.camera_profile.load_profile()` turns into the exact
`CameraConfig` the pipeline already takes — pipeline code doesn't change.
Fields: `homography_src_points` / `homography_dst_points` (4+ pixel ↔ world
correspondences), `meter_per_pixel` (fallback only), `stop_zones`, camera
id/location/lat/lon, `fps`, and a free-text `calibration_note` recording what
was measured or assumed (e.g. "lane width 3.5 m — standard Indian lane").
Prefer measured distances; if none are available, the standard Indian lane
width of 3.5 m is the sanctioned assumption — say so in the note.

A per-camera on-site field measurement (one known distance taped on the road,
entered via the calibration dialog) is a **one-time future task**: when it
arrives, swap the homography points + note in the profile JSON — no code
changes needed.

**Speed validation (acceptance gate):** with a calibrated clip, verify the
estimator against crossing time over a known-distance marker pair:

```bash
python -m vista_accident.tools.validate_speed \
    --source clip.mp4 \
    --profile camera_profiles/CAM-01.json \
    --markers '[[x1,y1],[x2,y2]]'
```

The marker distance is taken from the profile homography by default (or pass
`--marker-distance-m` for a directly measured value). The tool runs the real
detector/tracker, finds a track crossing both markers, and reports estimated
vs. ground-truth km/h plus error % (pass ≤ ±15%).

### Validation without team footage: synthetic clip

The team clips have no measured ground-plane reference yet. To validate the
homography + estimator end-to-end without inventing measurements, the repo
ships `tools/make_synthetic_clip.py`, which renders a clip with EXACT
geometry (two 3.5 m lanes, cars driven at exactly 50 km/h and 35 km/h, real
YOLO-detected car crops re-rendered so the detector still tracks them):

```bash
# regenerate the clip (calib_demo.mp4) + print the reference geometry
python -m vista_accident.tools.make_synthetic_clip --output calib_demo.mp4

# validate against the shipped profile (CAM-SYNTH.json) — markers on car A's
# path, 16 m apart, ground truth known exactly
python -m vista_accident.tools.validate_speed \
    --source calib_demo.mp4 \
    --profile camera_profiles/CAM-SYNTH.json \
    --markers "[[224.0,472.0],[1056.0,472.0]]"
```

Measured on 2026-08-05 (CPU): best track error **+0.36%** (GT 49.99 km/h vs
est 50.17 km/h) — passes the ±15% gate. `camera_profiles/CAM-SYNTH.json`
documents the scene geometry in its `calibration_note`. The clip itself is
small (~0.5 MB) but regenerate it rather than trusting a stale copy.
(Crossing times are interpolated at the path's perpendicular foot on each
marker, so the gate is independent of `--radius-px`.)

### Real clip with assumed calibration (clip_03)

Without a team measurement, `camera_profiles/CAM-03.json` calibrates
Video/clip_03.mp4 from documented assumptions (full details in its
`calibration_note`): YOLO car bbox width = 1.8 m (58 front/rear detections,
linear fit px/m = 0.3346·(y − 66.9), r² = 0.975) → camera height 2.99 m,
horizon y = 66.9, focal 500 px, cx = 320 (one-point perspective). The scale
was cross-checked on clip_06, where it converts lane-line gaps to median
3.24–3.36 m (IRC lane width 3.5 m, within ~5%). Validation markers on a
car's actual path (13.23 m from the homography):

```bash
python -m vista_accident.tools.validate_speed \
    --source ..\Video\clip_03.mp4 \
    --profile camera_profiles/CAM-03.json \
    --markers "[[558.0,300.0],[583.0,150.0]]"
```

Measured 2026-08-05 (CPU): **−5.79%** (GT 19.57 km/h vs est 18.44 km/h) —
passes the ±15% gate. This is an ASSUMED calibration: redo with a real
measured ground-plane distance before production use.

## Known simplifications (be upfront about these with judges)

- **Pixel-based velocity thresholds** in `config.py` are tuned for a rough
  ~1280x720 CCTV angle, in m/s, and calibrated against `meter_per_pixel`
  (or a homography, if configured). If you run without the ML speed
  estimator (`use_ml_speed=False`) AND without a homography, the flat
  `meter_per_pixel` scale is only correct for a perfectly top-down camera —
  significantly wrong at an angle. Run `tools/calibrate_camera.py` for a
  real per-camera homography before trusting absolute thresholds.
- **`stop_zones`** (legitimate stopping areas like intersections) default to
  empty — until you draw actual zone polygons per camera, `anomaly_stop` will
  false-positive at every red light. This is the single most important thing
  to configure before a real demo — use `tools/draw_stop_zones.py` to draw
  them by clicking on an actual frame instead of hand-editing coordinates.
- **Secondary YOLO11x confirmation** (`Enos-123/traffic-accident-detection-yolo11x`
  on HuggingFace) is wired as a pluggable optional step but ships disabled —
  download the weights yourself and pass `--secondary-weights` to enable it.
  Report both numbers to judges: alerts from heuristics alone vs. alerts that
  also survived secondary confirmation.
- **Dispatch is mocked** (`config.DispatchConfig` URLs are `mock://...`) —
  swap `AlertDispatcher._send_mock` for a real `requests.post(url, json=...)`
  once you have real webhook/SMS/API endpoints. Payloads are already
  HMAC-signed (see Additional features above); set a real `hmac_secret`
  before going live.
- **Weather / heavy occlusion is not specifically handled.** The low-light
  CLAHE preprocessing helps with underexposed night footage, but rain,
  fog, glare, and heavy multi-vehicle occlusion are not compensated for —
  detection recall will drop in those conditions same as any COCO-pretrained
  detector. Flag this as future work rather than a solved problem.
- **Repo ships three checkpoint files** (`yolo11m.pt`, `yolov8n.pt`,
  `yolo26n.pt`) but only `yolo11m.pt` is ever loaded by `detector.py`. The
  other two are stale — delete them unless you're specifically A/B testing
  detector backbones, they just bloat the deliverable.
- **`multi_camera.py`'s shared JSONL log** uses one independent file handle
  per camera thread (see its docstring) — fine at demo alert volumes, not a
  guarantee of atomicity under heavy concurrent write load in a real
  multi-camera deployment.
