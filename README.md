<div align="center">

# 🚨 VISTA

**Vision-based Incident Surveillance for Traffic & Assistance**

An AI system that watches live CCTV feeds in real time, detects road accidents and violence/road rage as they happen, and instantly routes an alert — with clip, location, and timestamp — to the right authority: **traffic police**, **hospital/EMS**, or **police control room**.

**Accident detection: implemented & validated** &nbsp;•&nbsp; **Violence detection: on the roadmap**

</div>

---

## 📌 The Problem

Existing AI-CCTV deployments in India (iRASTE in Telangana, Safe Kerala, Delhi ITMS) are **rule-violation systems** — they catch helmet violations, signal jumps, and speeding to issue e-challans. **None of them detect an actual incident happening** — a crash, a fight — and trigger an emergency response.

VISTA fills that gap: not *"did someone break a rule"* but *"does someone need help right now."*

- **~0.5–1 s** time-to-alert from the moment of impact
- **4 independent accident signals**, each verified across consecutive frames to cut false positives
- **Single-vehicle crashes** (into walls, medians) caught via abrupt speed collapse — not just two-car collisions
- **Dynamic severity** per event → routes to the right emergency channels automatically

---

## 🧠 Approach: Why Dynamics, Not Single-Frame Classification

A single-frame accident classifier (e.g., YOLO fine-tuned on crash images) cannot reliably detect the *moment* of impact — it only recognizes aftermath (crashed cars, debris) and fails on unseen camera angles. VISTA instead analyzes **vehicle dynamics across time**:

| Approach | Detects moment of impact? | Works on any camera? | Latency |
|---|---|---|---|
| Single-frame accident classifier | ❌ No (aftermath only) | ❌ Angle-dependent | Low |
| 3D-CNN video model | ✅ Yes | Needs fine-tuning | High (frame buffer) |
| **Tracking + heuristics (VISTA)** | **✅ Yes** | **✅ Yes** | **~0.5–1 s** |

---

## 🏗️ Architecture

```
Live video (CCTV / RTSP / file)
        │
        ▼
┌───────────────────────────────┐
│ YOLO11m — object detection    │  vehicle/person COCO classes, FP16 on GPU
└──────────────┬────────────────┘
               │ bounding boxes
               ▼
┌───────────────────────────────┐
│ ByteTrack — multi-object      │  stable IDs across frames (<1 ms)
│ tracking (IoU + Kalman)       │
└──────────────┬────────────────┘
               │ {track_id: [(x, y, t), ...]}
               ▼
┌───────────────────────────────┐
│ MlSpeedEstimator              │  ground-point → homography/world coords,
│                               │  least-squares velocity + Kalman smoothing
└──────────────┬────────────────┘
               ▼
┌───────────────────────────────┐
│ 4 heuristic signals / frame   │
│  speed_drop • collision       │
│  anomaly_stop • hit_and_run   │
└──────────────┬────────────────┘
               ▼
┌───────────────────────────────┐
│ Verification — N consecutive  │  per-kind windows (collision=3)
│ frames + spatial dedup        │  + cooldown, tracker-ID-churn handling
└──────────────┬────────────────┘
               ▼
┌───────────────────────────────┐
│ Incident fusion — one alert   │  merges related events (collision + speed
│ per physical crash            │  drops at one spot) into ONE incident
└──────────────┬────────────────┘
               ▼
┌───────────────────────────────┐
│ Severity assessment (0–1)     │  dynamic per event (speed, IoU, pedestrians)
│                               │  → channel routing
└──────────────┬────────────────┘
               ▼
┌───────────────────────────────┐
│ Optional YOLO11x secondary    │  confirm flagged impact frames with a
│ confirmation                  │  dedicated accident model (pluggable)
└──────────────┬────────────────┘
               ▼
┌───────────────────────────────┐
│ Multi-channel dispatch        │  traffic police • hospital/EMS •
│ (mocked webhooks + async log) │  police control room → alerts.jsonl
└───────────────────────────────┘
```

> **Design rule:** the future violence branch runs **fully independently** (own model, own verification, own dispatch, true concurrency via threads/CUDA streams) so neither branch ever waits on the other.

---

## ✨ Features

| | Feature | Status |
|---|---|---|
| 🚗 | **4-signal accident detection** (speed_drop, collision, anomaly_stop, hit_and_run) | ✅ Implemented |
| 🚦 | **Signal & traffic-jam suppression** — stopped queues at red lights don't alert (world-space + pixel-space) | ✅ Implemented |
| 🧍 | **Hit-and-run detection** — vehicle–pedestrian intersection + pedestrian velocity crash + vehicle flees | ✅ Implemented |
| 🎯 | **Per-kind verification** across consecutive frames + spatial dedup for tracker ID churn | ✅ Implemented |
| 🔗 | **Incident fusion** — one alert per physical crash, most severe signal wins | ✅ Implemented |
| 🏥 | **Dynamic severity scoring** (0.0–1.0) with automatic channel routing | ✅ Implemented |
| 📏 | **Real-world speed estimation** — homography calibration, least-squares fit, per-track Kalman filter | ✅ Implemented |
| ✅ | **Optional YOLO11x secondary confirmation** on flagged impact frames | ✅ Implemented |
| 🖥️ | **Desktop GUI (PyQt5)** — upload video, live annotated preview, per-track km/h, report cards + impact screenshots | ✅ Implemented |
| 📺 | **CLI demo runner** — annotated output video + JSONL alert dashboard log | ✅ Implemented |
| 📊 | **Async dashboard logging** — background thread, never blocks the alert path | ✅ Implemented |
| ⚡ | **GPU FP16 inference** (RTX 3050: ~66 fps yolo11m) with automatic CPU fallback | ✅ Implemented |
| ⚔️ | **Violence / road-rage detection** (YOLOv8-nano fight detector) | 🔜 Roadmap |
| 🔮 | **Near-miss / TTC (time-to-collision) prediction** | 🔜 Roadmap |

---

## 🧩 Detection Criteria (the four signals)

All thresholds live in `vista_accident/vista_accident/config.py` (`HeuristicConfig`) and are tuned for ~1280×720, 15–30 fps CCTV feeds.

### 1. `speed_drop` — sudden velocity collapse
A vehicle that was moving meaningfully and loses most of its speed in a fraction of a second.

| Criterion | Threshold |
|---|---|
| Prior speed (windowed avg, 0.5 s) | ≥ 2.0 m/s |
| Velocity drop ratio (prior vs. current windowed avg) | > 0.8 |
| Suppressed in | stop zones, stationary queues |

### 2. `collision` — two vehicles overlap with an impact signature
Bounding-box overlap **plus** at least one vehicle was moving fast before the overlap and its speed collapsed at the moment of contact (a windowed pre-impact measurement — impact frames are excluded so they can't hide the deceleration).

| Criterion | Threshold |
|---|---|
| IoU between two tracked vehicles | ≥ 0.45 |
| Pre-impact speed (window `[t−1.2s, t−0.2s]`) | ≥ 3.5 m/s |
| At impact: stopped (`≤ 0.75 m/s`) or lost | > 65% of pre-impact speed |

### 3. `anomaly_stop` — stopped mid-road where it shouldn't be
A vehicle parked/stopped on the carriageway, **not** at a known stop zone and **not** part of a stationary queue.

| Criterion | Threshold |
|---|---|
| Stationary (≤ 0.5 m/s) for | ≥ 2.0 s |
| Prior motion before the stop | ≥ 1.5 m/s (excludes parked/creeping cars) |
| Suppressed when | 3+ stationary vehicles within 2.5 m (world) / 90 px (pixel) — red light / jam |
| Suppressed in | configured stop zones (intersections, bus stops) |

### 4. `hit_and_run` — vehicle strikes a pedestrian and keeps going
| Criterion | Threshold |
|---|---|
| Vehicle–person box IoU | ≥ 0.15 |
| Pedestrian velocity crash | > 90% drop |
| Vehicle keeps moving after | ≥ 1.0 m/s |

**Fusion priority** (which signal names the incident): `hit_and_run > collision > speed_drop > anomaly_stop`.

---

## 🏥 Severity Scoring & Routing

Each confirmed event is scored **dynamically** (0.0–1.0) from pre-impact speed, overlap depth, deceleration magnitude, pedestrian proximity, and post-impact behavior.

| Score | Severity | Routed to |
|---|---|---|
| 0.00 – 0.35 | `low` | 🚔 Traffic police |
| 0.35 – 0.60 | `medium` | 🚔 Traffic police |
| 0.60 – 0.85 | `high` | 🚔 Traffic police + 🚑 Hospital/EMS |
| 0.85 – 1.00 | `critical` | 🚔 Traffic police + 🚑 Hospital/EMS + 🚓 Police control room |

A minor fender-bender only reaches traffic police; a high-speed crash with pedestrians nearby escalates to EMS and police control automatically.

---

## 🧮 Speed Estimation

- **Ground-point projection** — each box's bottom-center is mapped to real-world meters via a calibrated homography (`tools/calibrate_camera.py`; falls back to a flat `meter_per_pixel` scale)
- **Least-squares velocity fit** over a rolling window (robust to a single noisy detection)
- **Per-track constant-velocity Kalman filter** — innovation-gate re-initializes on hard stops so crash decelerations aren't smoothed away; feeds `instantaneous_velocity` used by the heuristics
- Per-track **km/h overlay** in the GUI/CLI (homography when calibrated, else the `meter_per_pixel` pixel-scale fallback)

---

## 📦 Installation

**Python 3.10 (CUDA 11.8) recommended** — `torch 2.5.1+cu118`, `ultralytics`, `supervision`, `PyQt5`.

```bash
pip install -r requirements.txt
# GUI only:
pip install PyQt5
```

First run downloads `yolo11m.pt` (~39 MB) automatically from the Ultralytics release.

| Model | Size | Role |
|---|---|---|
| `yolo11m.pt` | 39 MB | Primary detector (COCO: car, truck, bus, motorcycle, person, bicycle) — chosen after benchmarking (+79% more detections/frame than yolov8n on small CCTV objects, ~66 fps RTX 3050 FP16) |
| YOLO11x accident detector *(optional)* | — | Secondary confirmation on flagged impact frames (`Enos-123/traffic-accident-detection-yolo11x` from Hugging Face) |
| YOLOv8-nano fight detector *(roadmap)* | — | Violence branch (`Musawer14/fight_detection_yolov8`) |

---

## 🚀 Usage

### CLI demo

```bash
# GPU (annotated video + per-track km/h labels + alert panel)
python demo.py --source path/to/video.mp4 --output out.mp4

# CPU
python demo.py --source path/to/video.mp4 --output out.mp4 --device cpu

# RTSP live feed — cv2 treats URLs identically to files
python demo.py --source rtsp://camera-ip:554/stream --output out.mp4
```

| Flag | Purpose |
|---|---|
| `--camera-profile profiles/CAM-01.json` | Camera calibration profile (homography, stop zones, camera metadata) — real-world speeds without editing config |
| `--px-per-meter 18.5` | Manual calibration for km/h (else auto-estimated from box width) |
| `--stop-zones-json zones.json` | Polygons where stopping is legal (intersections/bus stops) |
| `--alert-display-seconds 4` | On-screen alert panel lifetime |
| `--secondary-weights model.pt` | Enable YOLO11x secondary confirmation |
| `--max-frames N` | Limit frames (debug) |

Outputs: `out.mp4` (annotated) + `alerts.jsonl` (dashboard log, one JSON record per dispatched alert).

### Desktop GUI

```bash
python gui_app.py
```

- Upload a video → live annotated preview (per-track boxes + km/h)
- Device selector (cpu/cuda) + px/m calibration spinbox
- **Camera Profile** selector — pick a saved calibration profile (or browse for a JSON) so analysis runs with real-world speeds
- **Calibrate…** button — full GUI homography flow: load a frame, click 4+ ground points, enter real-world meters, preview the bird's-eye view, save a profile JSON (reuses `tools/calibrate_camera.py` math)
- Right panel: per-alert **report cards** — severity strip, kind, tracks, dispatch channels
- Clickable **impact screenshots** (before / impact / after), saved to `vista_screenshots/`
- Pipeline runs in a background `QThread` — the UI never freezes

### Camera calibration workflow

Speeds are only as real as the calibration. The pipeline ships a complete
workflow — CLI or GUI — that replaces the flat `meter_per_pixel` guess with a
per-camera ground-plane homography:

```bash
# 1. CLI: click 4+ ground-plane reference points on a real frame
python vista_accident/tools/calibrate_camera.py --video path/to/clip.mp4
#    (or headless: --points '[[px,py,wx,wy],...]')

# 2. GUI: same flow in the desktop app (Controls -> Calibrate…),
#    with a bird's-eye preview and a Save Profile button
python gui_app.py

# 3. Run with the saved profile — no config edits needed
python demo.py --source clip.mp4 --camera-profile camera_profiles/CAM-01.json
```

Reference points must lie on the ground plane (road surface), be spread across
the region where speeds matter, and have known real-world distances (measured
distances preferred; the standard Indian lane width of 3.5 m is the documented
fallback assumption — record it in the profile's `calibration_note`).

A per-camera on-site field measurement (one known distance taped on the road,
then entered into the calibration dialog) is a **one-time future task**: when
it becomes available, replace the `homography_src_points`/`homography_dst_points`
and update the note in the profile JSON — no code changes needed.

Every profile is a plain JSON file (`camera_profiles/<id>.json`) holding the
homography source/destination points, `meter_per_pixel` fallback, stop zones,
camera id/location/GPS, and a `calibration_note`. `camera_profile.py` loads it
into the same `CameraConfig` the pipeline already takes, so nothing else
changes.

**Speed validation (acceptance gate):** given a clip + profile + two on-road
marker points, measure a vehicle's estimated speed against ground truth from
crossing time:

```bash
python -m vista_accident.tools.validate_speed \
    --source clip.mp4 \
    --profile camera_profiles/CAM-01.json \
    --markers '[[x1,y1],[x2,y2]]'
```

Reports estimated vs. ground-truth km/h and the error % (gate: ±15%).
Use `--marker-distance-m` to override the marker distance with a directly
measured value.

**Validated end-to-end (2026-08-05):**
- **Synthetic clip** — the team clips have no measured ground-plane reference,
  so `tools/make_synthetic_clip.py` renders one with EXACT geometry (two 3.5 m
  lanes, cars at exactly 50 and 35 km/h, real car crops re-rendered so YOLO
  tracks them). Calibrated with `camera_profiles/CAM-SYNTH.json`, validated
  with 16 m marker pairs: **+0.36% error** (GT 49.99 km/h vs est 50.17 km/h).
  Regenerate with `python -m vista_accident.tools.make_synthetic_clip`.
- **Real clip (clip_03, ASSUMED calibration)** — no team measurement was
  available, so `camera_profiles/CAM-03.json` uses documented assumptions
  (see its `calibration_note`): car bbox width = 1.8 m over 58 front/rear
  detections → linear fit px/m = 0.3346·(y − 66.9), r² = 0.975 → camera
  height 2.99 m, horizon y = 66.9, focal 500 px, cx = 320 (one-point
  perspective). Cross-checked against clip_06's lane-line gaps: median
  3.24–3.36 m per lane, within ~5% of the IRC 3.5 m standard. Validation
  markers on a car's actual path (13.23 m from the homography): **−5.79%
  error** (GT 19.57 km/h vs est 18.44 km/h) — passes the ±15% gate.
- `validate_speed.py` interpolates crossing times at the path's perpendicular
  foot on each marker, so the ±15% gate is immune to marker-radius bias.

### Tests (no video/GPU needed)

```bash
python test_scenario.py    # synthetic head-on collision + pedestrian strike → asserts pipeline wiring
python test_accuracy.py    # wall-crash speed drop, junction-queue suppression, hard stop, collision
```

---

## 📂 Project Structure

```
vista_accident/
├── demo.py                    # CLI runner (video/RTSP → annotated out + alert log)
├── gui_app.py                 # PyQt5 desktop application
├── test_scenario.py           # synthetic pipeline test (head-on collision)
├── test_accuracy.py           # tests for the 4-signal accuracy fixes
├── requirements.txt
├── tools/calibrate_camera.py  # homography calibration for speed estimation
└── vista_accident/
    ├── config.py              # HeuristicConfig / CameraConfig / DispatchConfig (all thresholds)
    ├── detector.py            # YOLO11m wrapper (FP16 GPU, CPU fallback, COCO class filter)
    ├── tracker.py             # ByteTrack via supervision
    ├── track_history.py       # rolling per-track buffers + velocity/IoU/stationary helpers
    ├── speed_estimator.py     # homography world coords, LSQ velocity, Kalman filter
    ├── heuristics.py          # the 4 signals (speed_drop, collision, anomaly_stop, hit_and_run)
    ├── verification.py        # per-kind confirm windows, cooldown, spatial dedup
    ├── fusion.py              # merges related events → one alert per crash
    ├── severity.py            # dynamic 0–1 scoring + channel routing
    ├── confirmation.py        # optional YOLO11x secondary confirmation
    ├── alert.py               # alert packaging + multi-channel mock dispatch + async logging
    ├── render.py              # shared overlay (boxes, km/h, severity panel)
    └── pipeline.py            # AccidentPipeline.process_frame() — wires everything
```

---

## ⚙️ Configuration

Everything is tunable in `config.py`:

- **`HeuristicConfig`** — all four signal thresholds, verification windows, cooldowns, jam suppression
- **`CameraConfig`** — camera id/location/GPS, stop zones, homography calibration, speed-estimator settings
- **`DispatchConfig`** — mock webhook endpoints, dashboard log path

```python
from vista_accident import AccidentPipeline, HeuristicConfig

pipeline = AccidentPipeline(
    heuristic_cfg=HeuristicConfig(),
    camera_cfg=...,   # CameraConfig(camera_id="CAM-01", stop_zones=[...])
    dispatch_cfg=..., # DispatchConfig(dashboard_log_path="alerts.jsonl")
    fps_hint=25.0,
)
result = pipeline.process_frame(frame, t)   # per frame: tracks, events, alerts, speeds
```

---

## 📊 Validation

| Test | What it verifies |
|---|---|
| `test_scenario.py` | Heuristics → verification → fusion → dispatch wiring; collision + speed_drop fuse into **one** alert |
| `test_accuracy.py` | Wall-crash → `speed_drop`; 3-car signal queue → **no** alert; hard-stop → `speed_drop`; head-on collision → `collision` dispatches |
| `10 real clips (Video/)` | Confirmed detections on collisions and hard stops; zero false alerts on signal queues / normal braking |
| `tools/validate_speed.py` | Calibrated speed within ±15% of crossing-time ground truth (acceptance gate for camera calibration) |

**Accuracy fixes (2026-08-02)** — signal-queue suppression (pixel-space fallback so it works uncalibrated), anomaly_stop prior-motion requirement, windowed velocity-drop tuning.

**Calibration (2026-08-04)** — camera profiles (`camera_profiles/*.json`) with a real homography load via `--camera-profile` (CLI) or the GUI profile picker; GUI Calibrate… flow for click-and-measure calibration; `tools/validate_speed.py` for the crossing-time acceptance gate; heuristic thresholds re-audited (unchanged — they were already physical m/s values).

---

## ⚠️ Notes & Disclaimer

- **Alert dispatch is mocked** (`mock://` webhook endpoints) — swap `DispatchConfig` URLs for real Telegram/Slack/government APIs in production.
- **Violence branch is on the roadmap** — architecture is designed for it to run independently and concurrently.

---

## 🏆 Acknowledgements

Built for a hackathon by **Chirayu Mahajan** and **Farhan Farooqui**.

Project references: [suzzzal/smart-cctv-ai](https://github.com/suzzzal/smart-cctv-ai) (multimodal AI CCTV webhook routing).
