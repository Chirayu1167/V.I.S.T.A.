# VISTA — Vision-based Incident Surveillance for Traffic & Assistance

## The Idea

An AI system that monitors live CCTV feeds to automatically detect road accidents and violence/road rage in real time, then instantly routes an alert — with clip, location, and timestamp — to the relevant authority (traffic police, hospital/EMS, or police control room).

**Why this matters:** existing AI-CCTV deployments in India (iRASTE in Telangana, Safe Kerala, Delhi ITMS) are rule-violation systems — they catch helmet violations, signal jumps, and speeding for fines (e-challans). None of them detect an actual *incident happening* — a crash, a fight — and trigger an emergency response. This system fills that gap: not "did someone break a rule" but "does someone need help right now."

---

## Final Workflow

```
Live video (CCTV/RTSP)
        │
        ├──────────────────────────────┐
        ▼                               ▼
 Accident Detector              Violence Detector
   (runs independently)          (runs independently)
        │                               │
        ▼                               ▼
 Accident Verification          Violence Verification
 (confirms across a few          (confirms across a few
  consecutive frames)             consecutive frames)
        │                               │
        ▼                               ▼
 Accident Alert Dispatch        Violence Alert Dispatch
   (severity scoring:              (routes to)
   injury vs no injury)                │
        │                               │
   ┌────┴────┐                          ▼
   ▼         ▼                   Police Control Room
Traffic   Hospital /
Police       EMS
   │         │                          │
   └────┬────┴──────────────────────────┘
        ▼
  Shared Incident Dashboard
  (async logging — live feed, event log,
   dispatch status, does not block alerts)
```

**Key design decision:** the two branches (accident, violence) are fully independent pipelines — each has its own model, its own verification logic, and its own alert dispatch. Neither branch waits on the other, so a violence detection isn't delayed by accident-model inference time, and vice versa. This is the lowest-latency architecture of the options considered, at the cost of some duplicated code between branches (acceptable trade-off for a hackathon).

---

## Features

1. **Multi-camera RTSP ingestion** — pull live frames from CCTV feeds
2. **Motion prefilter (violence branch only)** — skip static/unchanging frames before running the violence model to save compute; accident branch processes every frame since the moment of impact cannot be skipped
3. **Two independent, concurrently-running detection pipelines** (accident + violence), using true concurrency (threads/async/CUDA streams) — not sequential calls — so GPU inference genuinely overlaps
4. **Per-branch verification** — each pipeline confirms an event persists across a short window of consecutive frames before triggering an alert, to cut false positives without slowing the other branch
5. **Severity scoring on the accident branch** — distinguishes no-injury accidents (→ traffic police) from injury-flagged accidents (→ hospital/EMS); both can fire together for severe cases
6. **Automated alert packaging** — clip, GPS/camera location, timestamp, confidence score bundled per alert
7. **Multi-channel dispatch** — traffic police, hospital/EMS, police control room (mock webhook/SMS/API endpoints for the hackathon demo, since real government API access isn't available)
8. **Live incident dashboard** — camera feed view, flagged event log, dispatch status tracking (sent → acknowledged → resolved); logs asynchronously so it never blocks the alert path
9. **GPU-optimized inference (FP16)** on both models for lower latency

---

## Accident Detection Pipeline (Tracking + Heuristics + ML Confirmation)

A single-frame object detection model (like YOLO11x) cannot reliably detect the *moment* of an accident — it only sees aftermath (crashed cars, debris) and fails on unseen camera angles. Instead, we detect accidents by analyzing **vehicle dynamics across time**.

### Why this approach

| Approach | Detects moment of impact? | Works on any camera? | Latency |
|---|---|---|---|
| Single-frame accident classifier | No (aftermath only) | No (angle-dependent) | Low |
| 3D CNN video model | Yes | Needs fine-tuning | High (frame buffer) |
| **Tracking + heuristics (this project)** | **Yes** | **Yes** | **~0.5-1s** |

### Pipeline (frame-by-frame)

```
CCTV frame
    │
    ▼
┌──────────────────────────────┐
│ YOLO11m (COCO pretrained)     │ ← detects vehicles + persons/cyclists
│ Per-frame, ~15ms on GPU       │   (car, truck, bus, person, bicycle, etc.)
└──────────────┬───────────────┘
               │ bounding boxes
               ▼
┌──────────────────────────────┐
│ ByteTrack                    │ ← assigns/updates track IDs; matches same
│ <1ms, no neural net          │   vehicle across consecutive frames
│ (IoU matching + Kalman filter)│
└──────────────┬───────────────┘
               │ track history: {track_id: [(x,y,t), ...]}
               ▼
┌──────────────────────────────┐
│ Heuristic checks (per frame) │ ← three independent signals
└──────────────┬───────────────┘
               │
     ┌─────────┼─────────┐
     ▼         ▼         ▼
┌────────┐ ┌────────┐ ┌──────────┐
│ Speed  │ │ Colli- │ │ Anomaly  │
│ drop   │ │ sion   │ │ stop     │
│ >80%   │ │ detect │ │ (middle  │
│ in <   │ │ (two   │ │ of road  │
│ 0.5s   │ │ tracks │ │ > 2s)    │
│        │ │ overlap│ │          │
└───┬────┘ └───┬────┘ └────┬─────┘
    │         │         │
    └────┬────┴────┬────┘
         │         │
         ▼         ▼
   ┌──────────────┐
   │ Hit-and-run  │
   │ (vehicle +   │
   │ pedestrian   │
   │ intersection │
   │ + ped stops) │
   └──────┬───────┘
                 │ any trigger?
                    ▼
┌──────────────────────────────┐
│ Verification (5-frame window)│ ← confirm accident persists across frames
│                              │   to reject false positives
└──────────────┬───────────────┘
               │ confirmed
               ▼
┌──────────────────────────────┐
│ YOLO11x accident detector    │ ← secondary ML confirmation on the
│ (HF: Enos-123)               │   flagged frame (optional confidence boost)
│ mAP@0.5: 0.826               │
│ F1 (accident): 0.833         │
└──────────────┬───────────────┘
               │ confirmed
               ▼
    ┌──── Alert Dispatch ────┐
    ▼                         ▼
Traffic Police / EMS    Police Control Room
```

### Heuristic Signals Explained

**1. Speed drop** — For each tracked vehicle, compute velocity = `displacement / Δt` (pixels/second). If velocity drops >80% within 0.5 seconds, a collision likely occurred. Catches rear-ends, T-bones, head-ons.

**2. Collision detection** — For every pair of tracked vehicles, if both stop at the same location (bounding boxes overlap with IoU > 0.3 and both have near-zero velocity), a collision between them likely occurred.

**3. Anomaly stop** — If a vehicle stops in the middle of the road (not at an intersection or roadside) for >2 seconds, it may have hit something or someone.

**4. Hit-and-run / pedestrian strike** — When a vehicle track intersects with a pedestrian/cyclist track at the same location+time, and the pedestrian/cyclist track then shows sudden velocity drop >90% (fell/stopped) while the vehicle track continues moving — a hit-and-run occurred. This catches the most dangerous scenarios that the other three signals miss.

### Models & Libraries Used

| Component | What | Source | Why |
|---|---|---|---|
| Vehicle & pedestrian detector | **YOLO11m** (COCO pretrained) | [Ultralytics YOLO11](https://huggingface.co/Ultralytics/YOLO11) — `pip install ultralytics`, model: `yolo11m.pt` | ~39M params, ~66 fps on RTX 3050 (FP16, 640x360). Picked over yolov8n after benchmarking: +79% more detections/frame (small/distant vehicles) |
| Tracker | **ByteTrack** | [github.com/ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack) — `pip install bytetrack` | State-of-the-art tracker, <1ms per frame, no GPU needed for tracking |
| Secondary confirmation | **YOLO11x accident detector** | [HF: Enos-123/traffic-accident-detection-yolo11x](https://huggingface.co/Enos-123/traffic-accident-detection-yolo11x) | Fine-tuned on surveillance accident data; mAP@0.5: 0.826, recall: 0.855 |

### How to Use

```python
# 1. Vehicle detection
from ultralytics import YOLO
detector = YOLO("yolo11m.pt")  # COCO pretrained
results = detector(frame)       # returns boxes for cars, trucks, buses

# 2. Tracking
from bytetrack import ByteTrack
tracker = ByteTrack()
tracks = tracker.update(detections)  # each track has a unique ID + history

# 3. Heuristic checks (custom logic, ~50 lines)
for track in tracks:
    if track.velocity_drop() > 0.8:  # sudden stop
        trigger_accident_alert()

# 4. Optional confirmation
if heuristic_triggered:
    accident_model = YOLO("yolo11x-accident.pt")  # from HuggingFace
    result = accident_model(flagged_frame)
    if result[0].boxes.conf.max() > 0.5:
        dispatch_alert()
```

### Violence Detection (unchanged)

| Priority | Model | Source | Why |
|---|---|---|---|
| **Primary** | YOLOv8-nano fight detector | [HF: Musawer14/fight_detection_yolov8](https://huggingface.co/Musawer14/fight_detection_yolov8) | Lowest latency, ~1-2ms per frame, purpose-built for real-time |
| **Alternative** | DenseNet121 real-time violence | [github.com/vavi39/Real-Time-Violence-Detection-in-Surveillance-Streams](https://github.com/vavi39/Real-Time-Violence-Detection-in-Surveillance-Streams) | ~30 FPS with native RTSP multi-camera support |
| **Add-on** | VIGIL.AI weapon detection | [github.com/ash-iiiiish/VIGIl.AI-Violence-WeaponDetectionTool](https://github.com/ash-iiiiish/VIGIl.AI-Violence-WeaponDetectionTool) | R3D-18 + YOLOv8, optional severity input for police branch |

---

## Deployment Reference

- **Full-stack reference architecture** (FastAPI + React + Docker): [github.com/SarathL754/vigil3d-video-inference](https://github.com/SarathL754/vigil3d-video-inference) — useful as a starting scaffold even if you swap out the model
- **Closest prior-art system** (multimodal AI monitoring CCTV, reports to authorities via webhook): [github.com/suzzzal/smart-cctv-ai](https://github.com/suzzzal/smart-cctv-ai) — study its webhook/routing pattern

---

## Known Trade-offs to Address in Your Pitch

- **"Reports to authorities" will be mocked** in the demo (no real government API access) — be upfront about this; route to a mock webhook (Telegram bot / Slack / logged REST endpoint) rather than overselling it as live integration.
- **False positive rate is the first thing technical judges will probe.** The per-branch verification layer (confirming across multiple frames before alerting) is your answer — have a concrete number ready (e.g., "X% reduction in false positives after verification, tested on Y clips").
- **True concurrency matters for the latency claim.** Running both models sequentially in one thread doesn't give real parallelism — make sure the actual implementation uses threading, multiprocessing, or async/CUDA streams so both branches genuinely overlap on the GPU.

---

## Update Log

| Date | Changes |
|---|---|
| 2026-07-26 | Replaced static accident model list with tracking + heuristics pipeline (YOLOv8n + ByteTrack + 4 heuristic signals + YOLO11x secondary confirmation). Added hit-and-run detection. Moved motion prefilter to violence branch only. |
| 2026-07-29 | Implemented accident pipeline (YOLOv8n + ByteTrack + 4 heuristics + verification + dispatch). Validated with synthetic test and real video. GPU vs CPU comparison done. GPU default + CPU fallback. |
| 2026-07-29 (later) | Replaced static kind→severity map with dynamic SeverityAssessor. Scores each event 0.0–1.0 using pre-impact speed, IoU overlap, post-impact stop, pedestrian proximity. |
| 2026-07-31 | Added desktop GUI (`gui_app.py`, PyQt5) + shared `render.py` overlay (per-track km/h via SpeedEstimator, severity-colored alert panel). Fixed false-positive heuristics: collision impact-signature, windowed speed_drop + stop zones, per-kind verification, ID-churn dedup, incident fusion (demo1.mp4: 33 → 11 alerts). |
| 2026-08-01 | Upgraded detector from yolov8n → yolo11m. Benchmark on demo footage (RTX 3050, FP16): yolov8n 2.23 dets/frame → yolo11n 2.81 → yolo12s 3.39 → yolo11s 3.84 → yolo11m 3.99 (66 fps). YOLO12 underperforms on small CCTV objects; yolo11m wins on recall with 2x realtime margin. |
| 2026-08-02 | Accuracy pass (4 fixes): (1) Smoke/dust detection — new CV-only `smoke_detector.py` (grayish low-texture mask + connected components, alerts only on a blob that persists AND grows, so fog/static bright regions don't fire); fires its own `smoke` alert (wall crashes) and boosts collision severity. (2) Stopped-car false positives — jam suppression now has a pixel-space fallback (3+ stationary within 90px) so it works without ML calibration (previously silently disabled on uncalibrated feeds, which is why signal queues alerted); `anomaly_stop` requires prior motion ≥1.5 m/s right before the stop (parked/creeping cars excluded). (3) Jerk detection — new `check_jerk` reads Kalman-smoothed speed series for deceleration spikes >8 m/s² (normal braking 2–4) with prior-motion + near-stop checks; catches single-vehicle crashes into walls/medians that collision (2-track) misses. (4) Speed-drop retuned — "after" reading now uses Kalman instantaneous velocity (windowed average was masking the drop), ratio 0.8→0.65 plus absolute near-stop branch (ratio floor 0.5 so normal stops don't alert). New `test_accuracy.py` (4 scenarios) + all 10 clips revalidated. |
| 2026-08-02 (later) | Chirayu's `8289ded "Old bug fixes + video model"` added: per-alert video clips (`--clip-dir`, pre/post-impact mp4 export + `clip_path` in payload), license-plate OCR (`plate_reader.py`, easyocr, only on hit_and_run), multi-camera runner (`multi_camera.py`, one pipeline per camera on its own thread), tools (`dashboard.py` live alerts browser, `draw_stop_zones.py` click-to-draw polygons, `review_alerts.py` true/false-positive labeling → fp-rate-by-kind), per-camera burst rate-limiting + HMAC payload signing, hot-reloadable config (`ConfigWatcher`), low-light CLAHE enhancement, and m/s→km/h fixes in severity + fusion recency fix. IMPORTANT: that commit was branched from before the smoke/jerk accuracy pass, so it accidentally REVERTED smoke/jerk support (config fields, `TrackHistory.deceleration()`, smoke detector wiring, jerk/smoke severity scorers) while leaving the newer `heuristics.py`/`smoke_detector.py` intact — this crashed the GUI with `AttributeError` on frame 1. Restored the reverted smoke/jerk pieces (keeping Chirayu's new features) and wired clip saving into the GUI (`gui_app.py`, `vista_clips/` output dir + count in summary). `test_accuracy.py` + `test_scenario.py` all PASS. |
| 2026-08-02 (even later) | One card per crash: widened fusion window 1.5s → 6.0s (configurable `fusion_window_s`/`fusion_radius_px` on `HeuristicConfig`). A crash's cascade (collision → smoke → jerk → speed_drop arriving over seconds) now fuses into ONE incident → ONE dispatch → ONE clip + one screenshot set, instead of several alert cards each with its own media. Same-spot alerts within ~6s merge into a single card (fewest files, easiest to present). |
