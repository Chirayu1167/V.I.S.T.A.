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
2. **Motion prefilter** — skip static/unchanging frames before running heavy models, to save compute and reduce average latency
3. **Two independent, concurrently-running detection pipelines** (accident + violence), using true concurrency (threads/async/CUDA streams) — not sequential calls — so GPU inference genuinely overlaps
4. **Per-branch verification** — each pipeline confirms an event persists across a short window of consecutive frames before triggering an alert, to cut false positives without slowing the other branch
5. **Severity scoring on the accident branch** — distinguishes no-injury accidents (→ traffic police) from injury-flagged accidents (→ hospital/EMS); both can fire together for severe cases
6. **Automated alert packaging** — clip, GPS/camera location, timestamp, confidence score bundled per alert
7. **Multi-channel dispatch** — traffic police, hospital/EMS, police control room (mock webhook/SMS/API endpoints for the hackathon demo, since real government API access isn't available)
8. **Live incident dashboard** — camera feed view, flagged event log, dispatch status tracking (sent → acknowledged → resolved); logs asynchronously so it never blocks the alert path
9. **GPU-optimized inference (FP16)** on both models for lower latency

---

## Models to Use (accuracy + latency optimized)

You have a dedicated NVIDIA GPU, so prioritize models benchmarked for real-time GPU inference over ones optimized purely for edge/CPU deployment.

### Accident Detection

| Priority | Model | Source | Why |
|---|---|---|---|
| **Primary** | CSP-YOLOv9 | [github.com/sajid6230/csp-yolov9-traffic-accident-detection](https://github.com/sajid6230/csp-yolov9-traffic-accident-detection) | Newer YOLO architecture, better parameter efficiency than YOLOv8x at comparable accuracy |
| **Alternative (newer arch, HF-hosted)** | YOLO11x accident detector | [huggingface.co/Enos-123/traffic-accident-detection-yolo11x](https://huggingface.co/Enos-123/traffic-accident-detection-yolo11x) | YOLO11 generally improves accuracy/speed trade-off over v8/v9; worth benchmarking against CSP-YOLOv9 on your own footage |
| **Speed-priority fallback** | YOLOv8s (not YOLOv8x) | [huggingface.co/Enos-123/accident-evaluator-yolov8x](https://huggingface.co/Enos-123/accident-evaluator-yolov8x) (swap to the `s` variant) | YOLOv8x is the largest variant — trades speed for accuracy you likely don't need; `s` or `m` is faster per frame |
| **Zero-shot verification signal** | ACCIDENT (CLIP + optical flow) | [github.com/sarveshtalele/ACCIDENT-CVPR_2026](https://github.com/sarveshtalele/ACCIDENT-CVPR_2026) | No training required, works on a different signal (motion + CLIP) than YOLO — good secondary check to cross-verify and further reduce false positives if latency budget allows |

### Violence / Road-Rage Detection

| Priority | Model | Source | Why |
|---|---|---|---|
| **Primary (lowest latency)** | YOLOv8-nano or YOLOv8-small fight detector | [huggingface.co/Musawer14/fight_detection_yolov8](https://huggingface.co/Musawer14/fight_detection_yolov8) | Nano/small variants are purpose-built for real-time, resource-constrained detection — very low per-frame latency on GPU |
| **Alternative (higher accuracy, still real-time)** | DenseNet121 real-time violence model | [github.com/vavi39/Real-Time-Violence-Detection-in-Surveillance-Streams](https://github.com/vavi39/Real-Time-Violence-Detection-in-Surveillance-Streams) | Benchmarked at ~30 FPS with native RTSP multi-camera support — good if nano/small accuracy isn't sufficient |
| **Road-rage specific** | 3D CNN road rage detector | [github.com/tanveer744/road-rage-detection](https://github.com/tanveer744/road-rage-detection) | Trained specifically for road rage rather than general violence — closer to your actual use case if you want to distinguish "fight" from "road rage" as separate alert types |
| **Weapon detection add-on (optional)** | VIGIL.AI | [github.com/ash-iiiiish/VIGIl.AI-Violence-WeaponDetectionTool](https://github.com/ash-iiiiish/VIGIl.AI-Violence-WeaponDetectionTool) | Adds weapon detection (R3D-18 + YOLOv8) on top of violence — could feed into severity scoring for the police-control branch |

### Recommendation for best accuracy/latency trade-off

Start with **YOLOv8-nano/small (Musawer14)** for violence and **CSP-YOLOv9 or YOLO11x** for accidents — benchmark both against a short clip set from your target camera angle before committing, since real-world accuracy on your specific footage (angle, lighting, resolution) matters more than published benchmarks. If accuracy is too low on your test clips, step up to the DenseNet121 model for violence, since it's the only one with a stated real-time throughput number you can verify.

---

## Deployment Reference

- **Full-stack reference architecture** (FastAPI + React + Docker): [github.com/SarathL754/vigil3d-video-inference](https://github.com/SarathL754/vigil3d-video-inference) — useful as a starting scaffold even if you swap out the model
- **Closest prior-art system** (multimodal AI monitoring CCTV, reports to authorities via webhook): [github.com/suzzzal/smart-cctv-ai](https://github.com/suzzzal/smart-cctv-ai) — study its webhook/routing pattern

---

## Known Trade-offs to Address in Your Pitch

- **"Reports to authorities" will be mocked** in the demo (no real government API access) — be upfront about this; route to a mock webhook (Telegram bot / Slack / logged REST endpoint) rather than overselling it as live integration.
- **False positive rate is the first thing technical judges will probe.** The per-branch verification layer (confirming across multiple frames before alerting) is your answer — have a concrete number ready (e.g., "X% reduction in false positives after verification, tested on Y clips").
- **True concurrency matters for the latency claim.** Running both models sequentially in one thread doesn't give real parallelism — make sure the actual implementation uses threading, multiprocessing, or async/CUDA streams so both branches genuinely overlap on the GPU.
