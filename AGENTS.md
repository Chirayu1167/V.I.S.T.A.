# VISTA — Agent Context File

## Project
VISTA — Vision-based Incident Surveillance for Traffic & Assistance.
A hackathon project: AI system monitoring live CCTV feeds to detect road accidents and violence/road rage, then routing alerts to traffic police, hospital/EMS, or police control room.

## GitHub
- **Repo**: `Chirayu1167/V.I.S.T.A.` (note the trailing dot)
- **Clone URL**: `https://github.com/Chirayu1167/V.I.S.T.A..git`
- **Default branch**: `main`
- **Local clone**: `C:\Users\User\AppData\Local\Temp\opencode\vista_repo`
- **Commits go as**: user `FarhanFarooqui122`, email `farhanfarooqui312@gmail.com`

## Git Token
Fine-grained PAT (starts with `github_pat_...`). Must have repo access configured.
- **If push fails with 403**: go to https://github.com/settings/tokens, click the token, under "Repository access" select "All repositories" or add `Chirayu1167/V.I.S.T.A.`

## Commit Flow
1. Edit `C:\Users\User\Desktop\V.I.S.T.A\VISTA.md` (local)
2. Copy to clone: `Copy-Item -Path "C:\Users\User\Desktop\V.I.S.T.A\VISTA.md" -Destination "C:\Users\User\AppData\Local\Temp\opencode\vista_repo\VISTA.md" -Force`
3. cd `C:\Users\User\AppData\Local\Temp\opencode\vista_repo`
4. `git add VISTA.md`
5. `git commit -m "message"`
6. `git push origin main`

## Tech Stack
- **OS**: Windows, NVIDIA GPU
- **Vehicle/pedestrian detection**: YOLOv8n (COCO pretrained) — `pip install ultralytics`, model `yolov8n.pt`
- **Tracker**: ByteTrack — `pip install bytetrack`
- **Accident secondary confirmation**: YOLO11x accident detector (HF: `Enos-123/traffic-accident-detection-yolo11x`)
- **Violence detection**: YOLOv8-nano fight detector (HF: `Musawer14/fight_detection_yolov8`)

## Accident Detection Approach (Tracking + Heuristics)
Single-frame YOLO cannot detect the moment of impact reliably. Instead:
1. **YOLOv8n** detects vehicles + persons/cyclists (every frame)
2. **ByteTrack** tracks each object across frames with unique IDs
3. **4 heuristic signals** checked per frame:
   - **Speed drop**: velocity drop >80% in <0.5s
   - **Collision**: two tracked vehicles overlap + both stop
   - **Anomaly stop**: vehicle stops in middle of road >2s
   - **Hit-and-run**: vehicle + pedestrian tracks intersect, ped stops, vehicle continues
4. **Verification**: confirm across 5 consecutive frames
5. **YOLO11x** optional secondary ML confirmation on flagged frames

## Violence Detection
- YOLOv8-nano fight detector (primary, low latency)
- DenseNet121 (alternative, ~30 FPS)
- VIGIL.AI weapon detection (optional add-on)

## Key Design Rules
- Accident and violence pipelines run **independently** (true concurrency with threads/async/CUDA streams)
- Motion prefilter only applies to **violence branch** (accident branch needs every frame)
- Alert dispatch is **mocked** (webhook/Telegram/Slack for demo)
- Per-branch verification across frames to cut false positives

## Update Log
Latest: 2026-07-26 — Replaced static accident model list with tracking + heuristics pipeline. Added hit-and-run detection.
