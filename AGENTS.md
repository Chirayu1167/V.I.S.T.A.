# AGENTS.md — VISTA

VISTA (Vision-based Incident Surveillance for Traffic & Assistance) — real-time
CCTV AI that detects road accidents and routes alerts (clip, location, timestamp)
to traffic police / hospital-EMS / police control based on severity. Hackathon
project by Chirayu Mahajan and Farhan Farooqui.

Repo: `Chirayu1167/V.I.S.T.A.` (public). Cloned locally to `C:\Users\PC World\VISTA`
(no trailing dots — Windows forbids folder names ending in `.`).

## Language / Stack

- Python 3.10 recommended (CUDA 11.8), `torch 2.5.1+cu118`
- ultralytics (YOLO11m), supervision (ByteTrack), opencv-python, numpy
- Optional: PyQt5 (GUI), easyocr (plate OCR), torch GPU for FP16

## Run Commands

From `C:\Users\PC World\VISTA\vista_accident`:

```bash
# Tests (no video/GPU needed)
python test_scenario.py     # collision + hit-and-run fusion wiring
python test_accuracy.py     # 4-signal accuracy fixes

# CLI demo (GPU default; add --device cpu for CPU)
python demo.py --source path/to/video.mp4 --output out.mp4

# Desktop GUI
python gui_app.py

# Live dashboard for alerts.jsonl
python -m vista_accident.tools.dashboard --log alerts.jsonl --port 8787

# Calibration / stop-zone tools
python vista_accident/tools/calibrate_camera.py
python -m vista_accident.tools.draw_stop_zones --source video.mp4 --output stop_zones.json

# Multi-camera runner
python multi_camera.py --config cameras.json --log alerts.jsonl
```

First run auto-downloads `yolo11m.pt` (~39 MB).

## Architecture

Per-frame flow (`vista_accident/vista_accident/pipeline.py` → `process_frame`):

1. `Detector.detect()` — YOLO11m, FP16 GPU + CPU fallback, CLAHE low-light boost
2. `Tracker.update()` — ByteTrack via supervision (`tracker.py`)
3. `TrackHistory.update()` — rolling per-track buffers + speed estimator (`track_history.py`)
4. `run_all_heuristics()` — 4 signals (`heuristics.py`):
   - `speed_drop`: velocity drop > `speed_drop_ratio` (0.8), min prior 2.0 m/s
   - `collision`: IoU >= 0.45 + pre-impact speed >= 3.5 m/s + speed collapse at overlap
   - `anomaly_stop`: stopped >= 2 s mid-road (jam + stop-zone suppressed)
   - `hit_and_run`: vehicle-ped IoU + ped velocity crash + vehicle keeps moving
5. `Verifier.process()` — N consecutive frames per kind, cooldown, spatial dedup
6. `IncidentFuser` — merges sub-events of one crash into ONE alert
7. `SeverityAssessor` — dynamic 0–1 score → channel routing
8. `AlertDispatcher` — HMAC-signed mock dispatch + async JSONL logging + burst limit

All thresholds live in `config.py` (`HeuristicConfig`, `CameraConfig`, `DispatchConfig`).
Speed estimation (`speed_estimator.py`): homography ground-point → world coords,
least-squares velocity, per-track Kalman with innovation-gate re-init.

Event priority for fusion: `hit_and_run > collision > speed_drop > anomaly_stop`.
Severity routing: low/medium → traffic police; high → + EMS; critical → + police control.

## Key Conventions

- Velocities in heuristics/config are **m/s** (pixel fallback multiplies by `meter_per_pixel`).
- `SeverityConfig` thresholds are km/h-scale — convert m/s with `MPS_TO_KMH` before comparing.
- Tracker IDs are reused between heuristics and speed estimator (same `Tracker` instance).
- New event kinds: add to `heuristics.py`, `verification.py`, `fusion.py` (`EVENT_PRIORITY`),
  `severity.py` (`kind_baseline`, scorers), `config.py` thresholds.
- Two-branch design (MERGED 2026-08-06): accident (`AccidentPipeline`) and
  violence (`ViolencePipeline`) run independently (own model/verification/
  dispatch); `run_accident`/`run_violence` flags gate them in demo.py/GUI.
- `AlertDispatcher.close()` must be called after the frame loop or queued alerts are lost.

## Known Issues (from code review, 2026-08-04)

1. ~~**`TrackHistory.active_ids()` bug** (`track_history.py`)~~ **FIXED 2026-08-05**:
   both `include_stale` branches used to return `list(self.tracks.keys())`, so heuristics
   could evaluate stale tracks. Now returns tracks last seen within a recency window
   (`ACTIVE_ID_RECENCY_S = 0.25`, `__init__(active_id_recency_s=…)`, from
   `self._last_update_t`) so brief 1–2 frame detection dropouts don't starve the
   collision IoU pair (see 2026-08-02 revert history in VISTA.md — restricting to
   same-frame ids broke clip_03). `include_stale=True` still returns all tracks;
   `_current_frame_ids` kept informational.
2. ~~**Unit mismatch in `render.py` `SpeedEstimator.estimate_kmh`**~~ **FIXED 2026-08-05**:
   `history.velocity()` always returns m/s (ML homography m/s; pixel fallback already
   × `meter_per_pixel` in track_history), so `kmh = raw_v * 3.6` unconditionally — the
   old px/m division made km/h ~20× too low. `manual_px_per_meter` kept as a no-op for
   API compat; `REAL_WORLD_WIDTH_M`/`DEFAULT_REAL_WIDTH_M` (dead auto-estimate) removed.
   `--px-per-meter` (demo.py) and the GUI px/m field now land in
   `CameraConfig.meter_per_pixel = 1/px_per_m`, which is where the fallback scale is
   consumed (ignored when a homography/profile is present).
3. ~~**Doc drift**~~ **FIXED 2026-08-05**: README speed_drop table now matches
   `config.py` (prior ≥ 2.0 m/s, ratio > 0.8; stale near-stop row removed); km/h overlay
   bullet now says homography else `meter_per_pixel` fallback (no auto-scale).
   Note: VISTA.md 2026-08-02 entry describes the OLD speed_drop tuning (ratio 0.65) —
   history, not a doc to sync.
4. **Memory**: clip buffer holds ~112 full frames ≈ 300 MB peak (`pipeline.py:85`).
5. **Repo hygiene**: `yolo11m.pt` (39 MB) + test videos committed (repo 51 MB);
   `VISTA.md` largely duplicates README.
6. ~~`gui_app.py` `_nearest_incident` is dead code~~ **FIXED 2026-08-06**: removed (was
   deliberately disabled; every dispatch creates its own card — history in VISTA.md/commits).
7. Dashboard binds `0.0.0.0` unauthenticated (`tools/dashboard.py`) — demo only.
8. Multi-camera runner: N threads each open their own handle on the shared JSONL log —
   fine for demo volume, not atomic under heavy load.

## Untracked Local Files (git status notes)

- `alerts.jsonl` / `test_alerts.jsonl` — dispatch logs (regenerated on each run)
- `vista_screenshots/`, `vista_clips/`, `vista_output.mp4` — GUI/CLI artifacts
- `gui_error.log` — GUI crash traceback log
- `yolo11m.pt` shows untracked after re-download (repo hygiene issue #5; ignored via .gitignore)
- `New Text Document.txt` — user scratch file, leave alone
- COMMITTED 2026-08-06 (was untracked): `camera_profile.py`, `tools/validate_speed.py`,
  `tools/make_synthetic_clip.py`, `camera_profiles/` (CAM-SYNTH.json, CAM-03.json),
  `calib_demo.mp4`, `AGENTS.md`; modified: `demo.py`, `gui_app.py`, `render.py`,
  `track_history.py`, `config.py`, `speed_estimator.py`, `detector.py`, `verification.py`,
  `heuristics.py`, `pipeline.py`, `severity.py`, `alert.py`, `tools/calibrate_camera.py`,
  `tools/dashboard.py`, `tools/draw_stop_zones.py`, `test_accuracy.py`, `test_scenario.py`,
  `README.md`, `vista_accident/README.md`, `.gitignore` (new)

## Goal

Make all speed/velocity readings physically real by replacing the flat
`meter_per_pixel = 0.05` guess with a real ground-plane homography, and
verify speeds are accurate against known distances.

## Setup on YOUR machine (do this first)

1. Clone the repo (note the trailing dot in the name):
   git clone https://github.com/Chirayu1167/V.I.S.T.A..git
   cd V.I.S.T.A
2. Create a Python 3.10+ venv and install:
   pip install -r vista_accident/requirements.txt
   pip install PyQt5          # needed for the GUI calibration flow
   (No GPU needed for the tests or the calibration tool — CPU is fine,
   torch will install a CPU build automatically. CUDA only speeds up demo runs.)
3. All commands below run from the `vista_accident/` directory.
4. Set $env:PYTHONIOENCODING='utf-8' in PowerShell before running tests.
5. Read `vista_accident/README.md` for project context.

## Git flow

- Commit under YOUR OWN GitHub name/email (git config user.name/user.email) so
  the work is attributed to you. You need collaborator access to the repo —
  if your push is rejected with 403, stop and ask the team lead to add you.
- Push to the main branch (the team works directly on main for the hackathon).

## Current state (already implemented — do NOT rebuild from scratch)

- `vista_accident/vista_accident/tools/calibrate_camera.py` — interactive +
  headless click/point homography tool. Prints config snippet + bird's-eye
  preview. WORKS but is CLI-only.
- `vista_accident/vista_accident/speed_estimator.py` — `CameraCalibration`
  (lines 47-86) already computes homography from src/dst points;
  `_pixel_to_world` (lines 172-177) already uses it when present, else falls
  back to flat scale.
- `vista_accident/vista_accident/config.py` — `CameraConfig` already has
  meter_per_pixel, homography_src_points, homography_dst_points (lines 111-158).
- `vista_accident/demo.py` — supports --px-per-meter flag.
- `vista_accident/gui_app.py` — PyQt5 desktop app.
- Tests: `test_accuracy.py`, `test_scenario.py` (synthetic detections, no GPU).

## Demo footage

- The Video/clip_01..10 files are NOT in the repo. Ask the team lead for
  sample clips before starting work, or use your own CCTV-style footage.

## Ground-truth calibration measurement

- You need the real-world distance of at least one visible feature (lane
  width, road marking, measured object) in a frame of YOUR clip.
- If the team lead provides no measurement: use the standard Indian lane
  width of 3.5 m as the assumption and NOTE the assumption in the profile
  JSON + docs. Never invent specific measured values.

## Scope of work

1. **Camera profile file format** — add JSON profile support (e.g.
   camera_profiles/<id>.json with homography points, meter_per_pixel
   fallback, stop_zones, camera id/location/lat/lon). Wire
   AccidentPipeline/demo.py/gui_app.py to load a profile
   (flag like --camera-profile, GUI file picker).
2. **GUI calibration flow** (gui_app.py) — load a frame, click 4+ ground
   points, enter real-world meters per point, compute homography, preview
   bird's-eye view, save profile JSON. Reuse the math from
   calibrate_camera.py by importing it — do not duplicate.
3. **Calibrate at least one demo clip end-to-end** using the measurement
   from "Ground-truth calibration measurement" above.
4. **Speed validation script** — a script that, given a clip with a
   known-distance marker pair, measures a vehicle's estimated speed vs.
   ground truth (time to cross known distance) and reports error %.
   This is the acceptance gate.
5. **Threshold audit** — check whether HeuristicConfig m/s thresholds still
   make sense with honest speeds; adjust if needed, re-run both test suites,
   document in the config.py docstring.

## Acceptance criteria

- Calibration profile loads via CLI and GUI; pipeline runs unchanged otherwise.
- With a calibrated clip, on-screen km/h for a straight-moving car is within
  ±15% of ground truth (measured crossing time).
- test_accuracy.py and test_scenario.py all PASS
  (run with $env:PYTHONIOENCODING='utf-8').
- No regressions in demo runs; AccidentPipeline API unchanged for existing
  callers.
- README/vista_accident/README.md updated with the calibration workflow.

## Constraints

- Do NOT add emojis or unnecessary comments. Match existing code style.
- No new hard dependencies (cv2/numpy are fine; easyocr etc. stay optional).
- If you hit a blocker that needs a real-world measurement or repo access,
  stop and message the team lead instead of inventing numbers or guessing.

## Session progress (calibration scope — 2026-08-05)

All five scope items are DONE and verified (nothing committed — user commits manually):

1. **Camera profile format** — `vista_accident/vista_accident/camera_profile.py`
   (untracked, new): `load_profile/save_profile/world_distance_m/find_profiles`.
   JSON fields: homography src/dst, meter_per_pixel fallback, stop_zones,
   camera id/location/lat/lon, fps, `calibration_note`.
2. **GUI calibration flow** — `CalibrationDialog` in `gui_app.py` (click points,
   world meters, compute+preview bird's-eye, save profile; reuses
   `calibrate_camera.build_homography`/`render_birdseye_preview`). Profile
   picker: combo from `find_profiles()` + Browse… + Calibrate…. Verified
   headless (offscreen Qt) — see verification notes below.
3. **End-to-end calibration** — `vista_accident/camera_profiles/CAM-SYNTH.json`
   (untracked, new) + `vista_accident/calib_demo.mp4` (0.5 MB, untracked).
   The team clips have NO measured ground reference, so the calibration is on
   a SYNTHETIC clip with exact geometry (`tools/make_synthetic_clip.py`,
   untracked, new): two 3.5 m lanes (IRC standard), one-point-perspective
   camera (fx=520 fy=440 horizon y=120 z0=8 @1280x720), real YOLO-detected
   car crops from Video/clip_04 re-rendered as ground-plane stickers so YOLO
   tracks them. Car A: 13.9 m/s (50 km/h) lane y=2.0; Car B: 9.72 m/s
   (35 km/h) lane y=5.5. NOTE: car crops live in
   `C:\Users\PCWORL~1\AppData\Local\Temp\opencode\sprites2` — if missing,
   re-extract with the extract_sprites2 approach (YOLO-verified crops).
4. **Speed validation script** — `tools/validate_speed.py` (untracked, new).
   Fixed this session: crossing estimates are captured MID-RUN via
   `speed_estimator.velocity_between(tid, t1, t2)` (a track that leaves the
   scene before clip end was aged out of the estimator, so end-of-run lookup
   returned None). Acceptance gate: **PASS +1.88%** (GT 49.37 km/h vs est
   50.30 km/h; markers (224,472)<->(1056,472) = 16.00 m from homography).
   Re-run: `python -m vista_accident.tools.validate_speed --source calib_demo.mp4
   --profile camera_profiles/CAM-SYNTH.json --markers "[[224.0,472.0],[1056.0,472.0]]" --device cpu`
   NOTE: +1.88% was measured with the OLD circle-edge hit timing; after the
   crossing-time interpolation fix (see "real-clip assumed calibration"
   section below) the synthetic gate reads **+0.36%**.
5. **Threshold audit** — done in `config.py` docstring (m/s thresholds kept;
   they are already physical units).

Verification this session (all headless / CLI):
- `test_scenario.py` + `test_accuracy.py`: all PASS (run with
  $env:PYTHONIOENCODING='utf-8').
- GUI offscreen smoke (temp script `gui_smoke_test.py`): `_use_profile` loads
  profile into combo/status/camera_cfg; `CalibrationDialog` 4 points →
  homography → preview → save → `load_profile` round-trip; `VideoWorker` and
  `AccidentPipeline` carry the profile homography (exact mapping).
- `calibrate_camera.py --points` headless reproduces the exact profile
  geometry; `world_distance_m` returns exactly 16.000 m for the marker pair.
- `demo.py --camera-profile` + `--camera-id/--location/--stop-zones-json`
  overrides verified (homography preserved, overrides applied, clean run on
  synthetic clip, 0 alerts as expected).
- NOTE: this OpenCV 4.13 build rejects FLOAT center points in `cv2.circle`
  (must be int) — GUI clicks already emit ints; only a test artifact.

## Session progress (known-issue fixes — 2026-08-05, after calibration scope)

Known issues #1/#2/#3 fixed and verified (nothing committed — user commits manually):

- **#1 `TrackHistory.active_ids()`** — recency-window default
  (`ACTIVE_ID_RECENCY_S = 0.25` from `_last_update_t`), `include_stale` kept.
  Unit checks: stale track (0.30 s gap) excluded; 1–2 frame dropout (~0.05–0.11 s)
  still active; class filter + include_stale intact.
- **#2 `SpeedEstimator.estimate_kmh`** — `kmh = raw_v * 3.6` always;
  `manual_px_per_meter` now a no-op for API compat; dead
  `REAL_WORLD_WIDTH_M`/`DEFAULT_REAL_WIDTH_M` constants + COCO import removed.
  Unit checks: 13.9 m/s → 50.0 km/h with AND without `manual_px_per_meter`.
  `--px-per-meter` (demo.py) and the GUI px/m field now write
  `CameraConfig.meter_per_pixel = 1/px_per_m` (consumed only when no
  homography/profile; ML estimator ignores it) — demo.py:140-145, gui_app.py:197-201.
- **#3 Doc drift** — README speed_drop table now matches config (prior ≥ 2.0 m/s,
  ratio > 0.8, stale "ended near-stop" row removed); km/h overlay bullet says
  homography else meter_per_pixel fallback. VISTA.md 2026-08-02 entry left as history.

Regression verification (all PASS, $env:PYTHONIOENCODING='utf-8'):
- `test_scenario.py` (1 scenario) + `test_accuracy.py` (4 scenarios): all PASS.
- `validate_speed` re-run after track_history/render changes: **PASS +1.88%**
  (GT 49.37 km/h vs est 50.30 km/h) — same as before the fixes.
- **clip_03 collision regression** (the 2026-08-02 revert case): `demo.py
  --source ..\Video\clip_03.mp4 --device cpu` → CONFIRMED + DISPATCHED
  collision tracks=(10, 11) streak=3 severity=medium — recency window preserves
  the alert that same-frame-strict ids broke. Artifact vista_output_regr.mp4 removed.
- py_compile clean for demo.py / gui_app.py / render.py / track_history.py.
- **`--px-per-meter` wiring** (untracked before, now end-to-end verified):
  `demo.py --source calib_demo.mp4 --px-per-meter 20 --max-frames 60 --device cpu`
  → clean run, correct no-homography fallback warning, 0 alerts (expected).
  Numeric check: 25 px / 0.5 s = 50 px/s × `meter_per_pixel` 0.05 = 2.50 m/s
  = 9.0 km/h via `TrackHistory.velocity` — px/m → `CameraConfig.meter_per_pixel`
  consumption confirmed (demo.py:116-117, gui_app.py:198-199).

Remaining gaps (needs team lead, do NOT invent):
- Real measured ground-plane distance for Video/clip_01..10: until provided,
  clip_03 uses an ASSUMED calibration (CAM-03.json, assumptions documented in
  its calibration_note — car-width-derived, cross-checked vs lane lines).
  Redo with a measured distance before production use.
- READMEs already document the profile + validation workflow (uncommitted).
- Commit: waiting on user's explicit go-ahead (git status shows only intended
  files; nothing staged; user commits manually on main).

## Session progress (real-clip assumed calibration — 2026-08-05, after fixes)

User asked to proceed without the team measurement by assuming the ground
plane distance. Done via documented assumptions (NOT invented measured values):

- **Method**: YOLO car bbox width = 1.8 m standard car width (front/rear
  views only, conf>=0.5, bbox w/h<=2.0) gives px/m at each car's road row;
  flat road => px/m linear in y. clip_03 fit (58 inliers): px/m = 0.3346*(y
  - 66.9), r2=0.975 => H=2.99 m, horizon y=66.9. Model: one-point perspective,
  cx=320, focal f=500 px (~65 deg HFOV, documented).
- **Cross-check** (independent cue): on clip_06 the car-derived scale
  converts detected lane-line gaps to median 3.24/3.24/3.36 m at y=180/260/340
  — within ~5% of the IRC 3.5 m lane width. Car-width and lane-width
  assumptions agree.
- **`camera_profiles/CAM-03.json`** (new, untracked) — src quad
  (120,300)/(520,300)/(520,150)/(120,150), dst (-2.56,6.41)/(2.56,6.41)/
  (7.19,17.98)/(-7.19,17.98); full assumption list in calibration_note.
- **validate_speed methodology fix**: circle-edge hit timing biased the
  crossing window (radius 15 px = 0.19 m near vs 0.54 m far, and shallow
  path/tangent entry made t2 ~0.26 s early) — clip_03 read -20.5% FAIL with a
  correct estimator (manual LS over the true window matched velocity_between
  exactly). `find_crossing` now interpolates the crossing time at each
  marker's perpendicular foot on the path segment (radius only gates which
  segment qualifies). Unit sanity: synthetic re-run IMPROVED to +0.36%
  (was +1.88%) — bias removed, no regression.
- **Acceptance gate on a REAL clip**: clip_03 PASS -5.79% (GT 19.57 km/h vs
  est 18.44 km/h, markers (558,300)<->(583,150) = 13.23 m from homography,
  on track #1's actual path). Command:
  `py -m vista_accident.tools.validate_speed --source ..\Video\clip_03.mp4
  --profile camera_profiles\CAM-03.json --markers "[[558.0,300.0],[583.0,150.0]]" --device cpu`
- **Regression**: `demo.py --source ..\Video\clip_03.mp4
  --camera-profile camera_profiles\CAM-03.json --device cpu` -> collision
  CONFIRMED + DISPATCHED tracks=(11, 14) streak=3 severity=medium with
  homography speeds (8.6 / 16.9 km/h at impact — plausible junction speeds).
  Artifact vista_output_cam03.mp4 removed.
- README.md + vista_accident/README.md updated (synthetic +0.36%, clip_03
  -5.79%, assumptions, marker-foot interpolation note).
- NOTE: the car-path probe used duplicate timestamps (ByteTrack emitted the
  same box twice on some frames, e.g. t=0.63,0.63) — harmless for LS fits.

## Session progress (sanctioned assumption + cleanup — 2026-08-06, COMMITTED)

Team-lead scope confirmation (per his 2026-08-06 message): real camera
measurements are OUT OF SCOPE; use assumed coordinates from standard road
features, ALWAYS documented; field measurement per camera = one-time future
task. The team said "fix all bugs and unused variables" before the push —
done and verified below.

- **Sanctioned assumption in profiles**: both `camera_profiles/*.json` now
  carry `"assumption": "3.5m lane width, 9m dash cycle"` (IRC lane width /
  3 m stripe + 6 m gap dash cycle). CAM-03.json's `calibration_note` re-anchors
  on the 3.5 m lane width — the earlier car-width fit is demoted to an internal
  cross-check (measured lanes 3.24–3.36 m, within ~5%). Swapping in a real
  field measurement later = edit the JSON points + note only, no code changes.
- **READMEs** (both): add "per-camera on-site field measurement is a one-time
  future task" note after the calibration section.
- **Bug fixes** (ruff F/E4/E7/E9/ARG/B + pyflakes clean):
  - `gui_app.py` — missing `import numpy as np` (NameError in
    `CalibrationDialog.on_load_frame`); removed dead `_nearest_incident` (#6).
  - `config.py` (x2) + `detector.py` — `warnings.warn(..., stacklevel=2)`.
  - `tools/calibrate_camera.py` — `raise SystemExit(...) from None` (B904).
  - `tools/make_synthetic_clip.py` — `zip(..., strict=True)` (B905).
- **Unused params/vars removed** (25): `render.py` `SpeedEstimator.estimate_kmh`
  dropped `cls, bbox`; `draw_overlay` and `draw_alert_panel` dropped the dead
  `t`/`display_seconds` params (demo.py/gui_app.py call sites updated —
  `ALERT_DISPLAY_SECONDS` kept, demo.py `--alert-display-seconds` still uses
  it); `speed_estimator._kalman_update` dropped `bbox`; `track_history.update`
  `frame` -> `_frame` (API-compat); cv2 callback `flags/param` ->
  `_flags/_param` (calibrate_camera, draw_stop_zones); loop vars -> `_cls`/
  `_pts`; `make_synthetic_clip.draw_car` dropped unused `H_w2i`; `detector.py`
  ambiguous `l` -> `lum`; unused imports in alert/config/heuristics/pipeline/
  severity/verification/dashboard/validate_speed removed.
- **API-change heads-up for the team**: `render.draw_overlay` is now
  `draw_overlay(frame, tracks, history, active_alerts, speed_estimator,
  show_alert_panel=True)` (no `t`); `SpeedEstimator.estimate_kmh(track_id,
  history)` (no cls/bbox). Both call sites in-repo were updated.
- **Verification (all PASS, 2026-08-06)**: ruff + pyflakes clean, py_compile
  OK; `test_scenario.py` + `test_accuracy.py` all pass; validate_speed clip_03
  **-5.79%**, synthetic **+0.36%**; GUI offscreen smoke (dialog on_load_frame +
  new draw_overlay signature); `demo.py --camera-profile` run on calib_demo.mp4
  clean. Note: `alerts.jsonl`/`test_alerts.jsonl` were RESTORED to HEAD after
  the runs (regenerated logs are run artifacts, not committed).
- `.gitignore` (new): `__pycache__/`, `*.pyc`, `yolo11m.pt` (39 MB model,
  auto-downloaded on first run).

## Session progress (merge with violence branch — 2026-08-06, after push prep)

While calibrating, the parallel team stream landed on main: 5 commits adding
the VIOLENCE BRANCH (`violence_pipeline.py`, `violence_heuristics.py`,
`test_violence.py`, `batch_inference.py`, yolo11n-pose, GUI branch selectors,
control-room console in dashboard.py, dispatch hardening). Calibration commit
was rebased onto it and the merge conflicts resolved (all unions + new
`draw_overlay` signature kept). The "future violence branch" note below is
now HISTORY — the two branches coexist in one tree; `run_accident` /
`run_violence` flags gate them in demo.py/GUI. Post-merge cleanup (same lint
standard): config.py `List` import restored (violence fields use it — my
cleanup had removed it), render.py unused COCO_* imports dropped,
`violence_heuristics.limb_speed` zip -> `itertools.pairwise` (strict=True
breaks sliding pairs), unused args/imports fixed in pipeline/severity/
violence_pipeline/batch_inference/demo/test_violence/dashboard. All 3 test
suites (scenario/accuracy/violence) + both validate_speed gates PASS on the
merged tree.
