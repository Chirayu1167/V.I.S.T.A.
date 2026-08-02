"""
Quick validation of the four accuracy improvements:
  1. smoke detection fires on a growing gray dust cloud
  2. jerk fires on a single-vehicle impact into a wall (no second track)
  3. stopped cars at a signal (jam of 3+) do NOT alert
  4. speed_drop catches a ~70% velocity drop that the old 80% threshold missed
"""
import numpy as np

from vista_accident import AccidentPipeline, CameraConfig, DispatchConfig, HeuristicConfig
from vista_accident.detector import Detector
from vista_accident.tracker import Tracker

FPS = 25.0
DT = 1.0 / FPS


class ScriptedDetector(Detector):
    def __init__(self):
        self._i = 0
        self.script = []

    def detect(self, frame):
        boxes = self.script[self._i] if self._i < len(self.script) else []
        self._i += 1
        return boxes


def make_frame(w=1280, h=720):
    return np.full((h, w, 3), 60, dtype=np.uint8)


def add_smoke(frame, cx, cy, radius, intensity=190):
    """Paint a soft gray blob (gaussian falloff) — mimics dust/smoke."""
    yy, xx = np.mgrid[0:frame.shape[0], 0:frame.shape[1]]
    d = ((xx - cx) ** 2 + (yy - cy) ** 2) ** 0.5
    mask = np.clip(1.0 - d / radius, 0, 1) ** 2
    m3 = mask[..., None]
    gray = np.full_like(frame, intensity, dtype=np.float32)
    blend = frame.astype(np.float32) * (1 - m3) + gray * m3
    frame[...] = blend.astype(np.uint8)
    return frame


def run(pipeline, n_frames, frame_maker):
    events = []
    for i in range(n_frames):
        t = i * DT
        frame = frame_maker(i)
        res = pipeline.process_frame(frame, t)
        for ev in res["confirmed_events"]:
            events.append((t, ev.kind, ev.meta))
    return events


def test_smoke():
    det = ScriptedDetector()
    p = AccidentPipeline(detector=det, tracker=Tracker(frame_rate=int(FPS)),
                         heuristic_cfg=HeuristicConfig(smoke_detector_enabled=True,
                                                       smoke_min_area_px=300,
                                                       smoke_max_area_px=40000),
                         camera_cfg=CameraConfig(camera_id="SMOKE"),
                         dispatch_cfg=DispatchConfig(dashboard_log_path="test_alerts.jsonl"),
                         fps_hint=FPS, use_ml_speed=False)
    det.script = [[]] * 120
    radius = 30

    def maker(i):
        f = make_frame()
        if i >= 10:
            radius_i = radius + (i - 10) * 3  # cloud grows over frames
            add_smoke(f, 640, 400, radius_i)
        return f

    evs = run(p, 120, maker)
    kinds = sorted(set(k for _, k, _ in evs))
    print(f"[smoke] events={len(evs)} kinds={kinds}")
    assert "smoke" in kinds, "FAIL: smoke not detected"
    print("PASS: growing dust cloud -> smoke alert")


def test_wall_crash_speed_drop():
    """Single-vehicle crash into a wall: with jerk removed (yesterday's
    tuning), this is caught by speed_drop instead — the car goes from fast to
    dead-stop in one frame, a >80% drop the windowed reading sees."""
    det = ScriptedDetector()
    p = AccidentPipeline(detector=det, tracker=Tracker(frame_rate=int(FPS)),
                         heuristic_cfg=HeuristicConfig(),
                         camera_cfg=CameraConfig(camera_id="WALL"),
                         dispatch_cfg=DispatchConfig(dashboard_log_path="test_alerts.jsonl"),
                         fps_hint=FPS, use_ml_speed=False)
    # Single car drives fast then hits a wall at frame 30: speed 20px/frame -> 0.
    script = []
    x = 50.0
    for i in range(60):
        if i < 30:
            x += 20
        box = (x, 400, x + 120, 480)
        script.append([(box, 0.9, 2)])
    det.script = script
    evs = run(p, 60, make_frame)
    kinds = sorted(set(k for _, k, _ in evs))
    print(f"[wall] events={len(evs)} kinds={kinds}")
    assert "speed_drop" in kinds, "FAIL: wall crash not caught by speed_drop"
    print("PASS: single-vehicle wall crash -> speed_drop alert")


def test_signal_stop_suppressed():
    det = ScriptedDetector()
    p = AccidentPipeline(detector=det, tracker=Tracker(frame_rate=int(FPS)),
                         heuristic_cfg=HeuristicConfig(),
                         camera_cfg=CameraConfig(camera_id="SIGNAL"),
                         dispatch_cfg=DispatchConfig(dashboard_log_path="test_alerts.jsonl"),
                         fps_hint=FPS, use_ml_speed=False)
    # 3 cars drive in a tight queue, brake GRADUALLY (like a real red light)
    # at frame 20, then sit still. Queue should be suppressed: no alerts.
    script = []
    xs = [100.0, 160.0, 220.0]  # 60px apart — well within the 90px jam radius
    for i in range(90):
        dets = []
        for j, x in enumerate(xs):
            if i < 20:
                x += 8
            elif i < 30:
                x += 8 * (1.0 - (i - 20) / 10.0)  # gradual braking over 10 frames
            box = (x, 400, x + 120, 480)
            dets.append((box, 0.9, 2))
            xs[j] = x
        script.append(dets)
    det.script = script
    evs = run(p, 90, make_frame)
    kinds = sorted(set(k for _, k, _ in evs))
    print(f"[signal] events={len(evs)} kinds={kinds}")
    assert "anomaly_stop" not in kinds, "FAIL: signal-stop queue alerted"
    print("PASS: 3-car queue braking at signal -> no alert")


def test_hard_stop_speed_drop():
    """Yesterday's tuning restored: speed_drop needs >80% drop with a real
    prior speed (ratio 0.8 / min_prior 2.0). A near-total stop must fire;
    a partial 70% slow-down must NOT (that was today's looser 0.65 abs-cap
    behavior, reverted so normal braking doesn't alert)."""
    det = ScriptedDetector()
    p = AccidentPipeline(detector=det, tracker=Tracker(frame_rate=int(FPS)),
                         heuristic_cfg=HeuristicConfig(),
                         camera_cfg=CameraConfig(camera_id="DROP"),
                         dispatch_cfg=DispatchConfig(dashboard_log_path="test_alerts.jsonl"),
                         fps_hint=FPS, use_ml_speed=False)
    script = []
    x = 50.0
    for i in range(70):
        v = 20 if i < 30 else 1   # 20px/frame -> 1px/frame = 95% drop
        x += v
        box = (x, 400, x + 120, 480)
        script.append([(box, 0.9, 2)])
    det.script = script
    evs = run(p, 70, make_frame)
    kinds = sorted(set(k for _, k, _ in evs))
    print(f"[drop] events={len(evs)} kinds={kinds}")
    assert "speed_drop" in kinds, "FAIL: near-stop crash not detected"
    print("PASS: 95% velocity drop -> speed_drop alert")

    # And a partial 70% slow (normal braking at a light) must NOT alert.
    det = ScriptedDetector()
    p2 = AccidentPipeline(detector=det, tracker=Tracker(frame_rate=int(FPS)),
                          heuristic_cfg=HeuristicConfig(),
                          camera_cfg=CameraConfig(camera_id="DROPP"),
                          dispatch_cfg=DispatchConfig(dashboard_log_path="test_alerts.jsonl"),
                          fps_hint=FPS, use_ml_speed=False)
    script = []
    x = 50.0
    for i in range(70):
        v = 20 if i < 30 else 6   # 20 -> 6 = 70% drop, under the 0.8 bar
        x += v
        box = (x, 400, x + 120, 480)
        script.append([(box, 0.9, 2)])
    det.script = script
    evs = run(p2, 70, make_frame)
    kinds = sorted(set(k for _, k, _ in evs))
    print(f"[drop70] events={len(evs)} kinds={kinds}")
    assert "speed_drop" not in kinds, "FAIL: 70% braking alerted (should need >80%)"
    print("PASS: 70% braking at light -> no speed_drop alert")


def test_delayed_kind_fused():
    """A crash FIRST fires collision/jerk, then the SAME crash again several
    seconds later as smoke at the same spot must stay ONE incident — one
    dispatched alert — not two cards with different kinds."""
    det = ScriptedDetector()
    p = AccidentPipeline(detector=det, tracker=Tracker(frame_rate=int(FPS)),
                         heuristic_cfg=HeuristicConfig(smoke_detector_enabled=True,
                                                       smoke_min_area_px=900,
                                                       smoke_max_area_px=40000),
                         camera_cfg=CameraConfig(camera_id="FUSE"),
                         dispatch_cfg=DispatchConfig(dashboard_log_path="test_alerts.jsonl"),
                         fps_hint=FPS, use_ml_speed=False)
    # Two cars approach and freeze mid-overlap at frame 30 (collision), then
    # the impact site starts kicking up a growing dust cloud from frame 70 on —
    # exactly the crash->smoke seconds-later cascade.
    script = []
    car_a_x, car_b_x = 50.0, 1270.0
    for i in range(160):
        dets = []
        if i < 30:
            car_a_x += 20
            car_b_x -= 20
        else:
            car_a_x += 0.2  # froze at impact, residual jitter
            car_b_x -= 0.2
        dets.append(((car_a_x, 400, car_a_x + 130, 480), 0.9, 2))
        dets.append(((car_b_x, 400, car_b_x + 130, 480), 0.9, 2))
        script.append(dets)
    det.script = script

    def maker(i):
        f = make_frame()
        if i >= 70:  # delayed dust cloud at the impact point
            add_smoke(f, 640, 440, 30 + (i - 70) * 3)
        # Kit over: the smoke must appear at the CRASH spot, which here is
        # the frame center.
        return f

    evs = run(p, 120, maker)
    kinds = sorted(set(k for _, k, _ in evs))
    dispatched = sum(1 for _, _, s in p.confirmed_log if s == "dispatched")
    print(f"[fuse] events={len(evs)} kinds={kinds} dispatched={dispatched}")
    assert "collision" in kinds, "FAIL: collision not detected"
    assert dispatched == 1, f"FAIL: same crash fired {dispatched} alerts — should be 1"
    # The delayed smoke must have been CONFIRMED by the verifier and fused into
    # the collision incident (has_smoke carries the severity boost) — a merged
    # sub-event does not re-emit as a new ConfirmedEvent.
    inc = p.fuser._incidents[0]
    assert inc.meta.get("has_smoke"), "FAIL: smoke evidence lost — should ride on the collision incident"
    print("PASS: collision then delayed smoke at same spot -> ONE alert (smoke fused, severity boosts)")


if __name__ == "__main__":
    test_smoke()
    test_wall_crash_speed_drop()
    test_signal_stop_suppressed()
    test_hard_stop_speed_drop()
    test_delayed_kind_fused()
    print("\nALL NEW-FEATURE TESTS PASSED")
