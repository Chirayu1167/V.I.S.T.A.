"""
Synthetic tests for the pose-based violence branch — no video, no GPU model.

A ScriptedPoseDetector feeds fake person tracks (bboxes + COCO keypoints)
straight into a real ViolencePipeline (real tracker/verifier/fuser/dispatcher),
so the exact production chain is exercised:

  Scenario 1 (fight):    two people 50px apart, wrists oscillating at 2 Hz
                         -> "violence" alert dispatched, police_control_room routed
  Scenario 2 (far pair): two people 250px apart, static arms -> NO alert
  Scenario 3 (solo):     one person, waving -> NO alert (nobody to fight)
  Scenario 4 (grapple):  DISTANT CCTV — two small boxes 0.5+ IoU for >0.8s,
                         keypoints all NaN (unmeasurable limbs) -> alert via
                         the box-overlap entanglement signal

Run:  & "C:\\Users\\User\\AppData\\Local\\Python\\pythoncore-3.10-64\\python.exe" test_violence.py
(or: python test_violence.py)
"""

import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vista_accident import DispatchConfig, ViolenceConfig, ViolencePipeline  # noqa: E402

FPS = 25.0
CADENCE = 3          # must match ViolenceConfig default
BODY = [             # plausible COCO-17 keypoint x,y offsets from bbox top-left
    (40, 20),        # 0 nose
    (35, 15), (45, 15),
    (30, 20), (50, 20),
    (25, 50), (55, 50),   # shoulders
    (15, 85), (65, 85),   # elbows
    (10, 120), (70, 120), # wrists
    (30, 130), (50, 130), # hips
    (30, 165), (50, 165), # knees
    (30, 200), (50, 200), # ankles
]


def make_person(x, y, w, h, t, arm_freq=0.0, arm_amp=0.0, no_kpts=False):
    """One fake person: (bbox, conf, kpts(17,2)). Wrists (9,10) oscillate
    horizontally at arm_freq Hz with arm_amp px when arm_freq > 0.
    no_kpts=True -> all keypoints NaN (distant CCTV: pose unmeasurable)."""
    bbox = (float(x), float(y), float(x + w), float(y + h))
    kpts = np.full((17, 2), np.nan, dtype=np.float32)
    if no_kpts:
        return (bbox, 0.95, kpts)
    for k, (ox, oy) in enumerate(BODY):
        kx, ky = x + ox, y + oy
        if k in (9, 10) and arm_freq > 0:
            kx += arm_amp * math.sin(2 * math.pi * arm_freq * t)
        kpts[k] = (kx, ky)
    return (bbox, 0.95, kpts)


class ScriptedPoseDetector:
    """Returns the same scene every call (ignores the frame)."""

    def __init__(self, persons):
        self.persons = persons

    def detect(self, _frame):
        return list(self.persons)


def run_scenario(persons_fn, frames=60, label=""):
    log_path = os.path.join(tempfile.mkdtemp(), "test_alerts.jsonl")
    detector = ScriptedPoseDetector(persons_fn(0.0))
    pipeline = ViolencePipeline(
        cfg=ViolenceConfig(),
        detector=detector,
        dispatch_cfg=DispatchConfig(dashboard_log_path=log_path),
        device="cpu",
        fps_hint=FPS,
    )
    blank = np.zeros((360, 640, 3), dtype=np.uint8)
    payloads = []
    for i in range(frames):
        t = i / FPS
        if i % CADENCE == 0:
            detector.persons = persons_fn(t)
        res = pipeline.process_frame(blank, t)
        payloads.extend(res["alerts"])
    pipeline.close()

    print(f"[{label}] frames={frames} confirmed={len(pipeline.confirmed_log)} "
          f"dispatched={len(payloads)}")
    return payloads


def fight_scene(t):
    """Two people 50px apart, wrists oscillating at 2 Hz, 30px amplitude."""
    p1 = make_person(260, 110, 80, 180, t, arm_freq=2.0, arm_amp=30.0)
    p2 = make_person(310, 110, 80, 180, t, arm_freq=2.0, arm_amp=30.0)
    return [p1, p2]


def far_scene(t):
    """Two people 250px apart, static arms."""
    p1 = make_person(100, 110, 80, 180, t)
    p2 = make_person(350, 110, 80, 180, t)
    return [p1, p2]


def solo_scene(t):
    """One person waving fast — nobody close to fight."""
    return [make_person(260, 110, 80, 180, t, arm_freq=2.0, arm_amp=40.0)]


def grapple_scene(t):
    """Distant CCTV fight: two SMALL boxes (50x80) heavily overlapped
    (IoU ~0.5), ALL keypoints NaN so limb motion is unmeasurable. Must be
    caught by the box-overlap entanglement signal."""
    p1 = make_person(270, 120, 50, 80, t, no_kpts=True)
    p2 = make_person(285, 122, 50, 80, t, no_kpts=True)
    return [p1, p2]


def main():
    passed = True

    alerts = run_scenario(fight_scene, label="fight")
    ok1 = len(alerts) >= 1 and all(a.kind == "violence" for a in alerts)
    print(f"  fight -> {'PASS' if ok1 else 'FAIL'} (alerts={len(alerts)}, "
          f"kinds={[a.kind for a in alerts]})")
    passed &= ok1

    routed = any("police_control_room" in a.channels for a in alerts)
    print(f"  routing -> {'PASS' if routed else 'FAIL'} "
          f"(channels={[a.channels for a in alerts]})")
    passed &= routed

    far_alerts = run_scenario(far_scene, label="far-pair")
    ok2 = len(far_alerts) == 0
    print(f"  far-pair -> {'PASS' if ok2 else 'FAIL'} (alerts={len(far_alerts)})")
    passed &= ok2

    solo_alerts = run_scenario(solo_scene, label="solo")
    ok3 = len(solo_alerts) == 0
    print(f"  solo -> {'PASS' if ok3 else 'FAIL'} (alerts={len(solo_alerts)})")
    passed &= ok3

    grapple_alerts = run_scenario(grapple_scene, label="grapple-overlap")
    ok4 = len(grapple_alerts) >= 1 and all(a.kind == "violence" for a in grapple_alerts)
    print(f"  grapple-overlap -> {'PASS' if ok4 else 'FAIL'} (alerts={len(grapple_alerts)})")
    passed &= ok4

    print(f"\n{'ALL TESTS PASSED' if passed else 'SOME TESTS FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
