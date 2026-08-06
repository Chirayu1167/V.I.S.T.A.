"""
Synthetic-detection test: bypasses the real YOLO model with scripted
bounding boxes for two vehicles driving toward each other, stopping (as if
colliding), plus a pedestrian being struck. Validates that heuristics ->
verification -> secondary-confirmation -> dispatch all wire together
correctly, independent of real detector accuracy.
"""

import numpy as np

from vista_accident import AccidentPipeline, CameraConfig, DispatchConfig, HeuristicConfig
from vista_accident.detector import Detector
from vista_accident.tracker import Tracker

FPS = 25.0
DT = 1.0 / FPS


class ScriptedDetector(Detector):
    """Overrides detect() to return scripted boxes instead of running YOLO."""
    def __init__(self):
        self._frame_idx = 0
        self.script = []  # filled in by build_script()

    def detect(self, _frame):
        boxes = self.script[self._frame_idx] if self._frame_idx < len(self.script) else []
        self._frame_idx += 1
        return boxes  # already in (bbox, conf, cls) form


def build_collision_script(n_frames=60):
    """
    Car A drives right->left, Car B drives left->right, they meet in the
    middle around frame 30 and both stop dead (collision). A pedestrian
    walking near the impact point also goes from walking to motionless
    right as Car A appears to keep drifting slightly (hit-and-run signal).
    """
    script = []
    car_a_x, car_b_x = 50.0, 750.0
    ped_x, ped_y = 380.0, 500.0
    for i in range(n_frames):
        dets = []
        if i < 30:
            car_a_x += 12  # ~300px in 1.2s -> fast approach
            car_b_x -= 12
        else:
            # sudden stop: essentially freeze position (small residual jitter)
            car_a_x += 0.2
            car_b_x -= 0.2

        car_a_box = (car_a_x, 400, car_a_x + 120, 480)
        car_b_box = (car_b_x, 400, car_b_x + 120, 480)
        dets.append((car_a_box, 0.9, 2))  # class 2 = car
        dets.append((car_b_box, 0.9, 2))

        # Pedestrian walks steadily until struck around frame 32, then stops.
        if i < 32:
            ped_x += 3
            ped_box = (ped_x, ped_y, ped_x + 30, ped_y + 70)
        else:
            ped_box = (ped_x, ped_y, ped_x + 30, ped_y + 70)  # frozen -> velocity ~0
        dets.append((ped_box, 0.85, 0))  # class 0 = person

        script.append(dets)
    return script


def main():
    pipeline = AccidentPipeline(
        detector=ScriptedDetector(),
        tracker=Tracker(frame_rate=int(FPS)),
        heuristic_cfg=HeuristicConfig(),
        camera_cfg=CameraConfig(camera_id="TEST-CAM"),
        dispatch_cfg=DispatchConfig(dashboard_log_path="test_alerts.jsonl"),
        fps_hint=FPS,
        use_ml_speed=False,
    )
    pipeline.detector.script = build_collision_script()

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    all_events = []
    for i in range(len(pipeline.detector.script)):
        t = i * DT
        result = pipeline.process_frame(frame, t)
        for ev in result["confirmed_events"]:
            print(f"[t={t:.2f}s / frame {i}] CONFIRMED {ev.kind} tracks={ev.track_ids} meta={ev.meta}")
            all_events.append(ev)

    print(f"\nTotal confirmed events: {len(all_events)}")
    dispatched = sum(1 for _, _, s in pipeline.confirmed_log if s == "dispatched")
    print(f"Total dispatched alerts: {dispatched}")
    kinds = sorted(set(e.kind for e in all_events))
    print(f"Event kinds triggered: {kinds}")
    assert "collision" in kinds, "expected collision to fire on impact signature"
    # The post-crash speed drops and the collision are the SAME physical
    # incident — fusion must collapse them into a single dispatched alert.
    assert dispatched == 1, "expected collision + speed drops to fuse into ONE incident"
    print("\n\u2705 scenario test passed")


if __name__ == "__main__":
    main()
