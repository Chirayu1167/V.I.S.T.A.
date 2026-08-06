"""
tools/export_engine.py — export the accident and pose models to TensorRT.

A compiled TensorRT engine of the SAME weights typically runs 2-4x faster
than the raw PyTorch FP16 model on the same GPU, with no change to
detections, thresholds, or any downstream feature — it's just a faster
compiled form of the exact model you already validated. This is a one-time
export step, not something the pipeline needs to do at runtime.

Usage:
    python -m vista_accident.tools.export_engine
    python -m vista_accident.tools.export_engine --weights yolo11m.pt yolo11n-pose.pt

After exporting, point Detector(weights="yolo11m.engine") /
PoseDetector(cfg=ViolenceConfig(pose_weights="yolo11n-pose.engine")) at the
.engine file instead of the .pt file.

IMPORTANT — re-validate after exporting: TensorRT FP16 can shift confidence
scores very slightly vs. PyTorch FP16 for the same weights. Since the
heuristics and secondary confirmation key off confidence thresholds,
re-run test_scenario.py, test_violence.py, and test_accuracy.py against
the exported engine before relying on it in production, to confirm nothing
near a threshold boundary flipped.
"""

import argparse


def export_one(weights_path: str, half: bool = True) -> str:
    from ultralytics import YOLO
    model = YOLO(weights_path)
    engine_path = model.export(format="engine", half=half)
    print(f"[export_engine] {weights_path} -> {engine_path}")
    return engine_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--weights", nargs="+", default=["yolo11m.pt", "yolo11n-pose.pt"],
        help="Weight files to export (default: both accident and violence models).",
    )
    ap.add_argument("--no-half", action="store_true", help="Export FP32 instead of FP16.")
    args = ap.parse_args()

    for w in args.weights:
        export_one(w, half=not args.no_half)


if __name__ == "__main__":
    main()
