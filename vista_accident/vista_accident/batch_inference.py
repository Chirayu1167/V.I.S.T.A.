"""
batch_inference.py — shared, batched, multi-camera inference layer.

Problem this solves
--------------------
Without this module, every camera builds its own Detector (full YOLO11m
copy) and, if the violence branch is wired in too, its own PoseDetector —
so N cameras x 2 branches = 2N model copies in VRAM, each firing its own
unbatched .predict() call. That duplicates GPU memory N times over and
gets zero benefit from batched inference.

With this module there is exactly ONE Detector and ONE PoseDetector for
the whole process. Every camera's capture thread submits its latest frame
to a shared queue; a background collector thread gathers whatever frames
have arrived within a short bounded window, runs ONE batched forward pass
across all of them, and routes results back to each camera's own
AccidentPipeline / ViolencePipeline via the `detections=` / `persons=`
parameters those classes already accept (see pipeline.py,
violence_pipeline.py).

Everything downstream of detection — tracking, heuristics, verification,
severity, dispatch, clip export, config hot-reload, per-camera stop zones —
is completely untouched; those still run per-camera, in parallel, exactly
as before. Only where the detector call itself happens has moved.

Nothing here is required for single-camera use (demo.py, gui_app.py,
test_scenario.py, test_violence.py) — those keep building and calling
Detector/PoseDetector directly, unchanged. This module is opt-in and only
matters once you're running multiple camera streams concurrently.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np


@dataclass
class _PendingFrame:
    camera_id: str
    frame: np.ndarray
    submitted_at: float
    result_event: threading.Event
    result: object = None  # filled in by the collector before result_event.set()


class BatchCollector:
    """Generic micro-batch collector for ONE shared model.

    Cameras call submit(camera_id, frame) and block until their result is
    ready. A background thread wakes on a bounded timer, drains whatever
    frames arrived within that window (at least 1, at most `max_batch`),
    runs `detect_batch_fn` once, and wakes each waiting caller with its
    own slice of the result.

    Bounding the wait (rather than waiting to fill a full batch of N) caps
    worst-case added latency at `window_s` regardless of camera count or a
    momentarily slow/dropped camera — see the latency discussion in the
    design notes. A bad frame from one camera is isolated inside
    detect_batch_fn itself (Detector.detect_batch / PoseDetector.detect_batch
    already do this) so it can't stall or crash the batch for anyone else.
    """

    def __init__(
        self,
        detect_batch_fn: Callable[[List[np.ndarray]], List[object]],
        window_s: float = 0.008,
        max_batch: int = 32,
        name: str = "batch-collector",
    ):
        self._detect_batch_fn = detect_batch_fn
        self._window_s = window_s
        self._max_batch = max_batch
        self._queue: "queue.Queue[_PendingFrame]" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, camera_id: str, frame: np.ndarray, timeout_s: float = 2.0) -> object:
        """Submit one frame for this cycle's batch; blocks until this
        camera's slice of the batched result is ready (or timeout_s
        elapses, in which case an empty-result fallback is returned rather
        than hanging the caller forever)."""
        pending = _PendingFrame(
            camera_id=camera_id, frame=frame, submitted_at=time.monotonic(),
            result_event=threading.Event(),
        )
        self._queue.put(pending)
        if not pending.result_event.wait(timeout=timeout_s):
            # Detection unavailable this cycle (collector stalled/overloaded) —
            # degrade to "no detections" for this frame rather than blocking
            # the camera's whole processing loop indefinitely.
            return []
        return pending.result

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self):
        while not self._stop.is_set():
            try:
                first = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            batch = [first]
            deadline = time.monotonic() + self._window_s
            while len(batch) < self._max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._queue.get(timeout=remaining))
                except queue.Empty:
                    break

            frames = [p.frame for p in batch]
            try:
                results = self._detect_batch_fn(frames)
            except Exception as e:
                # The whole batch's detect_batch_fn call failed (not a
                # single-frame issue — those are already isolated inside
                # detect_batch itself). Degrade every pending caller in
                # this batch to "no detections" rather than losing frames
                # silently or crashing the collector thread.
                results = [[] for _ in batch]
                print(f"[BatchCollector:{self._thread.name}] batch inference error: {e}")

            for pending, result in zip(batch, results, strict=True):
                pending.result = result
                pending.result_event.set()


def _apply_camera_filter(
    detections: List[Tuple[Tuple[float, float, float, float], float, int]],
    conf_threshold: Optional[float],
    classes: Optional[List[int]],
) -> List[Tuple[Tuple[float, float, float, float], float, int]]:
    """Per-camera post-filter applied after the shared batched call returns.
    Mirrors CameraConfig.detection_conf_threshold / detection_classes (see
    config.py) — lets one camera be stricter/narrower than the shared
    detector's own defaults without needing a second model instance."""
    if conf_threshold is None and classes is None:
        return detections
    out = []
    for bbox, conf, cls in detections:
        if conf_threshold is not None and conf < conf_threshold:
            continue
        if classes is not None and cls not in classes:
            continue
        out.append((bbox, conf, cls))
    return out


class SharedInferenceHub:
    """Owns the ONE shared Detector and ONE shared PoseDetector for the
    whole process, plus their batch collectors. Cameras get a lightweight
    handle (CameraInferenceClient) rather than their own model instance.

    Usage (see multi_camera.py for the full wiring):

        hub = SharedInferenceHub(accident_detector, pose_detector)
        client = hub.client_for(camera_cfg)
        ...
        detections = client.detect_accident(frame)      # replaces detector.detect(frame)
        persons     = client.detect_violence(frame)      # replaces detector.detect(frame)
        result = accident_pipeline.process_frame(frame, t, detections=detections)
        vresult = violence_pipeline.process_frame(frame, t, persons=persons)
    """

    def __init__(
        self,
        accident_detector,
        pose_detector=None,
        accident_window_s: float = 0.008,
        violence_window_s: float = 0.015,
        max_batch: int = 32,
    ):
        self.accident_collector = BatchCollector(
            accident_detector.detect_batch, window_s=accident_window_s,
            max_batch=max_batch, name="accident-batch-collector",
        )
        self.violence_collector = None
        if pose_detector is not None:
            self.violence_collector = BatchCollector(
                pose_detector.detect_batch, window_s=violence_window_s,
                max_batch=max_batch, name="violence-batch-collector",
            )

    def client_for(self, camera_cfg) -> "CameraInferenceClient":
        return CameraInferenceClient(self, camera_cfg)

    def close(self):
        self.accident_collector.close()
        if self.violence_collector is not None:
            self.violence_collector.close()


class CameraInferenceClient:
    """Per-camera handle into the shared hub. Cheap — holds no model
    weights, just a reference to the shared collectors plus this camera's
    own detection_conf_threshold/detection_classes overrides."""

    def __init__(self, hub: SharedInferenceHub, camera_cfg):
        self._hub = hub
        self._camera_id = camera_cfg.camera_id
        self._conf_threshold = getattr(camera_cfg, "detection_conf_threshold", None)
        self._classes = getattr(camera_cfg, "detection_classes", None)

    def detect_accident(self, frame: np.ndarray):
        detections = self._hub.accident_collector.submit(self._camera_id, frame)
        return _apply_camera_filter(detections, self._conf_threshold, self._classes)

    def detect_violence(self, frame: np.ndarray):
        if self._hub.violence_collector is None:
            return []
        return self._hub.violence_collector.submit(self._camera_id, frame)
