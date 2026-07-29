"""
ByteTrack wrapper. Uses the `supervision` library's ByteTrack implementation
(IoU matching + Kalman filter, no neural net, <1ms/frame) rather than the raw
ByteTrack repo, since `supervision` ships a maintained, pip-installable
version with a clean numpy-in/numpy-out API.

pip install supervision
"""

from typing import List, Tuple

import numpy as np
import supervision as sv


class Tracker:
    def __init__(self, frame_rate: int = 25, track_activation_threshold: float = 0.35,
                 lost_track_buffer: int = 30, minimum_matching_threshold: float = 0.8):
        self._tracker = sv.ByteTrack(
            frame_rate=frame_rate,
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
        )

    def update(self, detections: List[Tuple[Tuple[float, float, float, float], float, int]]
                ) -> List[Tuple[int, Tuple[float, float, float, float], int]]:
        """
        detections: list of (bbox_xyxy, confidence, class_id)
        returns: list of (track_id, bbox_xyxy, class_id)
        """
        if not detections:
            sv_dets = sv.Detections.empty()
        else:
            xyxy = np.array([d[0] for d in detections], dtype=np.float32)
            confidence = np.array([d[1] for d in detections], dtype=np.float32)
            class_id = np.array([d[2] for d in detections], dtype=int)
            sv_dets = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

        tracked = self._tracker.update_with_detections(sv_dets)

        out = []
        if tracked.tracker_id is None:
            return out
        for i in range(len(tracked.tracker_id)):
            tid = int(tracked.tracker_id[i])
            bbox = tuple(float(v) for v in tracked.xyxy[i])
            cls = int(tracked.class_id[i]) if tracked.class_id is not None else -1
            out.append((tid, bbox, cls))
        return out
