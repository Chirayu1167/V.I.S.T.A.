"""
Alert packaging + multi-channel dispatch.

- Severity scoring decides whether an event is no-injury (-> traffic police)
  or injury-flagged (-> traffic police AND hospital/EMS, fired together).
- Dispatch to traffic police / hospital-EMS / police control room is MOCKED
  for the hackathon demo (see config.DispatchConfig) — swap `_send_mock` for
  a real webhook/SMS/API call when government API access exists.
- Dashboard logging is asynchronous (background thread + queue) so a slow
  disk/network write never delays the alert path itself.
"""

import json
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from .config import CameraConfig, DispatchConfig
from .verification import ConfirmedEvent


# Severity routing table. "high" -> traffic police + EMS (+ control room for
# violence-adjacent cases like hit-and-run). "medium"/"low" -> traffic police
# only. Tune against real incident data before deployment.
SEVERITY_BY_KIND = {
    "hit_and_run": "high",     # pedestrian/cyclist struck — always injury-flagged
    "collision": "high",       # vehicle-vehicle collision — assume injury possible
    "speed_drop": "medium",    # hard braking/impact, no confirmed collision partner
    "anomaly_stop": "low",     # stopped mid-road — could be breakdown, needs triage only
}

CHANNELS_BY_SEVERITY = {
    "high": ("traffic_police", "hospital_ems", "police_control_room"),
    "medium": ("traffic_police",),
    "low": ("traffic_police",),
}


@dataclass
class AlertPayload:
    alert_id: str
    kind: str
    severity: str
    track_ids: tuple
    camera_id: str
    location_name: str
    lat: float
    lon: float
    timestamp: float
    confidence_heuristic: int          # consecutive verified frames
    confidence_secondary: Optional[float]
    secondary_ran: bool
    clip_path: Optional[str]
    channels: tuple
    meta: dict = field(default_factory=dict)


class AlertDispatcher:
    def __init__(self, camera_cfg: CameraConfig, dispatch_cfg: DispatchConfig):
        self.camera_cfg = camera_cfg
        self.dispatch_cfg = dispatch_cfg
        self._counter = 0

        # Async dashboard logging so it never blocks the alert/dispatch path.
        self._log_queue: "queue.Queue" = queue.Queue()
        self._log_thread = threading.Thread(target=self._log_worker, daemon=True)
        self._log_thread.start()

    def build_and_dispatch(self, event: ConfirmedEvent, secondary_result: dict,
                            clip_path: Optional[str] = None) -> AlertPayload:
        self._counter += 1
        severity = SEVERITY_BY_KIND.get(event.kind, "medium")
        channels = CHANNELS_BY_SEVERITY[severity]

        payload = AlertPayload(
            alert_id=f"{self.camera_cfg.camera_id}-{int(event.t)}-{self._counter}",
            kind=event.kind,
            severity=severity,
            track_ids=event.track_ids,
            camera_id=self.camera_cfg.camera_id,
            location_name=self.camera_cfg.location_name,
            lat=self.camera_cfg.lat,
            lon=self.camera_cfg.lon,
            timestamp=event.t,
            confidence_heuristic=event.consecutive_frames,
            confidence_secondary=secondary_result.get("confidence"),
            secondary_ran=secondary_result.get("ran", False),
            clip_path=clip_path,
            channels=channels,
            meta=event.meta,
        )

        for channel in channels:
            self._send_mock(channel, payload)

        # Never let logging block dispatch — enqueue and return immediately.
        self._log_queue.put(("sent", payload))
        return payload

    def _send_mock(self, channel: str, payload: AlertPayload):
        url = getattr(self.dispatch_cfg, f"{channel}_webhook")
        # Hackathon demo: just print + would-be-POST. Replace with requests.post(url, json=...)
        # once you have real endpoints (Telegram bot / Slack webhook / gov API).
        print(f"[DISPATCH -> {channel} @ {url}] {payload.kind} ({payload.severity}) "
              f"alert_id={payload.alert_id} tracks={payload.track_ids}")

    def _log_worker(self):
        with open(self.dispatch_cfg.dashboard_log_path, "a") as f:
            while True:
                status, payload = self._log_queue.get()
                record = asdict(payload)
                record["status"] = status
                record["logged_at"] = time.time()
                f.write(json.dumps(record) + "\n")
                f.flush()
                self._log_queue.task_done()
