"""
Alert packaging + multi-channel dispatch.

- Severity scoring uses the dynamic SeverityAssessor by default (imported from
  .severity). Falls back to a static kind->severity map if no override given.
- Dispatch to traffic police / hospital-EMS / police control room is MOCKED
  for the hackathon demo (see config.DispatchConfig) — swap `_send_mock` for
  a real webhook/SMS/API call when government API access exists.
- Dashboard logging is asynchronous (background thread + queue) so a slow
  disk/network write never delays the alert path itself.
"""

import hashlib
import hmac
import json
import queue
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from .config import CameraConfig, DispatchConfig
from .severity import CHANNELS_BY_SEVERITY
from .verification import ConfirmedEvent


# Fallback static routing when no dynamic severity override is provided.
SEVERITY_BY_KIND = {
    "hit_and_run": "high",
    "collision": "high",
    "speed_drop": "medium",
    "anomaly_stop": "low",
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


_SHUTDOWN = object()  # sentinel pushed onto the log queue to stop the worker cleanly


class AlertDispatcher:
    def __init__(self, camera_cfg: CameraConfig, dispatch_cfg: DispatchConfig):
        self.camera_cfg = camera_cfg
        self.dispatch_cfg = dispatch_cfg
        self._counter = 0

        # Global rate limiting: if this many alerts fire within
        # rate_limit_window_s from THIS camera, further alerts in that
        # window are bundled into a single "burst" dispatch instead of each
        # individually paging every channel (a real multi-car pileup can
        # otherwise generate one alert per vehicle pair in a couple of
        # seconds). Per-event dedup/cooldown in verification.py already
        # handles the "same incident refiring" case; this handles "several
        # genuinely distinct incidents at once."
        self._recent_dispatch_times: deque = deque()
        self._burst_active_until = -1.0

        # Async dashboard logging so it never blocks the alert/dispatch path.
        self._log_queue: "queue.Queue" = queue.Queue()
        self._log_thread = threading.Thread(target=self._log_worker, daemon=True)
        self._log_thread.start()

    def build_and_dispatch(self, event: ConfirmedEvent, secondary_result: dict,
                            clip_path: Optional[str] = None,
                            severity: Optional[str] = None) -> AlertPayload:
        self._counter += 1
        if severity is None:
            severity = SEVERITY_BY_KIND.get(event.kind, "medium")
        channels = CHANNELS_BY_SEVERITY.get(severity, ("traffic_police",))

        bundled = self._check_rate_limit(event.t)
        if bundled:
            # Still log/record the alert (nothing is silently dropped), but
            # collapse the outbound channel fan-out to traffic_police only —
            # avoids re-paging EMS/police-control once per event during an
            # obvious multi-incident burst. The dashboard/log still shows
            # every individual event with severity="{orig} (bundled)".
            channels = ("traffic_police",)
            severity = f"{severity} (bundled)"

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

    def _check_rate_limit(self, t: float) -> bool:
        """Returns True if this dispatch should be treated as part of an
        active burst (channels collapsed to traffic_police only)."""
        cfg = self.dispatch_cfg
        window = getattr(cfg, "rate_limit_window_s", 10.0)
        max_alerts = getattr(cfg, "rate_limit_max_alerts", 4)
        if not max_alerts:
            return False  # rate limiting disabled

        self._recent_dispatch_times.append(t)
        while self._recent_dispatch_times and t - self._recent_dispatch_times[0] > window:
            self._recent_dispatch_times.popleft()

        if t < self._burst_active_until:
            return True
        if len(self._recent_dispatch_times) > max_alerts:
            self._burst_active_until = t + window
            return True
        return False

    def _send_mock(self, channel: str, payload: AlertPayload):
        url = getattr(self.dispatch_cfg, f"{channel}_webhook")
        signature = self._sign(payload)
        # Hackathon demo: just print + would-be-POST. Replace with
        # requests.post(url, json=asdict(payload), headers={"X-Vista-Signature": signature})
        # once you have real endpoints (Telegram bot / Slack webhook / gov API).
        # The HMAC signature is computed now so real endpoints can verify
        # payload integrity/authenticity from day one instead of retrofitting
        # it later — see DispatchConfig.hmac_secret.
        print(f"[DISPATCH -> {channel} @ {url}] {payload.kind} ({payload.severity}) "
              f"alert_id={payload.alert_id} tracks={payload.track_ids} sig={signature[:12]}...")

    def _sign(self, payload: AlertPayload) -> str:
        secret = getattr(self.dispatch_cfg, "hmac_secret", None) or "unset-demo-secret"
        body = json.dumps(asdict(payload), sort_keys=True, default=str).encode()
        return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    def _log_worker(self):
        with open(self.dispatch_cfg.dashboard_log_path, "a") as f:
            while True:
                item = self._log_queue.get()
                if item is _SHUTDOWN:
                    self._log_queue.task_done()
                    break
                status, payload = item
                record = asdict(payload)
                record["status"] = status
                record["logged_at"] = time.time()
                f.write(json.dumps(record) + "\n")
                f.flush()
                self._log_queue.task_done()

    def close(self, timeout: Optional[float] = 5.0):
        """Drain the log queue and stop the background thread cleanly.

        The log thread is a daemon so the process won't hang if this is
        never called, but without it, alerts still sitting in the queue
        when the process exits (end of video, Ctrl-C, GUI window close)
        are silently lost. Callers (demo.py, gui_app.py) call this after
        the frame loop ends.
        """
        self._log_queue.put(_SHUTDOWN)
        self._log_thread.join(timeout=timeout)
