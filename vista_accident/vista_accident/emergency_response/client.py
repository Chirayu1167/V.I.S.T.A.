"""
VISTA — Emergency Response client bridge.

Forwards every ML-dispatched alert (an `AlertPayload` from
`vista_accident.alert`) to the Emergency Response server's
`POST /api/incidents` endpoint, so a real detected crash shows up on the
hospital / police / traffic-police dashboards just like a manual
`/report` incident — closing the loop: ML detects -> dashboards show ->
nearest authorities notified.

The server (`server.py`) already routes each incident to the nearest 3
hospitals / 3 police stations / 3 traffic police stations via Haversine;
this client only maps the payload onto the same `POST /api/incidents`
schema the /report page uses. No knowledge of the server's internals
beyond that one endpoint.

Stdlib-only (`urllib.request`), consistent with the rest of the server
suite, so it works in any environment the server runs in. The forwarder
is deliberately fire-and-forget: a dead/absent server must NEVER slow the
ML pipeline down, so failures are logged and swallowed.
"""

import json
import urllib.error
import urllib.parse
import urllib.request

# The AlertPayload kinds our pipelines emit mapped onto the incident
# types the server understands (see server.INCIDENT_TYPES).
KIND_TO_INCIDENT_TYPE = {
    "collision": "accident",
    "speed_drop": "accident",
    "anomaly_stop": "accident",
    "hit_and_run": "hit_and_run",
    "violence": "road_rage",
    "road_rage": "road_rage",
}
DEFAULT_INCIDENT_TYPE = "accident"

DEFAULT_ENDPOINT = "http://127.0.0.1:8890/api/incidents"
DEFAULT_TIMEOUT_S = 2.0


def incident_type_for_kind(kind: str) -> str:
    """Map an AlertPayload.kind to the server's incident_type vocabulary."""
    return KIND_TO_INCIDENT_TYPE.get(kind or "", DEFAULT_INCIDENT_TYPE)


def build_payload(payload) -> dict:
    """Turn an AlertPayload (or anything with the same attribute names) into
    the POST /api/incidents body. meta is preserved and enriched so the
    dashboards can show that this is an ML-detected alert, not a manual
    report."""
    kind = getattr(payload, "kind", "")
    severity = getattr(payload, "severity", None)
    # During a multi-alert burst the dispatcher suffixes severity with
    # " (bundled)"; the server only knows bare severities — drop the suffix.
    if isinstance(severity, str) and " (" in severity:
        severity = severity.split(" (", 1)[0]
    meta = dict(getattr(payload, "meta", None) or {})
    meta.update({
        "kind": kind,
        "severity_bundled": getattr(payload, "severity", None),
        "alert_id": getattr(payload, "alert_id", None),
        "track_ids": list(getattr(payload, "track_ids", None) or ()),
        "camera_id": getattr(payload, "camera_id", None),
        "threshold": "ml_detected",
        "confidence_heuristic": getattr(payload, "confidence_heuristic", None),
        "confidence_secondary": getattr(payload, "confidence_secondary", None),
        "secondary_ran": getattr(payload, "secondary_ran", None),
        "clip_path": getattr(payload, "clip_path", None),
        "channels": list(getattr(payload, "channels", None) or ()),
    })
    return {
        "incident_type": KIND_TO_INCIDENT_TYPE.get(kind, DEFAULT_INCIDENT_TYPE),
        "severity": severity,
        "lat": getattr(payload, "lat", None),
        "lon": getattr(payload, "lon", None),
        "timestamp": getattr(payload, "timestamp", None),
        "meta": meta,
    }


def post_incident(payload, endpoint: str = None, timeout: float = DEFAULT_TIMEOUT_S) -> bool:
    """POST one alert payload to the Emergency Response server.

    Returns True on a 2xx response, False otherwise. Never raises: every
    failure (no server, timeout, HTTP error) is swallowed and logged so the
    calling alert pipeline is unaffected no matter what.
    """
    base = (endpoint or DEFAULT_ENDPOINT).rstrip("/")
    url = base + "/api/incidents" if not base.endswith("/api/incidents") else base
    body = json.dumps(build_payload(payload)).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ok = 200 <= resp.status < 300
            if not ok:
                print(f"[emergency_response.client] POST {url} -> HTTP {resp.status}")
            return ok
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError) as e:
        # Server down / not reachable — never break the pipeline for it.
        print(f"[emergency_response.client] forward to {url} failed: {e}")
        return False