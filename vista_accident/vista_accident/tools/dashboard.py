#!/usr/bin/env python3
"""
VISTA — Control-Room Console.

The dispatch-side display of the two-interface demo:

    Interface 1 (GUI)   gui_app.py          upload video -> detect accidents,
                                            writes alerts.jsonl + vista_clips/*.mp4
    Interface 2 (THIS)  control-room console
                                            big-screen viewer over those two outputs

The console is a separate process that watches the alert log and the clip
folder and, for every NEW dispatched alert:

    - sounds a WebAudio siren (synthesized, no audio files; browsers block
      autoplay until the operator clicks the ARM SIREN button)
    - flashes a severity-colored banner with kind / severity / location /
      coordinates / timestamp / camera / plate text
    - auto-plays the per-alert clip once the pipeline finishes writing it
      (~1.5 s after dispatch; the alert log carries the clip_path that only
      becomes a real file later)
    - shows which nearby hospitals / police stations the alert is routed to
      (nearest per role from recipients.json, matched against the severity's
      channels) — the "dispatch simulation" of the demo
    - lets the operator ACK each alert (written to acks.jsonl)

Deliberately stdlib-only (http.server + json), like the old dashboard it
replaces: no new project dependency, no external services.

Usage:
    python -m vista_accident.tools.dashboard --log alerts.jsonl --clips vista_clips
    # then open http://localhost:8787

Works with the multi-camera runner too (every AlertPayload carries its own
camera_id, so one console covers every camera). Run it while the GUI or
demo.py processes video — the console picks up alerts as they land.
"""

import argparse
import json
import math
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>VISTA — Control Room</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, Segoe UI, sans-serif; background:#0d0f13; color:#e7e7ea; margin:0; }
  header { display:flex; align-items:center; gap:14px; padding:12px 20px; background:#16181e; border-bottom:1px solid #23262e; }
  header h1 { font-size:17px; margin:0; letter-spacing:1px; }
  header .sub { color:#8a8d96; font-size:12px; margin-left:auto; }
  button { background:#2c2c33; color:#e7e7ea; border:1px solid #3d3d45; border-radius:5px; padding:6px 14px; cursor:pointer; font-size:12px; }
  button:hover { background:#35353d; }
  #armBtn { background:#7a2e2e; border-color:#a04a4a; font-weight:700; }
  #armBtn.armed { background:#2e6a33; border-color:#4a9a52; }
  #testBtn:disabled { opacity:.4; cursor:default; }
  #armState { font-size:12px; font-weight:600; letter-spacing:1px; }
  #armState.disarmed { color:#e0503c; }
  #armState.armed { color:#5fd15f; }
  main { display:grid; grid-template-columns: 1fr 420px; gap:14px; padding:14px 20px; }
  #viewer { display:flex; flex-direction:column; gap:8px; }
  video { width:100%; background:#000; border-radius:6px; min-height:340px; }
  #selInfo { background:#16181e; border-radius:6px; padding:10px 14px; font-size:12.5px; line-height:1.5; }
  #selInfo .kind { font-weight:700; text-transform:capitalize; }
  .sev { font-weight:700; }
  .sev-low { color:#5fd15f; } .sev-medium { color:#e0c34d; }
  .sev-high { color:#e08a3c; } .sev-critical { color:#e0503c; }
  .muted { color:#8a8d96; }
  #history h2 { font-size:13px; color:#8a8d96; margin:4px 0 8px; letter-spacing:1px; }
  #rows { display:flex; flex-direction:column; gap:8px; max-height:calc(100vh - 150px); overflow-y:auto; }
  .card { background:#16181e; border-left:4px solid #444; border-radius:5px; padding:10px 12px; cursor:pointer; font-size:12.5px; }
  .card.selected { outline:2px solid #3a63d8; }
  .card .top { display:flex; justify-content:space-between; gap:8px; align-items:baseline; }
  .card .meta-line { color:#9a9da6; font-size:11px; margin-top:3px; }
  .card .routed { color:#8fc3f5; font-size:11px; margin-top:4px; }
  .card .clip-status { font-size:11px; margin-top:4px; color:#8a8d96; }
  .card .clip-status.ready { color:#5fd15f; cursor:pointer; text-decoration:underline; }
  .card .ack { margin-top:6px; padding:3px 12px; font-size:11px; }
  .card .ack.done { background:#2e6a33; border-color:#4a9a52; }
  #banner { position:fixed; top:0; left:0; right:0; z-index:50; transform:translateY(-120%); transition:transform .25s; }
  #banner.show { transform:translateY(0); }
  #banner .inner { padding:18px 24px; font-size:16px; font-weight:700; display:flex; gap:20px; align-items:center; flex-wrap:wrap; }
  body.flash-critical { animation: flashRed 1s ease-in-out 5; }
  @keyframes flashRed { 0%,100% { background:#0d0f13; } 50% { background:#3a1111; } }
  #empty { padding:30px; text-align:center; color:#5a5d66; }
</style></head>
<body>
  <header>
    <h1>VISTA — CONTROL ROOM</h1>
    <span id="armState" class="disarmed">SIREN DISARMED</span>
    <button id="armBtn">ARM SIREN</button>
    <button id="testBtn" disabled>TEST</button>
    <span class="sub" id="meta">connecting…</span>
  </header>
  <div id="banner"><div class="inner" id="bannerInner"></div></div>
  <main>
    <section id="viewer">
      <video id="video" controls muted></video>
      <div id="selInfo" class="muted">No alert selected.</div>
    </section>
    <aside id="history">
      <h2>DISPATCHED ALERTS</h2>
      <div id="rows"></div>
      <div id="empty" style="display:none">No alerts logged yet.</div>
    </aside>
  </main>
<script>
const SEV_RANK = {low:0, medium:1, high:2, critical:3};
const SEV_COLOR = {low:'#5fd15f', medium:'#e0c34d', high:'#e08a3c', critical:'#e0503c'};
let armed = false, audioCtx = null, sirenLoop = null;
let seenIds = new Set(), selectedId = null;

const armBtn = document.getElementById('armBtn');
const testBtn = document.getElementById('testBtn');
const armState = document.getElementById('armState');
const video = document.getElementById('video');

armBtn.addEventListener('click', () => {
  if (!audioCtx) audioCtx = new (window.AudioContext||window.webkitAudioContext)();
  audioCtx.resume();
  armed = !armed;
  armBtn.classList.toggle('armed', armed);
  armBtn.textContent = armed ? 'DISARM SIREN' : 'ARM SIREN';
  armState.textContent = armed ? 'SIREN ARMED' : 'SIREN DISARMED';
  armState.className = armed ? 'armed' : 'disarmed';
  testBtn.disabled = !armed;
  if (armed) playSweep();
});

testBtn.addEventListener('click', playSweep);

function playSweep(duration=0.9, cycles=2) {
  if (!armed || !audioCtx || audioCtx.state === 'suspended') return;
  const t0 = audioCtx.currentTime;
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.type = 'sawtooth';
  osc.frequency.setValueAtTime(520, t0);
  for (let c = 0; c < cycles; c++) {
    osc.frequency.linearRampToValueAtTime(780, t0 + duration/2 + c*duration);
    osc.frequency.linearRampToValueAtTime(520, t0 + (c+1)*duration);
  }
  gain.gain.setValueAtTime(0.0001, t0);
  gain.gain.exponentialRampToValueAtTime(0.32, t0 + 0.04);
  gain.gain.exponentialRampToValueAtTime(0.0001, t0 + duration*cycles);
  osc.connect(gain).connect(audioCtx.destination);
  osc.start(t0); osc.stop(t0 + duration*cycles);
}

function startLoop() { stopLoop(); sirenLoop = setInterval(() => playSweep(0.9, 2), 1800); }
function stopLoop() { if (sirenLoop) { clearInterval(sirenLoop); sirenLoop = null; } }

function sevBase(s) { return String(s||'').split(' ')[0]; }

function showBanner(a) {
  const sev = sevBase(a.severity);
  const color = SEV_COLOR[sev] || '#e0503c';
  const el = document.getElementById('banner');
  document.getElementById('bannerInner').innerHTML =
    '<span style="color:' + color + '">⚠ ' + sev.toUpperCase() + '</span>' +
    '<span class="kind">' + (a.kind||'').replace('_',' ') + '</span>' +
    '<span>' + (a.location_name||'') + ' &middot; ' + new Date(a.timestamp*1000).toLocaleTimeString() + '</span>' +
    '<span>' + (a.camera_id||'') + '</span>' +
    (a.meta && a.meta.plate_text ? '<span>PLATE: ' + a.meta.plate_text + '</span>' : '');
  el.classList.add('show');
  clearTimeout(showBanner._t);
  showBanner._t = setTimeout(() => el.classList.remove('show'), 7000);
  if (sev === 'critical') {
    document.body.classList.remove('flash-critical');
    void document.body.offsetWidth;
    document.body.classList.add('flash-critical');
  }
  playSweep();
}

function routedHtml(routed) {
  if (!routed || !routed.length) return '';
  const parts = routed.map(r => r.name + ' (' + r.distance_km + ' km)');
  return '<div class="routed">→ ' + parts.join(' &middot; ') + '</div>';
}

function cardHtml(a) {
  const sev = sevBase(a.severity);
  const color = SEV_COLOR[sev] || '#e0503c';
  const time = new Date(a.timestamp*1000).toLocaleTimeString();
  const clip = a.clip_ready
    ? '<div class="clip-status ready" onclick="event.stopPropagation(); playAlert(\'' + a.alert_id + '\')">▶ PLAY CLIP</div>'
    : '<div class="clip-status">waiting for clip…</div>';
  const acked = (acks.indexOf(a.alert_id) >= 0);
  const ackBtn = '<button class="ack' + (acked ? ' done' : '') + '" ' + (acked ? 'disabled' : '') +
    ' onclick="event.stopPropagation(); ack(\'' + a.alert_id + '\')">' +
    (acked ? 'ACKNOWLEDGED' : 'ACK') + '</button>';
  return '<div class="card" data-id="' + a.alert_id + '" onclick="select(\'' + a.alert_id + '\')">' +
    '<div class="top"><b class="kind">' + (a.kind||'').replace('_',' ') + '</b>' +
    '<span class="sev sev-' + sev + '">' + (a.severity||'') + '</span></div>' +
    '<div class="meta-line">' + time + ' &middot; ' + (a.location_name||'') + ' &middot; ' +
      (a.camera_id||'') + ' &middot; tracks ' + (a.track_ids||[]).join(', ') + '</div>' +
    '<div class="meta-line">' + (a.lat||'') + ', ' + (a.lon||'') + '</div>' +
    routedHtml(a.routed) + clip + ackBtn + '</div>';
}

let acks = [];

function select(id) {
  selectedId = id;
  document.querySelectorAll('.card').forEach(c => c.classList.toggle('selected', c.dataset.id === id));
  const a = alertsById[id];
  if (!a) return;
  const sev = sevBase(a.severity);
  const color = SEV_COLOR[sev] || '#e0503c';
  document.getElementById('selInfo').innerHTML =
    '<span class="kind">' + (a.kind||'').replace('_',' ') + '</span> ' +
    '<span class="sev sev-' + sev + '">' + (a.severity||'') + '</span>' +
    ' &middot; <span class="muted">' + new Date(a.timestamp*1000).toLocaleString() + '</span><br>' +
    (a.location_name||'') + ' &middot; ' + (a.camera_id||'') + '<br>' +
    '<span class="muted">' + (a.lat||'') + ', ' + (a.lon||'') + '</span>' +
    (a.meta && a.meta.plate_text ? '<br>PLATE: <b>' + a.meta.plate_text + '</b>' : '') +
    routedHtml(a.routed);
  if (a.clip_ready) playAlert(id);
}

function playAlert(id) {
  const a = alertsById[id];
  if (!a || !a.clip_ready || !a.clip_name) return;
  video.src = '/clips/' + encodeURIComponent(a.clip_name);
  video.play().catch(() => {});
}
function ack(id) {
  fetch('/api/ack?alert_id=' + encodeURIComponent(id), {method:'POST'}).then(() => refresh());
}

let alertsById = {};

function refresh() {
  fetch('/api/alerts').then(r => r.json()).then(data => {
    document.getElementById('meta').style.color = '';
    onLine(data);
  }).catch(() => {
    document.getElementById('meta').textContent = '⚠ SERVER OFFLINE — retrying…';
    document.getElementById('meta').style.color = '#e0503c';
  });
}

function onLine(data) {
    alertsById = {};
    const newOnes = [];
    for (const a of data.alerts) {
      alertsById[a.alert_id] = a;
      if (!seenIds.has(a.alert_id)) { seenIds.add(a.alert_id); newOnes.push(a); }
    }
    if (seenIds.size > 1000) seenIds.clear();
    acks = data.acks || [];
    document.getElementById('meta').textContent =
      data.count + ' alert(s) · log: ' + (data.log_path || '') + ' · clips: ' + (data.clip_dir || '');
    const rows = document.getElementById('rows');
    rows.innerHTML = data.alerts.slice().reverse().map(cardHtml).join('');
    document.getElementById('empty').style.display = data.alerts.length ? 'none' : 'block';
    if (data.alerts.length && !selectedId) select(data.alerts[data.alerts.length-1].alert_id);

    let criticalOpen = false;
    for (const a of newOnes) {
      showBanner(a);
      if (sevBase(a.severity) === 'critical') criticalOpen = true;
    }
    for (const a of data.alerts) {
      if (sevBase(a.severity) === 'critical' && acks.indexOf(a.alert_id) < 0) criticalOpen = true;
    }
    if (criticalOpen) startLoop(); else stopLoop();

    if (selectedId && alertsById[selectedId]) {
      const sel = alertsById[selectedId];
      const selCard = document.querySelector('.card[data-id="' + selectedId + '"]');
      if (selCard) selCard.classList.add('selected');
      if (sel.clip_ready && (video.getAttribute('src') || '') !== '/clips/' + encodeURIComponent(sel.clip_name)) playAlert(selectedId);
    }
}

refresh();
setInterval(refresh, 1500);
</script>
</body></html>
"""


def _read_alerts(log_path: str, max_rows: int = 500):
    if not os.path.exists(log_path):
        return []
    alerts = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                alerts.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a partially-written last line mid-flush
    return alerts[-max_rows:]


def _read_acks(acks_path: str):
    if not os.path.exists(acks_path):
        return []
    ids = []
    with open(acks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.append(json.loads(line).get("alert_id"))
            except json.JSONDecodeError:
                continue
    return [i for i in ids if i]


def _line_count(log_path: str) -> int:
    if not os.path.exists(log_path):
        return 0
    count = 0
    with open(log_path, encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def _load_recipients(path: str):
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    return [r for r in data if r.get("role") and r.get("lat") is not None and r.get("lon") is not None]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers between two lat/lon points."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# Channel name (from AlertPayload.channels / severity routing) -> recipient role.
CHANNEL_ROLE = {
    "traffic_police": "police",
    "hospital_ems": "hospital",
    "police_control_room": "control",
}


def _route_recipients(alert: dict, recipients: list, nearest_k: int) -> list:
    """Nearest recipients per role present in the alert's channels. This is
    the demo's dispatch simulation: the console shows which nearby stations
    WOULD receive the alert (and later, its clip), matching the severity's
    channel routing already computed by the pipeline."""
    lat, lon = alert.get("lat"), alert.get("lon")
    if not recipients or not lat or not lon:
        return []
    channels = alert.get("channels") or []
    roles = {CHANNEL_ROLE[c] for c in channels if c in CHANNEL_ROLE}
    out = []
    for role in sorted(roles):
        cands = [r for r in recipients if r.get("role") == role]
        cands.sort(key=lambda r: haversine_km(lat, lon, r["lat"], r["lon"]))
        for r in cands[:nearest_k]:
            out.append({
                "role": role,
                "name": r["name"],
                "distance_km": round(haversine_km(lat, lon, r["lat"], r["lon"]), 1),
            })
    return out


def make_handler(log_path: str, clip_dir: str, recipients_path: str,
                 acks_path: str, nearest_k: int):
    recipients = _load_recipients(recipients_path)
    clip_dir = clip_dir or os.path.dirname(os.path.abspath(log_path))

    class Handler(BaseHTTPRequestHandler):
        server_version = "VISTA-ControlRoom/1.0"

        def log_message(self, fmt, *args):
            pass  # keep the console quiet; this is a demo server, not prod

        def _send_json(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/alerts"):
                alerts = _read_alerts(log_path)
                acked = _read_acks(acks_path)
                out = []
                for a in alerts:
                    rec = dict(a)
                    rec["routed"] = _route_recipients(rec, recipients, nearest_k)
                    clip_path = rec.get("clip_path")
                    rec["clip_ready"] = bool(clip_path and os.path.isfile(clip_path))
                    rec["clip_name"] = os.path.basename(clip_path) if clip_path else None
                    out.append(rec)
                self._send_json({
                    "seq": _line_count(log_path),
                    "count": len(out),
                    "log_path": os.path.abspath(log_path),
                    "clip_dir": clip_dir,
                    "acks": acked,
                    "alerts": out,
                })
            elif parsed.path.startswith("/clips/"):
                self._serve_clip(parsed.path[len("/clips/"):])
            else:
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path == "/api/ack":
                alert_id = parse_qs(parsed.query).get("alert_id", [""])[0]
                if alert_id:
                    try:
                        with open(acks_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps({"alert_id": alert_id, "acked_at": time.time()}) + "\n")
                    except OSError:
                        pass
                self._send_json({"ok": True, "acked": alert_id})
            else:
                self.send_error(404)

        def _serve_clip(self, name):
            """Serve a clip file from the clips dir with Range support so the
            browser can seek/play HTML5 video reliably. Name is basename-only
            to prevent path traversal."""
            safe = os.path.basename(name)
            path = os.path.join(clip_dir, safe)
            if not os.path.isfile(path):
                self.send_error(404)
                return
            size = os.path.getsize(path)
            start, end = 0, size - 1
            range_header = self.headers.get("Range")
            if range_header and range_header.startswith("bytes="):
                spec = range_header[6:].split(",")[0].strip()
                if "-" in spec:
                    s, e = spec.split("-", 1)
                    if s:
                        start = int(s)
                    if e:
                        end = min(int(e), size - 1)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default="alerts.jsonl", help="Path to the alerts JSONL log")
    ap.add_argument("--clips", default="vista_clips",
                    help="Directory containing per-alert clip files (served under /clips/)")
    ap.add_argument("--recipients", default="recipients.json",
                    help="JSON list of nearby hospitals/police stations for routing display")
    ap.add_argument("--acks", default="acks.jsonl", help="Where ACK actions are appended")
    ap.add_argument("--nearest-k", type=int, default=1,
                    help="How many nearest recipients to show per role")
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()

    log_abs = os.path.abspath(args.log)
    clip_abs = os.path.abspath(args.clips)
    server = ThreadingHTTPServer(
        ("0.0.0.0", args.port),
        make_handler(log_abs, clip_abs, args.recipients, args.acks, args.nearest_k),
    )
    print(f"[control-room] Watching {log_abs} + {clip_abs} at "
          f"http://localhost:{args.port}  (Ctrl+C to stop)")
    print(f"[control-room] The GUI must write to the SAME alerts.jsonl path "
          f"shown above, or no alerts will appear here.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
