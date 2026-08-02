#!/usr/bin/env python3
"""
Minimal live dashboard for alerts.jsonl — replaces "tail -f the log file" as
the way judges/operators see dispatched alerts.

Deliberately stdlib-only (http.server + json), no new project dependency,
no external services. Not meant to replace a real ops dashboard, just to
turn the async JSONL log that already exists into something you point a
browser at.

Usage:
    python -m vista_accident.tools.dashboard --log alerts.jsonl --port 8787
    # then open http://localhost:8787

Works with the multi-camera runner too: point --log at the shared JSONL
file multi_camera.py writes to (every AlertPayload already carries its own
camera_id, so one dashboard covers every camera).
"""

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>VISTA — Live Alerts</title>
<style>
  body { font-family: -apple-system, Segoe UI, sans-serif; background:#111318; color:#e7e7ea; margin:0; }
  header { padding:14px 20px; background:#1a1c22; border-bottom:1px solid #2a2c33; }
  header h1 { font-size:16px; margin:0; font-weight:600; }
  header .sub { color:#8a8d96; font-size:12px; margin-top:2px; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th, td { text-align:left; padding:8px 14px; border-bottom:1px solid #23252c; }
  th { color:#8a8d96; font-weight:500; position:sticky; top:0; background:#111318; }
  .sev-low { color:#5fd15f; } .sev-medium { color:#e0c34d; }
  .sev-high { color:#e08a3c; } .sev-critical { color:#e0503c; font-weight:600; }
  .kind { text-transform:capitalize; }
  .chan { color:#8a8d96; font-size:12px; }
  #empty { padding:40px; text-align:center; color:#5a5d66; }
</style></head>
<body>
  <header>
    <h1>VISTA — Live Alerts</h1>
    <div class="sub" id="meta">connecting…</div>
  </header>
  <table id="tbl">
    <thead><tr><th>Time</th><th>Camera</th><th>Location</th><th>Kind</th>
      <th>Severity</th><th>Tracks</th><th>Channels</th><th>Plate</th></tr></thead>
    <tbody id="rows"></tbody>
  </table>
  <div id="empty" style="display:none">No alerts logged yet.</div>
<script>
async function refresh() {
  const r = await fetch('/api/alerts');
  const data = await r.json();
  const rows = document.getElementById('rows');
  const meta = document.getElementById('meta');
  const empty = document.getElementById('empty');
  meta.textContent = data.count + " alert(s) — auto-refreshing every 2s";
  empty.style.display = data.alerts.length ? 'none' : 'block';
  rows.innerHTML = data.alerts.slice().reverse().map(a => `
    <tr>
      <td>${new Date(a.timestamp*1000).toLocaleTimeString()}</td>
      <td>${a.camera_id||''}</td>
      <td>${a.location_name||''}</td>
      <td class="kind">${(a.kind||'').replace('_',' ')}</td>
      <td class="sev-${(a.severity||'').split(' ')[0]}">${a.severity||''}</td>
      <td>${(a.track_ids||[]).join(', ')}</td>
      <td class="chan">${(a.channels||[]).join(', ')}</td>
      <td>${(a.meta && a.meta.plate_text) || '—'}</td>
    </tr>`).join('');
}
refresh();
setInterval(refresh, 2000);
</script>
</body></html>
"""


def _read_alerts(log_path: str, max_rows: int = 500):
    if not os.path.exists(log_path):
        return []
    alerts = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                alerts.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a partially-written last line mid-flush
    return alerts[-max_rows:]


def make_handler(log_path: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # keep the console quiet; this is a demo dashboard, not a prod server

        def do_GET(self):
            if self.path.startswith("/api/alerts"):
                alerts = _read_alerts(log_path)
                body = json.dumps({"count": len(alerts), "alerts": alerts}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="alerts.jsonl", help="Path to the alerts JSONL log to serve")
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(args.log))
    print(f"[dashboard] Serving {args.log} at http://localhost:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
