#!/usr/bin/env python3
"""
VISTA — Emergency Response demo suite.

Three authority dashboards (Traffic Police / Police / Hospital) plus a
citizen "report incident" page that captures REAL GPS via the browser
Geolocation API, all served by one small stdlib-only HTTP+SQLite service.

Workflow implemented end to end:
    1. /report captures the reporter's actual lat/lon (navigator.geolocation)
       and posts an incident to POST /api/incidents.
    2. The server stores the incident (with GPS + timestamp), computes the
       distance from the incident to every registered authority using the
       Haversine formula (geo.py — no hardcoded "nearest" list), and picks
       the nearest 3 hospitals, 3 police stations, and 3 traffic police
       stations.
    3. A notification record is written for each of those 9 authorities.
    4. Each of the three dashboards polls GET /api/incidents for its own
       authority type and renders the incidents routed to it, on a live map.

This module is purely additive: it does not import from or modify
`vista_accident/` (the ML pipeline) at all.

Usage:
    python -m vista_accident.emergency_response.server
    # then open http://localhost:8890
"""

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import db
from .geo import nearest

AUTHORITY_TYPES = ("hospital", "police", "traffic_police")
AUTHORITY_LABELS = {
    "hospital": "Hospital",
    "police": "Police",
    "traffic_police": "Traffic Police",
}
INCIDENT_TYPES = ["accident", "road_rage", "hit_and_run", "breakdown", "other"]
SEVERITIES = ["low", "medium", "high", "critical"]
NEAREST_N = 3

# Evidential clip files written by the ML pipelines (gui_app.py / demo.py)
# land here. Each ML-detected incident carries a clip_path in meta; we serve
# the clips by basename under /clips/ so the dashboards can play the footage.
DEFAULT_CLIP_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "vista_clips",
))

# ---------------------------------------------------------------------------
# Shared page chrome (kept as string templates, stdlib-only — same approach
# as tools/dashboard.py elsewhere in this project).
# ---------------------------------------------------------------------------

BASE_CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, Segoe UI, sans-serif; background:#0d0f13; color:#e7e7ea; margin:0; }
header { display:flex; align-items:center; gap:14px; padding:12px 20px; background:#16181e; border-bottom:1px solid #23262e; }
header h1 { font-size:17px; margin:0; letter-spacing:.5px; }
header .sub { color:#8a8d96; font-size:12px; margin-left:auto; }
nav a { color:#8fc3f5; text-decoration:none; font-size:12.5px; margin-right:14px; }
nav a:hover { text-decoration:underline; }
main { padding:16px 20px; max-width:1300px; margin:0 auto; }
.grid { display:grid; grid-template-columns: 1fr 420px; gap:16px; }
#map { height:520px; border-radius:8px; background:#16181e; }
button { background:#2c2c33; color:#e7e7ea; border:1px solid #3d3d45; border-radius:5px; padding:8px 16px; cursor:pointer; font-size:13px; }
button:hover { background:#35353d; }
button.primary { background:#2e5da0; border-color:#3d78c9; font-weight:600; }
button.primary:hover { background:#3568b5; }
button.danger { background:#7a2e2e; border-color:#a04a4a; font-weight:600; }
button.danger:hover { background:#8c3535; }
button:disabled { opacity:.4; cursor:default; }
select, input { background:#1b1d24; color:#e7e7ea; border:1px solid #3d3d45; border-radius:5px; padding:7px 10px; font-size:13px; }
label { font-size:12px; color:#9a9da6; display:block; margin:10px 0 4px; }
.card { background:#16181e; border-left:4px solid #444; border-radius:6px; padding:12px 14px; margin-bottom:10px; font-size:13px; }
.card .top { display:flex; justify-content:space-between; gap:8px; align-items:baseline; }
.card .kind { font-weight:700; text-transform:capitalize; }
.card .meta { color:#9a9da6; font-size:11.5px; margin-top:4px; line-height:1.5; }
.sev-low { color:#5fd15f; } .sev-medium { color:#e0c34d; } .sev-high { color:#e08a3c; } .sev-critical { color:#e0503c; }
.card.sev-low { border-left-color:#5fd15f; } .card.sev-medium { border-left-color:#e0c34d; }
.card.sev-high { border-left-color:#e08a3c; } .card.sev-critical { border-left-color:#e0503c; }
.status-pill { display:inline-block; padding:2px 8px; border-radius:10px; font-size:10.5px; font-weight:700; text-transform:uppercase; letter-spacing:.4px; background:#2c2c33; }
.status-notified { color:#e0c34d; } .status-acknowledged { color:#8fc3f5; }
.status-dispatched { color:#c98fe0; } .status-resolved { color:#5fd15f; }
.panel { background:#16181e; border-radius:8px; padding:14px; }
.panel h2 { font-size:13px; color:#8a8d96; margin:0 0 10px; letter-spacing:1px; text-transform:uppercase; }
#list { max-height:520px; overflow-y:auto; }
.empty { color:#6d707a; font-size:12.5px; padding:20px 0; text-align:center; }
.banner { background:#16181e; border:1px solid #2e5da0; border-radius:8px; padding:14px; margin-bottom:14px; }
.form-row { display:flex; gap:14px; flex-wrap:wrap; }
.form-row > div { flex:1; min-width:180px; }
#gpsStatus { font-size:12.5px; margin-top:8px; color:#9a9da6; }
#gpsStatus.ok { color:#5fd15f; } #gpsStatus.err { color:#e0503c; }
footer { text-align:center; color:#5a5d66; font-size:11px; padding:20px; }
#sirenBar { display:flex; align-items:center; gap:12px; padding:10px 0 14px; }
#sirenBar #armState { font-size:12px; font-weight:600; letter-spacing:1px; }
#sirenBar #armState.disarmed { color:#e0503c; }
#sirenBar #armState.armed { color:#5fd15f; }
#armBtn { background:#7a2e2e; border-color:#a04a4a; font-weight:700; }
#armBtn.armed { background:#2e6a33; border-color:#4a9a52; }
#testBtn:disabled { opacity:.4; cursor:default; }
#clipBox { position:relative; margin-bottom:14px; }
#clipBox video { width:100%; max-height:360px; background:#000; border-radius:8px; }
#clipBox .close { position:absolute; top:8px; right:8px; background:#2c2c33; border:1px solid #3d3d45; }
#banner { position:fixed; top:0; left:0; right:0; z-index:50; transform:translateY(-120%); transition:transform .25s; }
#banner.show { transform:translateY(0); }
#banner .inner { padding:18px 24px; font-size:16px; font-weight:700; display:flex; gap:20px; align-items:center; flex-wrap:wrap; background:#16181e; border-bottom:1px solid #23262e; }
body.flash-critical { animation: flashRed 1s ease-in-out 5; }
@keyframes flashRed { 0%,100% { background:#0d0f13; } 50% { background:#3a1111; } }
.clip-link { color:#5fd15f; cursor:pointer; text-decoration:underline; margin-top:6px; display:inline-block; font-size:11.5px; }
"""

def page(title, body, extra_head=""):
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<style>{BASE_CSS}</style>
{extra_head}
</head><body>
<header>
  <h1>VISTA &mdash; Emergency Response</h1>
  <nav>
    <a href="/">Home</a>
    <a href="/report">Report Incident</a>
    <a href="/dashboard/hospital">Hospital</a>
    <a href="/dashboard/police">Police</a>
    <a href="/dashboard/traffic_police">Traffic Police</a>
  </nav>
  <div class="sub">{title}</div>
</header>
<main>{body}</main>
<footer>Demo system &mdash; GPS captured via browser Geolocation API, distances via Haversine formula. No hardcoded routing.</footer>
</body></html>"""


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def render_index():
    cards = "".join(
        f"""<div class="card"><div class="top"><span class="kind">{AUTHORITY_LABELS[t]} Dashboard</span></div>
        <div class="meta">Live incidents routed to registered {AUTHORITY_LABELS[t].lower()} authorities.</div>
        <div style="margin-top:8px"><a href="/dashboard/{t}"><button>Open dashboard</button></a></div></div>"""
        for t in AUTHORITY_TYPES
    )
    body = f"""
    <div class="banner">
      <h2 style="margin-top:0">Report an incident</h2>
      <p style="color:#9a9da6; font-size:13px">Captures your real GPS location, then automatically routes the incident
      to the nearest 3 hospitals, 3 police stations, and 3 traffic police stations using the Haversine formula.</p>
      <a href="/report"><button class="primary">Report Incident</button></a>
    </div>
    <div class="panel"><h2>Authority Dashboards</h2>{cards}</div>
    """
    return page("Home", body)


def render_report_page():
    options = "".join(f'<option value="{t}">{t.replace("_", " ").title()}</option>' for t in INCIDENT_TYPES)
    sev_options = "".join(f'<option value="{s}">{s.title()}</option>' for s in SEVERITIES)
    body = f"""
    <div class="grid">
      <div>
        <div id="map"></div>
      </div>
      <div class="panel">
        <h2>Report Incident</h2>
        <div id="gpsStatus">Requesting GPS permission&hellip;</div>
        <div class="form-row">
          <div>
            <label>Incident type</label>
            <select id="incidentType">{options}</select>
          </div>
          <div>
            <label>Severity (if known)</label>
            <select id="severity">{sev_options}</select>
          </div>
        </div>
        <div style="margin-top:14px">
          <button class="primary" id="submitBtn" disabled>Report Incident</button>
        </div>
        <div id="result" style="margin-top:16px"></div>
      </div>
    </div>
    <script>
    let map, incidentMarker;
    let coords = null;

    function initMap(lat, lon) {{
      map = L.map('map').setView([lat, lon], 14);
      L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
        attribution: '&copy; OpenStreetMap contributors'
      }}).addTo(map);
      incidentMarker = L.marker([lat, lon]).addTo(map).bindPopup('Your reported location').openPopup();
    }}

    function setStatus(msg, cls) {{
      const el = document.getElementById('gpsStatus');
      el.textContent = msg;
      el.className = cls || '';
    }}

    if (!navigator.geolocation) {{
      setStatus('Geolocation is not supported by this browser.', 'err');
    }} else {{
      navigator.geolocation.getCurrentPosition(function(pos) {{
        coords = {{ lat: pos.coords.latitude, lon: pos.coords.longitude }};
        setStatus('GPS location captured (accuracy ~' + Math.round(pos.coords.accuracy) + 'm).', 'ok');
        initMap(coords.lat, coords.lon);
        document.getElementById('submitBtn').disabled = false;
      }}, function(err) {{
        setStatus('Location permission denied or unavailable: ' + err.message, 'err');
      }}, {{ enableHighAccuracy: true, timeout: 15000 }});
    }}

    document.getElementById('submitBtn').addEventListener('click', async function() {{
      if (!coords) return;
      this.disabled = true;
      this.textContent = 'Reporting...';
      const payload = {{
        incident_type: document.getElementById('incidentType').value,
        severity: document.getElementById('severity').value,
        lat: coords.lat,
        lon: coords.lon
      }};
      try {{
        const res = await fetch('/api/incidents', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify(payload)
        }});
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Request failed');
        renderResult(data);
      }} catch (e) {{
        document.getElementById('result').innerHTML = '<div class="card sev-critical">Error: ' + e.message + '</div>';
        this.disabled = false;
        this.textContent = 'Report Incident';
      }}
    }});

    function renderResult(incident) {{
      let html = '<div class="banner"><h2 style="margin-top:0">Incident ' + incident.id + ' reported</h2>';
      html += '<div class="meta">Notified ' + incident.notifications.length + ' authorities nearest to your location:</div></div>';
      const byType = {{}};
      incident.notifications.forEach(n => {{
        (byType[n.authority_type] = byType[n.authority_type] || []).push(n);
        L.marker([n.authority_lat, n.authority_lon]).addTo(map)
          .bindPopup(n.authority_name + ' (' + n.distance_km.toFixed(2) + ' km)');
      }});
      for (const t in byType) {{
        html += '<div class="panel" style="margin-top:10px"><h2>' + t.replace('_',' ') + '</h2>';
        byType[t].forEach(n => {{
          html += '<div class="card"><div class="top"><span class="kind">' + n.authority_name + '</span>' +
                  '<span>' + n.distance_km.toFixed(2) + ' km</span></div></div>';
        }});
        html += '</div>';
      }}
      document.getElementById('result').innerHTML = html;
    }}
    </script>
    """
    return page("Report Incident", body)


def render_dashboard_page(authority_type):
    label = AUTHORITY_LABELS[authority_type]
    body = f"""
    <div id="sirenBar">
      <span id="armState" class="disarmed">SIREN DISARMED</span>
      <button id="armBtn">ARM SIREN</button>
      <button id="testBtn" disabled>TEST</button>
    </div>
    <div id="banner"><div class="inner" id="bannerInner"></div></div>
    <div class="panel" style="margin-bottom:14px">
      <label>Filter by station</label>
      <select id="authoritySelect"><option value="">All {label} stations</option></select>
    </div>
    <div class="panel" style="margin-bottom:14px; display:flex; gap:14px; align-items:flex-end; flex-wrap:wrap">
      <div>
        <label>Show incidents within</label>
        <select id="ageSelect">
          <option value="0">Any time</option>
          <option value="900">Last 15 min</option>
          <option value="3600">Last 1 hour</option>
          <option value="86400">Last 24 hours</option>
        </select>
      </div>
      <div style="margin-left:auto">
        <button id="clearBtn" class="danger" title="Delete all incidents and notifications from every dashboard">Clear all incidents</button>
      </div>
    </div>
    <div id="clipBox" style="display:none">
      <video id="video" controls muted></video>
      <button class="close" id="clipClose">✕ Close clip</button>
    </div>
    <div class="grid">
      <div><div id="map"></div></div>
      <div class="panel">
        <h2>Incoming Incidents</h2>
        <div id="list"><div class="empty">Loading&hellip;</div></div>
      </div>
    </div>
    <script>
    const AUTHORITY_TYPE = {json.dumps(authority_type)};
    const SEV_COLOR = {{low:'#5fd15f', medium:'#e0c34d', high:'#e08a3c', critical:'#e0503c'}};
    let map = L.map('map').setView([22.7196, 75.8577], 12);
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);
    let markers = [];

    // --- WebAudio siren: ARM in the browser (autoplay policy), then every new
    // critical/unacknowledged incident keeps the loop; ACK to silence it.
    let armed = false, audioCtx = null, sirenLoop = null;
    const armBtn = document.getElementById('armBtn');
    const testBtn = document.getElementById('testBtn');
    const armState = document.getElementById('armState');

    armBtn.addEventListener('click', () => {{
      if (!audioCtx) audioCtx = new (window.AudioContext||window.webkitAudioContext)();
      audioCtx.resume();
      armed = !armed;
      armBtn.classList.toggle('armed', armed);
      armBtn.textContent = armed ? 'DISARM SIREN' : 'ARM SIREN';
      armState.textContent = armed ? 'SIREN ARMED' : 'SIREN DISARMED';
      armState.className = armed ? 'armed' : 'disarmed';
      testBtn.disabled = !armed;
      if (armed) playSweep();
    }});
    testBtn.addEventListener('click', playSweep);

    function playSweep(duration=0.9, cycles=2) {{
      if (!armed || !audioCtx || audioCtx.state === 'suspended') return;
      const t0 = audioCtx.currentTime;
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(520, t0);
      for (let c = 0; c < cycles; c++) {{
        osc.frequency.linearRampToValueAtTime(780, t0 + duration/2 + c*duration);
        osc.frequency.linearRampToValueAtTime(520, t0 + (c+1)*duration);
      }}
      gain.gain.setValueAtTime(0.0001, t0);
      gain.gain.exponentialRampToValueAtTime(0.32, t0 + 0.04);
      gain.gain.exponentialRampToValueAtTime(0.0001, t0 + duration*cycles);
      osc.connect(gain).connect(audioCtx.destination);
      osc.start(t0); osc.stop(t0 + duration*cycles);
    }}
    function startLoop() {{ stopLoop(); sirenLoop = setInterval(() => playSweep(0.9, 2), 1800); }}
    function stopLoop() {{ if (sirenLoop) {{ clearInterval(sirenLoop); sirenLoop = null; }} }}

    function sevBase(s) {{ return String(s||'').split(' ')[0]; }}

    function showBanner(r) {{
      const sev = sevBase(r.severity);
      const color = SEV_COLOR[sev] || '#e0503c';
      const meta = r.meta || {{}};
      const el = document.getElementById('banner');
      document.getElementById('bannerInner').innerHTML =
        '<span style="color:' + color + '">⚠ ' + sev.toUpperCase() + '</span>' +
        '<span class="kind">' + String(meta.kind || r.incident_type || '').replace('_',' ') + '</span>' +
        '<span>Incident ' + r.incident_id + ' &middot; ' + new Date(r.timestamp*1000).toLocaleTimeString() + '</span>' +
        '<span>' + r.authority_name + '</span>';
      el.classList.add('show');
      clearTimeout(showBanner._t);
      showBanner._t = setTimeout(() => el.classList.remove('show'), 7000);
      if (sev === 'critical') {{
        document.body.classList.remove('flash-critical');
        void document.body.offsetWidth;
        document.body.classList.add('flash-critical');
      }}
      playSweep();
    }}

    function playClip(name) {{
      const box = document.getElementById('clipBox');
      const video = document.getElementById('video');
      box.style.display = 'block';
      video.src = '/clips/' + encodeURIComponent(name);
      video.play().catch(() => {{}});
      document.getElementById('clipClose').onclick = () => {{ box.style.display='none'; video.pause(); video.removeAttribute('src'); }};
      box.scrollIntoView({{block:'nearest'}});
    }}

    async function loadAuthorities() {{
      const res = await fetch('/api/authorities?type=' + AUTHORITY_TYPE);
      const data = await res.json();
      const sel = document.getElementById('authoritySelect');
      data.forEach(a => {{
        const opt = document.createElement('option');
        opt.value = a.id;
        opt.textContent = a.name;
        sel.appendChild(opt);
        L.marker([a.lat, a.lon], {{opacity:0.6}}).addTo(map).bindPopup(a.name + ' (registered)');
      }});
    }}

    function clearIncidentMarkers() {{
      markers.forEach(m => map.removeLayer(m));
      markers = [];
    }}

    function sevClass(sev) {{ return 'sev-' + (sev || 'low'); }}

    function renderSource(r) {{
      const meta = r.meta || {{}};
      if (meta.threshold === 'ml_detected') {{
        return '<span class="status-pill status-dispatched">ML AUTO-DETECTED</span> ' +
               (meta.kind ? '<span class="status-pill">' + meta.kind.replace('_',' ') + '</span>' : '');
      }}
      return '<span class="status-pill">Citizen report</span>';
    }}

    function renderClip(r) {{
      if (!(r.is_ml && r.clip_ready && r.clip_name)) return '';
      return '<span class="clip-link" onclick="playClip(' + JSON.stringify(r.clip_name) + ')">▶ PLAY CLIP</span>';
    }}

    async function loadIncidents() {{
      const authorityId = document.getElementById('authoritySelect').value;
      const maxAge = document.getElementById('ageSelect').value;
      let url = '/api/incidents?authority_type=' + AUTHORITY_TYPE;
      if (authorityId) url += '&authority_id=' + authorityId;
      if (maxAge) url += '&max_age=' + maxAge;
      const res = await fetch(url);
      const rows = await res.json();
      clearIncidentMarkers();
      const list = document.getElementById('list');
      if (rows.length === 0) {{
        list.innerHTML = '<div class="empty">No incidents reported yet.</div>';
        stopLoop();
        return;
      }}
      list.innerHTML = rows.map(r => {{
        const dt = new Date(r.timestamp * 1000).toLocaleString();
        return `<div class="card ${{sevClass(r.severity)}}">
          <div class="top"><span class="kind">${{r.incident_type.replace('_',' ')}}</span>
          <span class="sev-${{r.severity||'low'}}">${{(r.severity||'unknown').toUpperCase()}}</span></div>
          <div class="meta">
            ${{renderSource(r)}}<br>
            Incident ${{r.incident_id}} &middot; ${{dt}}<br>
            GPS: ${{r.lat.toFixed(5)}}, ${{r.lon.toFixed(5)}} &middot; ${{r.distance_km.toFixed(2)}} km from ${{r.authority_name}}<br>
            <span class="status-pill status-${{r.status}}">${{r.status}}</span><br>
            ${{renderClip(r)}}
          </div>
          <div style="margin-top:8px">
            <select data-nid="${{r.notification_id}}" class="statusSelect">
              <option value="notified" ${{r.status==='notified'?'selected':''}}>Notified</option>
              <option value="acknowledged" ${{r.status==='acknowledged'?'selected':''}}>Acknowledged</option>
              <option value="dispatched" ${{r.status==='dispatched'?'selected':''}}>Dispatched</option>
              <option value="resolved" ${{r.status==='resolved'?'selected':''}}>Resolved</option>
            </select>
          </div>
        </div>`;
      }}).join('');

      // Banner + siren for impactful NEW incidents (per dashboard; ACK silences).
      if (!window.__seenIncidents) window.__seenIncidents = new Set();
      let criticalOpen = false;
      rows.forEach(r => {{
        if (r.is_ml && !window.__seenIncidents.has(r.incident_id)) {{
          window.__seenIncidents.add(r.incident_id);
          showBanner(r);
        }}
        if (r.status === 'notified' || r.status === 'acknowledged') {{
          if (sevBase(r.severity) === 'critical') criticalOpen = true;
        }}
      }});
      if (criticalOpen) startLoop(); else stopLoop();

      rows.forEach(r => {{
        const m = L.marker([r.lat, r.lon], {{title: r.incident_id}}).addTo(map)
          .bindPopup(r.incident_type + ' &mdash; ' + (r.severity||'unknown'));
        markers.push(m);
      }});

      document.querySelectorAll('.statusSelect').forEach(sel => {{
        sel.addEventListener('change', async (e) => {{
          const nid = e.target.getAttribute('data-nid');
          await fetch('/api/notifications/' + nid, {{
            method: 'PATCH',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{status: e.target.value}})
          }});
          loadIncidents();
        }});
      }});
    }}

    document.getElementById('authoritySelect').addEventListener('change', loadIncidents);
    document.getElementById('ageSelect').addEventListener('change', loadIncidents);
    document.getElementById('clearBtn').addEventListener('click', async () => {{
      if (!confirm('Clear ALL incidents from every dashboard? This cannot be undone.')) return;
      const res = await fetch('/api/incidents/clear', {{method:'POST'}});
      if (!res.ok) return alert('Failed to clear incidents.');
      const data = await res.json();
      loadIncidents();
      alert('Cleared ' + data.removed + ' incident(s). All dashboards are now clean.');
    }});
    loadAuthorities().then(loadIncidents);
    setInterval(loadIncidents, 8000);
    </script>
    """
    return page(f"{label} Dashboard", body)


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    db_path = db.DEFAULT_DB_PATH
    clip_dir = DEFAULT_CLIP_DIR

    def log_message(self, fmt, *args):
        pass  # keep stdout quiet; override if you want request logging

    def _send(self, status, body, content_type="text/html; charset=utf-8"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, status, obj):
        self._send(status, json.dumps(obj), "application/json")

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    # -- routing -----------------------------------------------------------

    @staticmethod
    def _augment_incident(row: dict) -> dict:
        """Add clip_ready / clip_name / camera / kind from meta so the page JS
        never has to parse paths. Only ML-detected incidents carry a clip."""
        d = dict(row)
        meta = d.get("meta") or {}
        clip_path = meta.get("clip_path")
        d["clip_ready"] = bool(clip_path and os.path.isfile(clip_path))
        d["clip_name"] = os.path.basename(clip_path) if clip_path else None
        d["is_ml"] = meta.get("threshold") == "ml_detected"
        return d

    def do_GET(self):
        parsed = urlparse(self.path)
        path, qs = parsed.path, parse_qs(parsed.query)

        if path == "/":
            return self._send(200, render_index())
        if path == "/report":
            return self._send(200, render_report_page())
        if path.startswith("/dashboard/"):
            authority_type = path.split("/dashboard/", 1)[1]
            if authority_type not in AUTHORITY_TYPES:
                return self._send(404, "Unknown authority type")
            return self._send(200, render_dashboard_page(authority_type))

        if path == "/api/authorities":
            authority_type = qs.get("type", [None])[0]
            with db.connect(self.db_path) as conn:
                return self._send_json(200, db.get_authorities(conn, authority_type))

        if path == "/api/incidents":
            authority_type = qs.get("authority_type", [None])[0]
            authority_id = qs.get("authority_id", [None])[0]
            if not authority_type or authority_type not in AUTHORITY_TYPES:
                return self._send_json(400, {"error": "authority_type is required (hospital|police|traffic_police)"})
            max_age_s = None
            raw_age = qs.get("max_age", [None])[0]
            if raw_age:
                try:
                    max_age_s = float(raw_age)
                except ValueError:
                    max_age_s = None
            with db.connect(self.db_path) as conn:
                rows = [self._augment_incident(r) for r in
                        db.list_incidents_for_authority(conn, authority_type, authority_id,
                                                        max_age_s=max_age_s)]
                return self._send_json(200, rows)

        if path.startswith("/clips/"):
            return self._serve_clip(path[len("/clips/"):])

        if path.startswith("/api/incidents/"):
            incident_id = path.split("/api/incidents/", 1)[1]
            with db.connect(self.db_path) as conn:
                incident = db.get_incident(conn, incident_id)
                if not incident:
                    return self._send_json(404, {"error": "not found"})
                return self._send_json(200, incident)

        return self._send(404, "Not found")

    def _serve_clip(self, name):
        """Serve a clip file from the clips dir with Range support so the
        browser can seek/play HTML5 video reliably. Name is basename-only
        to prevent path traversal."""
        safe = os.path.basename(name)
        path = os.path.join(self.clip_dir, safe)
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

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/incidents":
            try:
                payload = self._read_json_body()
                lat = float(payload["lat"])
                lon = float(payload["lon"])
            except (KeyError, ValueError, json.JSONDecodeError):
                return self._send_json(400, {"error": "lat/lon (real GPS coordinates) are required"})

            incident_type = payload.get("incident_type", "other")
            if incident_type not in INCIDENT_TYPES:
                incident_type = "other"
            severity = payload.get("severity")
            if severity not in SEVERITIES:
                severity = None
            timestamp = float(payload.get("timestamp") or time.time())
            meta = payload.get("meta") or {}

            with db.connect(self.db_path) as conn:
                incident_id = db.new_incident_id(conn)
                db.insert_incident(conn, {
                    "id": incident_id,
                    "incident_type": incident_type,
                    "lat": lat,
                    "lon": lon,
                    "timestamp": timestamp,
                    "severity": severity,
                    "meta": meta,
                })

                notified_at = time.time()
                for authority_type in AUTHORITY_TYPES:
                    candidates = db.get_authorities(conn, authority_type)
                    for a in nearest(lat, lon, candidates, limit=NEAREST_N):
                        db.insert_notification(conn, incident_id, a, notified_at)

                incident = db.get_incident(conn, incident_id)
            return self._send_json(201, incident)

        if parsed.path == "/api/incidents/clear":
            # Demo 'clean slate' button: wipe every incident + notification.
            with db.connect(self.db_path) as conn:
                removed = db.clear_incidents(conn)
            return self._send_json(200, {"removed": removed})

        return self._send(404, "Not found")

    def do_PATCH(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/notifications/"):
            try:
                notification_id = int(parsed.path.split("/api/notifications/", 1)[1])
            except ValueError:
                return self._send_json(400, {"error": "invalid notification id"})
            payload = self._read_json_body()
            status = payload.get("status")
            if status not in ("notified", "acknowledged", "dispatched", "resolved"):
                return self._send_json(400, {"error": "invalid status"})
            with db.connect(self.db_path) as conn:
                ok = db.update_notification_status(conn, notification_id, status)
            if not ok:
                return self._send_json(404, {"error": "not found"})
            return self._send_json(200, {"id": notification_id, "status": status})

        return self._send(404, "Not found")


def main():
    parser = argparse.ArgumentParser(description="VISTA Emergency Response demo server")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--db", default=db.DEFAULT_DB_PATH)
    parser.add_argument("--clips", default=DEFAULT_CLIP_DIR,
                        help="Directory of per-alert evidence clips, served under /clips/ "
                             "(ML-detected incidents advertise their clip in meta.clip_path)")
    args = parser.parse_args()

    db.init_db(args.db)
    Handler.db_path = args.db
    Handler.clip_dir = os.path.abspath(args.clips)

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"VISTA Emergency Response demo running at http://localhost:{args.port}")
    print(f"  Home:            http://localhost:{args.port}/")
    print(f"  Report incident: http://localhost:{args.port}/report")
    print(f"  Hospital dash:   http://localhost:{args.port}/dashboard/hospital")
    print(f"  Police dash:     http://localhost:{args.port}/dashboard/police")
    print(f"  Traffic dash:    http://localhost:{args.port}/dashboard/traffic_police")
    print(f"  Clips:           serving {Handler.clip_dir} at /clips/ (ML evidence footage)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
