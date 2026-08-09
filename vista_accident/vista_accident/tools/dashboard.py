# #!/usr/bin/env python3
# """
# VISTA — Control-Room Console.

# The dispatch-side display of the two-interface demo:

#     Interface 1 (GUI)   gui_app.py          upload video -> detect accidents,
#                                             writes alerts.jsonl + vista_clips/*.mp4
#     Interface 2 (THIS)  control-room console
#                                             big-screen viewer over those two outputs

# The console is a separate process that watches the alert log and the clip
# folder and, for every NEW dispatched alert:

#     - sounds a WebAudio siren (synthesized, no audio files; browsers block
#       autoplay until the operator clicks the ARM SIREN button)
#     - flashes a severity-colored banner with kind / severity / location /
#       coordinates / timestamp / camera / plate text
#     - auto-plays the per-alert clip once the pipeline finishes writing it
#       (~1.5 s after dispatch; the alert log carries the clip_path that only
#       becomes a real file later)
#     - shows which nearby hospitals / police stations the alert is routed to
#       (nearest per role from recipients.json, matched against the severity's
#       channels) — the "dispatch simulation" of the demo
#     - lets the operator ACK each alert (written to acks.jsonl)

# Deliberately stdlib-only (http.server + json), like the old dashboard it
# replaces: no new project dependency, no external services.

# Usage:
#     python -m vista_accident.tools.dashboard --log alerts.jsonl --clips vista_clips
#     # then open http://localhost:8787

# Works with the multi-camera runner too (every AlertPayload carries its own
# camera_id, so one console covers every camera). Run it while the GUI or
# demo.py processes video — the console picks up alerts as they land.
# """

# import argparse
# import json
# import math
# import os
# import time
# from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
# from urllib.parse import parse_qs, urlparse

# PAGE = """<!doctype html>
# <html><head><meta charset="utf-8"><title>VISTA — Control Room</title>
# <style>
#   :root{
#     --bg:            #F2F4F7;
#     --bg-raised:     #F7F9FC;
#     --bg-panel:      #FFFFFF;
#     --bg-panel-2:    #E7EBF0;
#     --border:        #D9DEE7;
#     --border-soft:   #E7EBF0;
#     --text:          #172033;
#     --text-dim:      #55607A;
#     --text-faint:    #94A0B8;
#     --slate:         #334E68;
#     --red:           #D92D20;
#     --red-strong:    #D92D20;
#     --red-dim:       #FEE4E2;
#     --green:         #16803C;
#     --green-dim:     #E6F4EA;
#     --amber:         #D97706;
#     --orange:        #C2410C;
#     --blue:          #2563EB;
#     --blue-dim:      #EFF6FF;
#     --radius:        12px;
#     --radius-sm:     8px;
#   }
#   * { box-sizing: border-box; }
#   html, body { height:100%; }
#   body {
#     font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
#     background: var(--bg);
#     color: var(--text);
#     margin:0;
#     -webkit-font-smoothing: antialiased;
#     display:flex; flex-direction:column;
#   }

#   /* ---------- header ---------- */
#   header {
#     display:flex; align-items:center; justify-content:space-between;
#     height:68px; flex:none;
#     padding:0 32px;
#     background: var(--bg-raised);
#     border-bottom:1px solid var(--border);
#   }

#   .brand { display:flex; align-items:center; gap:14px; flex:none; }
#   .brand .name { font-size:19px; font-weight:800; letter-spacing:.4px; color:var(--text); }
#   .brand .divider { width:1px; height:22px; background:var(--border); flex:none; }
#   .brand .room { font-size:15px; font-weight:600; color:var(--text-dim); white-space:nowrap; }

#   .header-actions {
#     display:flex; align-items:center;
#     gap:28px;
#     margin-left:auto;
#   }

#   .siren-block { display:flex; align-items:center; gap:8px; font-size:13px; color:var(--text-dim); font-weight:600; white-space:nowrap; }
#   .siren-block .siren-label { color:var(--text-faint); font-weight:500; }
#   #armState { font-size:13px; font-weight:700; letter-spacing:.2px; display:flex; align-items:center; gap:6px; }
#   #armState::before { content:''; width:7px; height:7px; border-radius:50%; background:currentColor; }
#   #armState.disarmed { color:var(--amber); }
#   #armState.armed { color:var(--green); }

#   button {
#     display:flex; align-items:center; gap:8px;
#     background: var(--bg-panel); color:var(--slate);
#     border:1px solid var(--border);
#     border-radius: var(--radius-sm);
#     padding:0 18px; height:38px;
#     cursor:pointer; font-size:13px; font-weight:700; letter-spacing:.2px; white-space:nowrap;
#     transition: background .15s, border-color .15s, opacity .15s, color .15s;
#   }
#   button svg { width:15px; height:15px; flex:none; }
#   button:hover { background:var(--bg-panel-2); border-color:#C7D0DD; }
#   #armBtn { background: var(--red); border-color:var(--red); color:#fff; }
#   #armBtn:hover { background:#B7241A; border-color:#B7241A; }
#   #armBtn.armed { background:var(--green); border-color:var(--green); color:#fff; }
#   #armBtn.armed:hover { background:#116230; border-color:#116230; }
#   #testBtn { color:var(--blue); border-color:var(--border); background:var(--bg-panel); }
#   #testBtn:hover { background:var(--blue-dim); border-color:#BFDBFE; }
#   #testBtn:disabled { opacity:.5; cursor:default; }
#   #testBtn:disabled:hover { background:var(--bg-panel); border-color:var(--border); }

#   .conn-status { display:flex; align-items:center; padding-left:4px; border-left:1px solid var(--border); }
#   .sub {
#     color:var(--amber); font-size:12.5px; font-weight:600;
#     display:flex; align-items:center; gap:6px; white-space:nowrap; padding-left:20px;
#   }
#   .sub::before { content:''; width:6px; height:6px; border-radius:50%; background:currentColor; flex:none; }

#   /* ---------- layout ---------- */
#   main {
#     display:grid; grid-template-columns: minmax(0,1fr) 360px; gap:20px;
#     padding:20px 28px 28px; align-items:stretch;
#     flex:1; min-height:0;
#   }
#   #viewer { display:flex; flex-direction:column; gap:16px; min-height:0; }

#   /* ---------- video card ---------- */
#   .video-card {
#     background: var(--bg-panel);
#     border:1px solid var(--border);
#     border-radius: var(--radius);
#     overflow:hidden;
#     padding:18px 20px 0;
#     display:flex; flex-direction:column;
#     flex:1; min-height:0;
#   }
#   .video-card-head {
#     display:flex; align-items:center; gap:8px;
#     padding:0 0 14px; flex:none;
#   }
#   .video-card-head .r { width:8px; height:8px; border-radius:50%; background:var(--red); flex:none; }
#   .video-card-head h2 {
#     margin:0; font-size:15px; font-weight:700; letter-spacing:.3px; color:var(--text);
#   }
#   .video-frame {
#     position:relative; background:#E7EBF0;
#     border-radius: 8px; overflow:hidden;
#     flex:1; min-height:280px;
#   }
#   .video-frame video { width:100%; height:100%; display:block; object-fit:contain; background:#E7EBF0; }
#   .no-signal {
#     position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center;
#     gap:12px; color:var(--slate); pointer-events:none;
#   }
#   .no-signal svg { width:46px; height:46px; color:var(--slate); }
#   .no-signal .no-signal-title { font-size:18px; font-weight:700; color:var(--text); }
#   .no-signal .no-signal-sub { font-size:13.5px; color:var(--text-dim); }
#   .no-signal.hidden { display:none; }

#   .video-controls {
#     position:absolute; left:0; right:0; bottom:0; z-index:2;
#     display:flex; align-items:center; gap:10px;
#     padding:10px 16px;
#     background: var(--bg-panel);
#     border-top:1px solid var(--border);
#   }
#   .ctrl-btn {
#     background:transparent; border:none; color:var(--slate);
#     font-size:14px; padding:2px 4px; cursor:pointer; line-height:1;
#     height:auto; display:inline-flex;
#   }
#   .ctrl-btn:hover { color:var(--text); background:transparent; }
#   .video-controls .time { font-size:11px; color:var(--text-dim); min-width:32px; text-align:center; font-variant-numeric: tabular-nums; }
#   .seek {
#     flex:1; -webkit-appearance:none; appearance:none; height:4px; border-radius:2px;
#     background: var(--border); outline:none; cursor:pointer;
#   }
#   .seek::-webkit-slider-thumb {
#     -webkit-appearance:none; width:12px; height:12px; border-radius:50%;
#     background:var(--slate); cursor:pointer; margin-top:0;
#   }
#   .seek::-moz-range-thumb { width:12px; height:12px; border-radius:50%; background:var(--slate); border:none; cursor:pointer; }

#   /* ---------- selected-alert bar ---------- */
#   #selInfo {
#     background: var(--bg-panel);
#     border:1px solid var(--border);
#     border-radius: var(--radius);
#     padding:12px 16px;
#     font-size:12.5px; line-height:1.55;
#     display:flex; align-items:center; gap:10px;
#   }
#   #selInfo .info-icon {
#     width:20px; height:20px; border-radius:50%; background:var(--slate); color:#fff;
#     display:flex; align-items:center; justify-content:center; font-size:11px; flex:none;
#   }
#   #selInfo.muted { color:var(--text-faint); }
#   #selInfo .dash { color:var(--text-faint); }
#   #selInfo .kind { font-weight:700; text-transform:capitalize; color:var(--text); }
#   .sev { font-weight:700; }
#   .sev-low { color:var(--green); } .sev-medium { color:var(--amber); }
#   .sev-high { color:var(--orange); } .sev-critical { color:var(--red); }
#   .muted { color:var(--text-dim); }

#   /* ---------- alerts panel ---------- */
#   #history {
#     background: var(--bg-panel);
#     border:1px solid var(--border);
#     border-radius: var(--radius);
#     padding:18px;
#     display:flex; flex-direction:column;
#     height:100%;
#   }
#   .history-head {
#     display:flex; align-items:center; justify-content:space-between;
#     margin:0 0 14px; padding-bottom:14px;
#     border-bottom:1px solid var(--border);
#   }
#   .history-head h2 { font-size:13px; color:var(--slate); margin:0; letter-spacing:1px; font-weight:700; }
#   #alertCount {
#     background: var(--bg-panel-2); border:1px solid var(--border); color:var(--text-dim);
#     font-size:10.5px; font-weight:700; min-width:18px; height:18px; padding:0 5px;
#     border-radius:9px; display:flex; align-items:center; justify-content:center;
#   }
#   #rows { display:flex; flex-direction:column; gap:8px; overflow-y:auto; flex:1; margin:0 -4px; padding:0 4px; }
#   #rows::-webkit-scrollbar { width:6px; }
#   #rows::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }

#   .card {
#     background: var(--bg-panel-2);
#     border:1px solid var(--border-soft);
#     border-left:3px solid #B7C2D0;
#     border-radius: var(--radius-sm);
#     padding:10px 12px;
#     cursor:pointer; font-size:12.5px;
#     transition: border-color .15s, background .15s;
#   }
#   .card:hover { background:var(--bg-panel); border-color:var(--border); }
#   .card.selected { border-color:var(--blue); outline:1px solid var(--blue); }
#   .card .top { display:flex; justify-content:space-between; gap:8px; align-items:baseline; }
#   .card .kind { font-weight:700; color:var(--text); }
#   .card .meta-line { color:var(--text-faint); font-size:11px; margin-top:4px; }
#   .card .routed { color:var(--blue); font-size:11px; margin-top:5px; }
#   .card .clip-status { font-size:11px; margin-top:5px; color:var(--text-faint); }
#   .card .clip-status.ready { color:var(--green); cursor:pointer; text-decoration:underline; }
#   .card .ack { margin-top:8px; padding:4px 12px; font-size:10.5px; height:auto; }
#   .card .ack.done { background:var(--green-dim); border-color:#BEE3C8; color:var(--green); }

#   #empty {
#     flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center;
#     gap:12px; color:var(--text-faint); text-align:center; padding:30px 10px;
#   }
#   #empty .empty-icon {
#     width:48px; height:48px; border-radius:50%; background:var(--slate);
#     display:flex; align-items:center; justify-content:center; color:#fff;
#   }
#   #empty .empty-icon svg { width:22px; height:22px; }
#   #empty .empty-title { font-size:15px; font-weight:700; color:var(--text); }
#   #empty .empty-text { font-size:13px; color:var(--text-faint); }

#   /* ---------- banner ---------- */
#   #banner { position:fixed; top:0; left:0; right:0; z-index:50; transform:translateY(-120%); transition:transform .25s; }
#   #banner .inner {
#     padding:16px 24px; font-size:15px; font-weight:700; display:flex; gap:20px; align-items:center; flex-wrap:wrap;
#     background: var(--bg-panel); border-bottom:1px solid var(--border); color:var(--text);
#   }
#   #banner.show { transform:translateY(0); }
#   body.flash-critical { animation: flashRed 1s ease-in-out 5; }
#   @keyframes flashRed { 0%,100% { background:var(--bg); } 50% { background:var(--red-dim); } }
# </style></head>
# <body>
#   <header>
#     <div class="brand">
#       <span class="name">VISTA</span>
#       <span class="divider"></span>
#       <span class="room">Control room</span>
#     </div>
#     <div class="header-actions">
#       <div class="siren-block">
#         <span class="siren-label">Siren</span>
#         <span id="armState" class="disarmed">Disarmed</span>
#       </div>
#       <button id="armBtn">
#         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M10 21a2 2 0 0 0 4 0"/><path d="M3.2 18h17.6c.6 0 1-.6.7-1.2C20.4 15 19 13 19 9a7 7 0 0 0-14 0c0 4-1.4 6-2.5 7.8-.3.6.1 1.2.7 1.2z"/></svg>
#         <span id="armBtnLabel">Arm siren</span>
#       </button>
#       <button id="testBtn" disabled>
#         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h.01"/><path d="M8.5 16.4a5 5 0 0 1 7 0"/><path d="M5 12.9a10 10 0 0 1 14 0"/><path d="M1.5 9.4a15 15 0 0 1 21 0"/></svg>
#         <span>Test</span>
#       </button>
#       <div class="conn-status">
#         <span class="sub" id="meta">connecting…</span>
#       </div>
#     </div>
#   </header>
#   <div id="banner"><div class="inner" id="bannerInner"></div></div>
#   <main>
#     <section id="viewer">
#       <div class="video-card">
#         <div class="video-card-head">
#           <span class="r"></span>
#           <h2>Live Feed</h2>
#         </div>
#         <div class="video-frame">
#           <video id="video" muted></video>
#           <div id="noSignal" class="no-signal">
#             <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M16 8.5h-9a2 2 0 0 0-2 2v5a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2v-5a2 2 0 0 0-2-2z"/><path d="M18 11.5l4-2.3v7.6l-4-2.3"/><path d="M2 2l20 20"/></svg>
#             <div class="no-signal-title">No signal</div>
#             <div class="no-signal-sub">Camera feed unavailable</div>
#           </div>
#           <div class="video-controls">
#             <button id="playBtn" class="ctrl-btn" type="button">▶</button>
#             <span id="curTime" class="time">0:00</span>
#             <input id="seek" class="seek" type="range" min="0" max="1000" value="0">
#             <span id="durTime" class="time">0:00</span>
#             <button id="fsBtn" class="ctrl-btn" type="button">⛶</button>
#           </div>
#         </div>
#       </div>
#       <div id="selInfo" class="muted"><span class="info-icon">i</span>No alert selected.</div>
#     </section>
#     <aside id="history">
#       <div class="history-head">
#         <h2>DISPATCHED ALERTS</h2>
#         <span id="alertCount">0</span>
#       </div>
#       <div id="rows"></div>
#       <div id="empty" style="display:flex">
#         <div class="empty-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 8a6 6 0 0 0-12 0c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/></svg></div>
#         <div class="empty-title">No active alerts at this time</div>
#       </div>
#     </aside>
#   </main>
# <script>
# const SEV_RANK = {low:0, medium:1, high:2, critical:3};
# const SEV_COLOR = {low:'#16803C', medium:'#D97706', high:'#C2410C', critical:'#D92D20'};
# let armed = false, audioCtx = null, sirenLoop = null;
# let seenIds = new Set(), selectedId = null;

# const armBtn = document.getElementById('armBtn');
# const armBtnLabel = document.getElementById('armBtnLabel');
# const testBtn = document.getElementById('testBtn');
# const armState = document.getElementById('armState');
# const video = document.getElementById('video');

# armBtn.addEventListener('click', () => {
#   if (!audioCtx) audioCtx = new (window.AudioContext||window.webkitAudioContext)();
#   audioCtx.resume();
#   armed = !armed;
#   armBtn.classList.toggle('armed', armed);
#   armBtnLabel.textContent = armed ? 'Disarm siren' : 'Arm siren';
#   armState.textContent = armed ? 'Armed' : 'Disarmed';
#   armState.className = armed ? 'armed' : 'disarmed';
#   testBtn.disabled = !armed;
#   if (armed) playSweep();
# });

# testBtn.addEventListener('click', playSweep);

# function playSweep(duration=0.9, cycles=2) {
#   if (!armed || !audioCtx || audioCtx.state === 'suspended') return;
#   const t0 = audioCtx.currentTime;
#   const osc = audioCtx.createOscillator();
#   const gain = audioCtx.createGain();
#   osc.type = 'sawtooth';
#   osc.frequency.setValueAtTime(520, t0);
#   for (let c = 0; c < cycles; c++) {
#     osc.frequency.linearRampToValueAtTime(780, t0 + duration/2 + c*duration);
#     osc.frequency.linearRampToValueAtTime(520, t0 + (c+1)*duration);
#   }
#   gain.gain.setValueAtTime(0.0001, t0);
#   gain.gain.exponentialRampToValueAtTime(0.32, t0 + 0.04);
#   gain.gain.exponentialRampToValueAtTime(0.0001, t0 + duration*cycles);
#   osc.connect(gain).connect(audioCtx.destination);
#   osc.start(t0); osc.stop(t0 + duration*cycles);
# }

# function startLoop() { stopLoop(); sirenLoop = setInterval(() => playSweep(0.9, 2), 1800); }
# function stopLoop() { if (sirenLoop) { clearInterval(sirenLoop); sirenLoop = null; } }

# function sevBase(s) { return String(s||'').split(' ')[0]; }

# function showBanner(a) {
#   const sev = sevBase(a.severity);
#   const color = SEV_COLOR[sev] || '#D92D20';
#   const el = document.getElementById('banner');
#   document.getElementById('bannerInner').innerHTML =
#     '<span style="color:' + color + '">⚠ ' + sev.toUpperCase() + '</span>' +
#     '<span class="kind">' + (a.kind||'').replace('_',' ') + '</span>' +
#     '<span>' + (a.location_name||'') + ' &middot; ' + new Date(a.timestamp*1000).toLocaleTimeString() + '</span>' +
#     '<span>' + (a.camera_id||'') + '</span>' +
#     (a.meta && a.meta.plate_text ? '<span>PLATE: ' + a.meta.plate_text + '</span>' : '');
#   el.classList.add('show');
#   clearTimeout(showBanner._t);
#   showBanner._t = setTimeout(() => el.classList.remove('show'), 7000);
#   if (sev === 'critical') {
#     document.body.classList.remove('flash-critical');
#     void document.body.offsetWidth;
#     document.body.classList.add('flash-critical');
#   }
#   playSweep();
# }

# function routedHtml(routed) {
#   if (!routed || !routed.length) return '';
#   const parts = routed.map(r => r.name + ' (' + r.distance_km + ' km)');
#   return '<div class="routed">→ ' + parts.join(' &middot; ') + '</div>';
# }

# function cardHtml(a) {
#   const sev = sevBase(a.severity);
#   const color = SEV_COLOR[sev] || '#D92D20';
#   const time = new Date(a.timestamp*1000).toLocaleTimeString();
#   const clip = a.clip_ready
#     ? '<div class="clip-status ready" onclick="event.stopPropagation(); playAlert(\'' + a.alert_id + '\')">▶ PLAY CLIP</div>'
#     : '<div class="clip-status">waiting for clip…</div>';
#   const acked = (acks.indexOf(a.alert_id) >= 0);
#   const ackBtn = '<button class="ack' + (acked ? ' done' : '') + '" ' + (acked ? 'disabled' : '') +
#     ' onclick="event.stopPropagation(); ack(\'' + a.alert_id + '\')">' +
#     (acked ? 'ACKNOWLEDGED' : 'ACK') + '</button>';
#   return '<div class="card" style="border-left-color:' + color + '" data-id="' + a.alert_id + '" onclick="select(\'' + a.alert_id + '\')">' +
#     '<div class="top"><b class="kind">' + (a.kind||'').replace('_',' ') + '</b>' +
#     '<span class="sev sev-' + sev + '">' + (a.severity||'') + '</span></div>' +
#     '<div class="meta-line">' + time + ' &middot; ' + (a.location_name||'') + ' &middot; ' +
#       (a.camera_id||'') + ' &middot; tracks ' + (a.track_ids||[]).join(', ') + '</div>' +
#     '<div class="meta-line">' + (a.lat||'') + ', ' + (a.lon||'') + '</div>' +
#     routedHtml(a.routed) + clip + ackBtn + '</div>';
# }

# let acks = [];

# function select(id) {
#   selectedId = id;
#   document.querySelectorAll('.card').forEach(c => c.classList.toggle('selected', c.dataset.id === id));
#   const a = alertsById[id];
#   if (!a) return;
#   const sev = sevBase(a.severity);
#   const color = SEV_COLOR[sev] || '#D92D20';
#   const selInfo = document.getElementById('selInfo');
#   selInfo.classList.remove('muted');
#   selInfo.innerHTML =
#     '<span class="info-icon">i</span>' +
#     '<span class="kind">' + (a.kind||'').replace('_',' ') + '</span> ' +
#     '<span class="sev sev-' + sev + '">' + (a.severity||'') + '</span>' +
#     ' &middot; <span class="muted">' + new Date(a.timestamp*1000).toLocaleString() + '</span><br>' +
#     (a.location_name||'') + ' &middot; ' + (a.camera_id||'') + '<br>' +
#     '<span class="muted">' + (a.lat||'') + ', ' + (a.lon||'') + '</span>' +
#     (a.meta && a.meta.plate_text ? '<br>PLATE: <b>' + a.meta.plate_text + '</b>' : '') +
#     routedHtml(a.routed);
#   if (a.clip_ready) playAlert(id);
# }

# function playAlert(id) {
#   const a = alertsById[id];
#   if (!a || !a.clip_ready || !a.clip_name) return;
#   video.src = '/clips/' + encodeURIComponent(a.clip_name);
#   setNoSignal(false);
#   video.play().catch(() => {});
# }
# function ack(id) {
#   fetch('/api/ack?alert_id=' + encodeURIComponent(id), {method:'POST'}).then(() => refresh());
# }

# let alertsById = {};

# function refresh() {
#   fetch('/api/alerts').then(r => r.json()).then(data => {
#     document.getElementById('meta').style.color = '#16803C';
#     onLine(data);
#   }).catch(() => {
#     document.getElementById('meta').textContent = '⚠ SERVER OFFLINE — retrying…';
#     document.getElementById('meta').style.color = '#D92D20';
#   });
# }

# function onLine(data) {
#     alertsById = {};
#     const newOnes = [];
#     for (const a of data.alerts) {
#       alertsById[a.alert_id] = a;
#       if (!seenIds.has(a.alert_id)) { seenIds.add(a.alert_id); newOnes.push(a); }
#     }
#     if (seenIds.size > 1000) seenIds.clear();
#     acks = data.acks || [];
#     document.getElementById('meta').textContent =
#       data.count + ' alert(s) · log: ' + (data.log_path || '') + ' · clips: ' + (data.clip_dir || '');
#     document.getElementById('alertCount').textContent = data.count;
#     const rows = document.getElementById('rows');
#     rows.innerHTML = data.alerts.slice().reverse().map(cardHtml).join('');
#     rows.style.display = data.alerts.length ? 'flex' : 'none';
#     document.getElementById('empty').style.display = data.alerts.length ? 'none' : 'flex';
#     if (data.alerts.length && !selectedId) select(data.alerts[data.alerts.length-1].alert_id);

#     let criticalOpen = false;
#     for (const a of newOnes) {
#       showBanner(a);
#       if (sevBase(a.severity) === 'critical') criticalOpen = true;
#     }
#     for (const a of data.alerts) {
#       if (sevBase(a.severity) === 'critical' && acks.indexOf(a.alert_id) < 0) criticalOpen = true;
#     }
#     if (criticalOpen) startLoop(); else stopLoop();

#     if (selectedId && alertsById[selectedId]) {
#       const sel = alertsById[selectedId];
#       const selCard = document.querySelector('.card[data-id="' + selectedId + '"]');
#       if (selCard) selCard.classList.add('selected');
#       if (sel.clip_ready && (video.getAttribute('src') || '') !== '/clips/' + encodeURIComponent(sel.clip_name)) playAlert(selectedId);
#     }
# }

# /* ---------- custom video controls (cosmetic layer over the native <video>) ---------- */
# const playBtn = document.getElementById('playBtn');
# const seekEl = document.getElementById('seek');
# const curTimeEl = document.getElementById('curTime');
# const durTimeEl = document.getElementById('durTime');
# const fsBtn = document.getElementById('fsBtn');
# const noSignalEl = document.getElementById('noSignal');
# let seeking = false;

# function setNoSignal(show) { noSignalEl.classList.toggle('hidden', !show); }
# setNoSignal(true);

# function fmtTime(s) {
#   if (!isFinite(s) || s < 0) return '0:00';
#   s = Math.floor(s);
#   const m = Math.floor(s / 60);
#   const sec = String(s % 60).padStart(2, '0');
#   return m + ':' + sec;
# }

# playBtn.addEventListener('click', () => {
#   if (!video.src) return;
#   if (video.paused) video.play().catch(() => {}); else video.pause();
# });
# video.addEventListener('play', () => { playBtn.textContent = '❚❚'; });
# video.addEventListener('pause', () => { playBtn.textContent = '▶'; });
# video.addEventListener('ended', () => { playBtn.textContent = '▶'; });
# video.addEventListener('loadedmetadata', () => { durTimeEl.textContent = fmtTime(video.duration); });
# video.addEventListener('timeupdate', () => {
#   curTimeEl.textContent = fmtTime(video.currentTime);
#   if (!seeking && isFinite(video.duration) && video.duration > 0) {
#     seekEl.value = String((video.currentTime / video.duration) * 1000);
#   }
# });
# seekEl.addEventListener('input', () => {
#   seeking = true;
#   if (isFinite(video.duration) && video.duration > 0) {
#     curTimeEl.textContent = fmtTime((seekEl.value / 1000) * video.duration);
#   }
# });
# seekEl.addEventListener('change', () => {
#   if (isFinite(video.duration) && video.duration > 0) {
#     video.currentTime = (seekEl.value / 1000) * video.duration;
#   }
#   seeking = false;
# });
# fsBtn.addEventListener('click', () => {
#   const frame = document.querySelector('.video-frame');
#   if (frame.requestFullscreen) frame.requestFullscreen().catch(() => {});
# });

# refresh();
# setInterval(refresh, 1500);
# </script>
# </body></html>
# """


# def _read_alerts(log_path: str, max_rows: int = 500):
#     if not os.path.exists(log_path):
#         return []
#     alerts = []
#     with open(log_path, encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 alerts.append(json.loads(line))
#             except json.JSONDecodeError:
#                 continue  # tolerate a partially-written last line mid-flush
#     return alerts[-max_rows:]


# def _read_acks(acks_path: str):
#     if not os.path.exists(acks_path):
#         return []
#     ids = []
#     with open(acks_path, encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 ids.append(json.loads(line).get("alert_id"))
#             except json.JSONDecodeError:
#                 continue
#     return [i for i in ids if i]


# def _line_count(log_path: str) -> int:
#     if not os.path.exists(log_path):
#         return 0
#     count = 0
#     with open(log_path, encoding="utf-8") as f:
#         for _ in f:
#             count += 1
#     return count


# def _load_recipients(path: str):
#     if not os.path.exists(path):
#         return []
#     try:
#         with open(path, encoding="utf-8") as f:
#             data = json.load(f)
#     except (OSError, json.JSONDecodeError):
#         return []
#     return [r for r in data if r.get("role") and r.get("lat") is not None and r.get("lon") is not None]


# def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
#     """Great-circle distance in kilometers between two lat/lon points."""
#     r = 6371.0
#     p1, p2 = math.radians(lat1), math.radians(lat2)
#     dp = math.radians(lat2 - lat1)
#     dl = math.radians(lon2 - lon1)
#     a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
#     return 2 * r * math.asin(math.sqrt(a))


# # Channel name (from AlertPayload.channels / severity routing) -> recipient role.
# CHANNEL_ROLE = {
#     "traffic_police": "police",
#     "hospital_ems": "hospital",
#     "police_control_room": "control",
# }


# def _route_recipients(alert: dict, recipients: list, nearest_k: int) -> list:
#     """Nearest recipients per role present in the alert's channels. This is
#     the demo's dispatch simulation: the console shows which nearby stations
#     WOULD receive the alert (and later, its clip), matching the severity's
#     channel routing already computed by the pipeline."""
#     lat, lon = alert.get("lat"), alert.get("lon")
#     if not recipients or not lat or not lon:
#         return []
#     channels = alert.get("channels") or []
#     roles = {CHANNEL_ROLE[c] for c in channels if c in CHANNEL_ROLE}
#     out = []
#     for role in sorted(roles):
#         cands = [r for r in recipients if r.get("role") == role]
#         cands.sort(key=lambda r: haversine_km(lat, lon, r["lat"], r["lon"]))
#         for r in cands[:nearest_k]:
#             out.append({
#                 "role": role,
#                 "name": r["name"],
#                 "distance_km": round(haversine_km(lat, lon, r["lat"], r["lon"]), 1),
#             })
#     return out


# def make_handler(log_path: str, clip_dir: str, recipients_path: str,
#                  acks_path: str, nearest_k: int):
#     recipients = _load_recipients(recipients_path)
#     clip_dir = clip_dir or os.path.dirname(os.path.abspath(log_path))

#     class Handler(BaseHTTPRequestHandler):
#         server_version = "VISTA-ControlRoom/1.0"

#         def log_message(self, fmt, *args):
#             pass  # keep the console quiet; this is a demo server, not prod

#         def _send_json(self, obj):
#             body = json.dumps(obj).encode()
#             self.send_response(200)
#             self.send_header("Content-Type", "application/json")
#             self.send_header("Content-Length", str(len(body)))
#             self.end_headers()
#             self.wfile.write(body)

#         def do_GET(self):
#             parsed = urlparse(self.path)
#             if parsed.path.startswith("/api/alerts"):
#                 alerts = _read_alerts(log_path)
#                 acked = _read_acks(acks_path)
#                 out = []
#                 for a in alerts:
#                     rec = dict(a)
#                     rec["routed"] = _route_recipients(rec, recipients, nearest_k)
#                     clip_path = rec.get("clip_path")
#                     rec["clip_ready"] = bool(clip_path and os.path.isfile(clip_path))
#                     rec["clip_name"] = os.path.basename(clip_path) if clip_path else None
#                     out.append(rec)
#                 self._send_json({
#                     "seq": _line_count(log_path),
#                     "count": len(out),
#                     "log_path": os.path.abspath(log_path),
#                     "clip_dir": clip_dir,
#                     "acks": acked,
#                     "alerts": out,
#                 })
#             elif parsed.path.startswith("/clips/"):
#                 self._serve_clip(parsed.path[len("/clips/"):])
#             else:
#                 body = PAGE.encode()
#                 self.send_response(200)
#                 self.send_header("Content-Type", "text/html; charset=utf-8")
#                 self.send_header("Content-Length", str(len(body)))
#                 self.end_headers()
#                 self.wfile.write(body)

#         def do_POST(self):
#             parsed = urlparse(self.path)
#             if parsed.path == "/api/ack":
#                 alert_id = parse_qs(parsed.query).get("alert_id", [""])[0]
#                 if alert_id:
#                     try:
#                         with open(acks_path, "a", encoding="utf-8") as f:
#                             f.write(json.dumps({"alert_id": alert_id, "acked_at": time.time()}) + "\n")
#                     except OSError:
#                         pass
#                 self._send_json({"ok": True, "acked": alert_id})
#             else:
#                 self.send_error(404)

#         def _serve_clip(self, name):
#             """Serve a clip file from the clips dir with Range support so the
#             browser can seek/play HTML5 video reliably. Name is basename-only
#             to prevent path traversal."""
#             safe = os.path.basename(name)
#             path = os.path.join(clip_dir, safe)
#             if not os.path.isfile(path):
#                 self.send_error(404)
#                 return
#             size = os.path.getsize(path)
#             start, end = 0, size - 1
#             range_header = self.headers.get("Range")
#             if range_header and range_header.startswith("bytes="):
#                 spec = range_header[6:].split(",")[0].strip()
#                 if "-" in spec:
#                     s, e = spec.split("-", 1)
#                     if s:
#                         start = int(s)
#                     if e:
#                         end = min(int(e), size - 1)
#                 self.send_response(206)
#                 self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
#             else:
#                 self.send_response(200)
#             self.send_header("Accept-Ranges", "bytes")
#             self.send_header("Content-Type", "video/mp4")
#             self.send_header("Content-Length", str(end - start + 1))
#             self.end_headers()
#             with open(path, "rb") as f:
#                 f.seek(start)
#                 remaining = end - start + 1
#                 while remaining > 0:
#                     chunk = f.read(min(65536, remaining))
#                     if not chunk:
#                         break
#                     self.wfile.write(chunk)
#                     remaining -= len(chunk)

#     return Handler


# def main():
#     ap = argparse.ArgumentParser(description=__doc__,
#                                  formatter_class=argparse.RawDescriptionHelpFormatter)
#     ap.add_argument("--log", default="alerts.jsonl", help="Path to the alerts JSONL log")
#     ap.add_argument("--clips", default="vista_clips",
#                     help="Directory containing per-alert clip files (served under /clips/)")
#     ap.add_argument("--recipients", default="recipients.json",
#                     help="JSON list of nearby hospitals/police stations for routing display")
#     ap.add_argument("--acks", default="acks.jsonl", help="Where ACK actions are appended")
#     ap.add_argument("--nearest-k", type=int, default=1,
#                     help="How many nearest recipients to show per role")
#     ap.add_argument("--port", type=int, default=8787)
#     args = ap.parse_args()

#     log_abs = os.path.abspath(args.log)
#     clip_abs = os.path.abspath(args.clips)
#     server = ThreadingHTTPServer(
#         ("0.0.0.0", args.port),
#         make_handler(log_abs, clip_abs, args.recipients, args.acks, args.nearest_k),
#     )
#     print(f"[control-room] Watching {log_abs} + {clip_abs} at "
#           f"http://localhost:{args.port}  (Ctrl+C to stop)")
#     print("[control-room] The GUI must write to the SAME alerts.jsonl path "
#           "shown above, or no alerts will appear here.")
#     try:
#         server.serve_forever()
#     except KeyboardInterrupt:
#         pass
#     finally:
#         server.server_close()


# if __name__ == "__main__":
#     main()

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
  :root{
    --bg:            #F2F4F7;
    --bg-raised:     #F7F9FC;
    --bg-panel:      #FFFFFF;
    --bg-panel-2:    #E7EBF0;
    --border:        #D9DEE7;
    --border-soft:   #E7EBF0;
    --text:          #172033;
    --text-dim:      #55607A;
    --text-faint:    #94A0B8;
    --slate:         #334E68;
    --red:           #D92D20;
    --red-strong:    #D92D20;
    --red-dim:       #FEE4E2;
    --green:         #16803C;
    --green-dim:     #E6F4EA;
    --amber:         #D97706;
    --orange:        #C2410C;
    --blue:          #2563EB;
    --blue-dim:      #EFF6FF;
    --radius:        12px;
    --radius-sm:     8px;
  }
  * { box-sizing: border-box; }
  html, body { height:100%; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    margin:0;
    -webkit-font-smoothing: antialiased;
    display:flex; flex-direction:column;
  }

  /* ---------- header ---------- */
  header {
    display:flex; align-items:center; justify-content:space-between;
    height:68px; flex:none;
    padding:0 32px;
    background: var(--bg-raised);
    border-bottom:1px solid var(--border);
  }

  .brand { display:flex; align-items:center; gap:14px; flex:none; }
  .brand .name { font-size:19px; font-weight:800; letter-spacing:.4px; color:var(--text); }
  .brand .divider { width:1px; height:22px; background:var(--border); flex:none; }
  .brand .room { font-size:15px; font-weight:600; color:var(--text-dim); white-space:nowrap; }

  .header-actions {
    display:flex; align-items:center;
    gap:28px;
    margin-left:auto;
  }

  .siren-block { display:flex; align-items:center; gap:8px; font-size:13px; color:var(--text-dim); font-weight:600; white-space:nowrap; }
  .siren-block .siren-label { color:var(--text-faint); font-weight:500; }
  #armState { font-size:13px; font-weight:700; letter-spacing:.2px; display:flex; align-items:center; gap:6px; }
  #armState::before { content:''; width:7px; height:7px; border-radius:50%; background:currentColor; }
  #armState.disarmed { color:var(--amber); }
  #armState.armed { color:var(--green); }

  button {
    display:flex; align-items:center; gap:8px;
    background: var(--bg-panel); color:var(--slate);
    border:1px solid var(--border);
    border-radius: var(--radius-sm);
    padding:0 18px; height:38px;
    cursor:pointer; font-size:13px; font-weight:700; letter-spacing:.2px; white-space:nowrap;
    transition: background .15s, border-color .15s, opacity .15s, color .15s;
  }
  button svg { width:15px; height:15px; flex:none; }
  button:hover { background:var(--bg-panel-2); border-color:#C7D0DD; }
  #armBtn { background: var(--red); border-color:var(--red); color:#fff; }
  #armBtn:hover { background:#B7241A; border-color:#B7241A; }
  #armBtn.armed { background:var(--green); border-color:var(--green); color:#fff; }
  #armBtn.armed:hover { background:#116230; border-color:#116230; }
  #testBtn { color:var(--blue); border-color:var(--border); background:var(--bg-panel); }
  #testBtn:hover { background:var(--blue-dim); border-color:#BFDBFE; }
  #testBtn:disabled { opacity:.5; cursor:default; }
  #testBtn:disabled:hover { background:var(--bg-panel); border-color:var(--border); }

  .conn-status { display:flex; align-items:center; padding-left:4px; border-left:1px solid var(--border); }
  .sub {
    color:var(--amber); font-size:12.5px; font-weight:600;
    display:flex; align-items:center; gap:6px; white-space:nowrap; padding-left:20px;
  }
  .sub::before { content:''; width:6px; height:6px; border-radius:50%; background:currentColor; flex:none; }

  /* ---------- layout ---------- */
  main {
    display:grid; grid-template-columns: minmax(0,1fr) 360px; gap:20px;
    padding:20px 28px 28px; align-items:stretch;
    flex:1; min-height:0;
  }
  #viewer { display:flex; flex-direction:column; gap:16px; min-height:0; }

  /* ---------- video card ---------- */
  .video-card {
    background: var(--bg-panel);
    border:1px solid var(--border);
    border-radius: var(--radius);
    overflow:hidden;
    padding:18px 20px 0;
    display:flex; flex-direction:column;
    flex:1; min-height:0;
  }
  .video-card-head {
    display:flex; align-items:center; gap:8px;
    padding:0 0 14px; flex:none;
  }
  .video-card-head .r { width:8px; height:8px; border-radius:50%; background:var(--red); flex:none; }
  .video-card-head h2 {
    margin:0; font-size:15px; font-weight:700; letter-spacing:.3px; color:var(--text);
  }
  .video-frame {
    position:relative; background:#E7EBF0;
    border-radius: 8px; overflow:hidden;
    flex:1; min-height:280px;
  }
  .video-frame video { width:100%; height:100%; display:block; object-fit:contain; background:#E7EBF0; }
  .no-signal {
    position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center;
    gap:12px; color:var(--slate); pointer-events:none;
  }
  .no-signal svg { width:46px; height:46px; color:var(--slate); }
  .no-signal .no-signal-title { font-size:18px; font-weight:700; color:var(--text); }
  .no-signal .no-signal-sub { font-size:13.5px; color:var(--text-dim); }
  .no-signal.hidden { display:none; }

  .video-controls {
    position:absolute; left:0; right:0; bottom:0; z-index:2;
    display:flex; align-items:center; gap:10px;
    padding:10px 16px;
    background: var(--bg-panel);
    border-top:1px solid var(--border);
  }
  .ctrl-btn {
    background:transparent; border:none; color:var(--slate);
    font-size:14px; padding:2px 4px; cursor:pointer; line-height:1;
    height:auto; display:inline-flex;
  }
  .ctrl-btn:hover { color:var(--text); background:transparent; }
  .video-controls .time { font-size:11px; color:var(--text-dim); min-width:32px; text-align:center; font-variant-numeric: tabular-nums; }
  .seek {
    flex:1; -webkit-appearance:none; appearance:none; height:4px; border-radius:2px;
    background: var(--border); outline:none; cursor:pointer;
  }
  .seek::-webkit-slider-thumb {
    -webkit-appearance:none; width:12px; height:12px; border-radius:50%;
    background:var(--slate); cursor:pointer; margin-top:0;
  }
  .seek::-moz-range-thumb { width:12px; height:12px; border-radius:50%; background:var(--slate); border:none; cursor:pointer; }

  /* ---------- selected-alert bar ---------- */
  #selInfo {
    background: var(--bg-panel);
    border:1px solid var(--border);
    border-radius: var(--radius);
    padding:12px 16px;
    font-size:12.5px; line-height:1.55;
    display:flex; align-items:center; gap:10px;
  }
  #selInfo .info-icon {
    width:20px; height:20px; border-radius:50%; background:var(--slate); color:#fff;
    display:flex; align-items:center; justify-content:center; font-size:11px; flex:none;
  }
  #selInfo.muted { color:var(--text-faint); }
  #selInfo .dash { color:var(--text-faint); }
  #selInfo .kind { font-weight:700; text-transform:capitalize; color:var(--text); }
  .sev { font-weight:700; }
  .sev-low { color:var(--green); } .sev-medium { color:var(--amber); }
  .sev-high { color:var(--orange); } .sev-critical { color:var(--red); }
  .muted { color:var(--text-dim); }

  /* ---------- alerts panel ---------- */
  #history {
    background: var(--bg-panel);
    border:1px solid var(--border);
    border-radius: var(--radius);
    padding:18px;
    display:flex; flex-direction:column;
    height:100%;
  }
  .history-head {
    display:flex; align-items:center; justify-content:space-between;
    margin:0 0 14px; padding-bottom:14px;
    border-bottom:1px solid var(--border);
  }
  .history-head h2 { font-size:13px; color:var(--slate); margin:0; letter-spacing:1px; font-weight:700; }
  #alertCount {
    background: var(--bg-panel-2); border:1px solid var(--border); color:var(--text-dim);
    font-size:10.5px; font-weight:700; min-width:18px; height:18px; padding:0 5px;
    border-radius:9px; display:flex; align-items:center; justify-content:center;
  }
  #rows { display:flex; flex-direction:column; gap:8px; overflow-y:auto; flex:1; margin:0 -4px; padding:0 4px; }
  #rows::-webkit-scrollbar { width:6px; }
  #rows::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }

  .card {
    background: var(--bg-panel-2);
    border:1px solid var(--border-soft);
    border-left:3px solid #B7C2D0;
    border-radius: var(--radius-sm);
    padding:10px 12px;
    cursor:pointer; font-size:12.5px;
    transition: border-color .15s, background .15s;
  }
  .card:hover { background:var(--bg-panel); border-color:var(--border); }
  .card.selected { border-color:var(--blue); outline:1px solid var(--blue); }
  .card .top { display:flex; justify-content:space-between; gap:8px; align-items:baseline; }
  .card .kind { font-weight:700; color:var(--text); }
  .card .meta-line { color:var(--text-faint); font-size:11px; margin-top:4px; }
  .card .routed { color:var(--blue); font-size:11px; margin-top:5px; }
  .card .clip-status { font-size:11px; margin-top:5px; color:var(--text-faint); }
  .card .clip-status.ready { color:var(--green); cursor:pointer; text-decoration:underline; }
  .card .ack { margin-top:8px; padding:4px 12px; font-size:10.5px; height:auto; }
  .card .ack.done { background:var(--green-dim); border-color:#BEE3C8; color:var(--green); }

  #empty {
    flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center;
    gap:12px; color:var(--text-faint); text-align:center; padding:30px 10px;
  }
  #empty .empty-icon {
    width:48px; height:48px; border-radius:50%; background:var(--slate);
    display:flex; align-items:center; justify-content:center; color:#fff;
  }
  #empty .empty-icon svg { width:22px; height:22px; }
  #empty .empty-title { font-size:15px; font-weight:700; color:var(--text); }
  #empty .empty-text { font-size:13px; color:var(--text-faint); }

  /* ---------- banner ---------- */
  #banner { position:fixed; top:0; left:0; right:0; z-index:50; transform:translateY(-120%); transition:transform .25s; }
  #banner .inner {
    padding:16px 24px; font-size:15px; font-weight:700; display:flex; gap:20px; align-items:center; flex-wrap:wrap;
    background: var(--bg-panel); border-bottom:1px solid var(--border); color:var(--text);
  }
  #banner.show { transform:translateY(0); }
  body.flash-critical { animation: flashRed 1s ease-in-out 5; }
  @keyframes flashRed { 0%,100% { background:var(--bg); } 50% { background:var(--red-dim); } }
</style></head>
<body>
  <header>
    <div class="brand">
      <span class="name">VISTA</span>
      <span class="divider"></span>
      <span class="room">Control room</span>
    </div>
    <div class="header-actions">
      <div class="siren-block">
        <span class="siren-label">Siren</span>
        <span id="armState" class="disarmed">Disarmed</span>
      </div>
      <button id="armBtn">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M10 21a2 2 0 0 0 4 0"/><path d="M3.2 18h17.6c.6 0 1-.6.7-1.2C20.4 15 19 13 19 9a7 7 0 0 0-14 0c0 4-1.4 6-2.5 7.8-.3.6.1 1.2.7 1.2z"/></svg>
        <span id="armBtnLabel">Arm siren</span>
      </button>
      <button id="testBtn" disabled>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h.01"/><path d="M8.5 16.4a5 5 0 0 1 7 0"/><path d="M5 12.9a10 10 0 0 1 14 0"/><path d="M1.5 9.4a15 15 0 0 1 21 0"/></svg>
        <span>Test</span>
      </button>
      <div class="conn-status">
        <span class="sub" id="meta">connecting…</span>
      </div>
    </div>
  </header>
  <div id="banner"><div class="inner" id="bannerInner"></div></div>
  <main>
    <section id="viewer">
      <div class="video-card">
        <div class="video-card-head">
          <span class="r"></span>
          <h2>Live Feed</h2>
        </div>
        <div class="video-frame">
          <video id="video" muted></video>
          <div id="noSignal" class="no-signal">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M16 8.5h-9a2 2 0 0 0-2 2v5a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2v-5a2 2 0 0 0-2-2z"/><path d="M18 11.5l4-2.3v7.6l-4-2.3"/><path d="M2 2l20 20"/></svg>
            <div class="no-signal-title">No signal</div>
            <div class="no-signal-sub">Camera feed unavailable</div>
          </div>
          <div class="video-controls">
            <button id="playBtn" class="ctrl-btn" type="button">▶</button>
            <span id="curTime" class="time">0:00</span>
            <input id="seek" class="seek" type="range" min="0" max="1000" value="0">
            <span id="durTime" class="time">0:00</span>
            <button id="fsBtn" class="ctrl-btn" type="button">⛶</button>
          </div>
        </div>
      </div>
      <div id="selInfo" class="muted"><span class="info-icon">i</span>No alert selected.</div>
    </section>
    <aside id="history">
      <div class="history-head">
        <h2>DISPATCHED ALERTS</h2>
        <span id="alertCount">0</span>
      </div>
      <div id="rows"></div>
      <div id="empty" style="display:flex">
        <div class="empty-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 8a6 6 0 0 0-12 0c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/></svg></div>
        <div class="empty-title">No active alerts at this time</div>
      </div>
    </aside>
  </main>
<script>
const SEV_RANK = {low:0, medium:1, high:2, critical:3};
const SEV_COLOR = {low:'#16803C', medium:'#D97706', high:'#C2410C', critical:'#D92D20'};
let armed = false, audioCtx = null, sirenLoop = null;
let seenIds = new Set(), selectedId = null;

const armBtn = document.getElementById('armBtn');
const armBtnLabel = document.getElementById('armBtnLabel');
const testBtn = document.getElementById('testBtn');
const armState = document.getElementById('armState');
const video = document.getElementById('video');

armBtn.addEventListener('click', () => {
  if (!audioCtx) audioCtx = new (window.AudioContext||window.webkitAudioContext)();
  audioCtx.resume();
  armed = !armed;
  armBtn.classList.toggle('armed', armed);
  armBtnLabel.textContent = armed ? 'Disarm siren' : 'Arm siren';
  armState.textContent = armed ? 'Armed' : 'Disarmed';
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
  const color = SEV_COLOR[sev] || '#D92D20';
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
  const color = SEV_COLOR[sev] || '#D92D20';
  const time = new Date(a.timestamp*1000).toLocaleTimeString();
  const clip = a.clip_ready
    ? '<div class="clip-status ready" onclick="event.stopPropagation(); playAlert(\'' + a.alert_id + '\')">▶ PLAY CLIP</div>'
    : '<div class="clip-status">waiting for clip…</div>';
  const acked = (acks.indexOf(a.alert_id) >= 0);
  const ackBtn = '<button class="ack' + (acked ? ' done' : '') + '" ' + (acked ? 'disabled' : '') +
    ' onclick="event.stopPropagation(); ack(\'' + a.alert_id + '\')">' +
    (acked ? 'ACKNOWLEDGED' : 'ACK') + '</button>';
  return '<div class="card" style="border-left-color:' + color + '" data-id="' + a.alert_id + '" onclick="select(\'' + a.alert_id + '\')">' +
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
  const color = SEV_COLOR[sev] || '#D92D20';
  const selInfo = document.getElementById('selInfo');
  selInfo.classList.remove('muted');
  selInfo.innerHTML =
    '<span class="info-icon">i</span>' +
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
  setNoSignal(false);
  video.play().catch(() => {});
}
function ack(id) {
  fetch('/api/ack?alert_id=' + encodeURIComponent(id), {method:'POST'}).then(() => refresh());
}

let alertsById = {};

function refresh() {
  fetch('/api/alerts').then(r => r.json()).then(data => {
    document.getElementById('meta').style.color = '#16803C';
    onLine(data);
  }).catch(() => {
    document.getElementById('meta').textContent = '⚠ SERVER OFFLINE — retrying…';
    document.getElementById('meta').style.color = '#D92D20';
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
    document.getElementById('alertCount').textContent = data.count;
    const rows = document.getElementById('rows');
    rows.innerHTML = data.alerts.slice().reverse().map(cardHtml).join('');
    rows.style.display = data.alerts.length ? 'flex' : 'none';
    document.getElementById('empty').style.display = data.alerts.length ? 'none' : 'flex';
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

/* ---------- custom video controls (cosmetic layer over the native <video>) ---------- */
const playBtn = document.getElementById('playBtn');
const seekEl = document.getElementById('seek');
const curTimeEl = document.getElementById('curTime');
const durTimeEl = document.getElementById('durTime');
const fsBtn = document.getElementById('fsBtn');
const noSignalEl = document.getElementById('noSignal');
let seeking = false;

function setNoSignal(show) { noSignalEl.classList.toggle('hidden', !show); }
setNoSignal(true);

function fmtTime(s) {
  if (!isFinite(s) || s < 0) return '0:00';
  s = Math.floor(s);
  const m = Math.floor(s / 60);
  const sec = String(s % 60).padStart(2, '0');
  return m + ':' + sec;
}

playBtn.addEventListener('click', () => {
  if (!video.src) return;
  if (video.paused) video.play().catch(() => {}); else video.pause();
});
video.addEventListener('play', () => { playBtn.textContent = '❚❚'; });
video.addEventListener('pause', () => { playBtn.textContent = '▶'; });
video.addEventListener('ended', () => { playBtn.textContent = '▶'; });
video.addEventListener('loadedmetadata', () => { durTimeEl.textContent = fmtTime(video.duration); });
video.addEventListener('timeupdate', () => {
  curTimeEl.textContent = fmtTime(video.currentTime);
  if (!seeking && isFinite(video.duration) && video.duration > 0) {
    seekEl.value = String((video.currentTime / video.duration) * 1000);
  }
});
seekEl.addEventListener('input', () => {
  seeking = true;
  if (isFinite(video.duration) && video.duration > 0) {
    curTimeEl.textContent = fmtTime((seekEl.value / 1000) * video.duration);
  }
});
seekEl.addEventListener('change', () => {
  if (isFinite(video.duration) && video.duration > 0) {
    video.currentTime = (seekEl.value / 1000) * video.duration;
  }
  seeking = false;
});
fsBtn.addEventListener('click', () => {
  const frame = document.querySelector('.video-frame');
  if (frame.requestFullscreen) frame.requestFullscreen().catch(() => {});
});

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
    print("[control-room] The GUI must write to the SAME alerts.jsonl path "
          "shown above, or no alerts will appear here.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()