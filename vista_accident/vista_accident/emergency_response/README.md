# VISTA — Emergency Response Demo Suite

Additive extension of the existing VISTA accident-detection project.
Nothing under `vista_accident/` (the ML pipeline) was changed — this is a
new, self-contained module that adds three authority dashboards and a
citizen incident-report page on top of the same demo.

## What this adds

- **`/report`** — a page that requests the browser's real GPS location
  (`navigator.geolocation`, no hardcoded/random coordinates), lets you pick
  an incident type and severity, and reports it.
- **`/dashboard/hospital`**, **`/dashboard/police`**, **`/dashboard/traffic_police`**
  — three separate live dashboards, each showing the incidents routed to
  that authority type, with GPS coordinates, distance, severity, timestamp,
  a status control (notified → acknowledged → dispatched → resolved), and
  an interactive Leaflet/OpenStreetMap map.
- A small stdlib-only HTTP + SQLite backend (`server.py`, `db.py`, `geo.py`)
  that:
  1. Stores every incident with its real captured lat/lon and timestamp.
  2. Calculates the distance from the incident to **every** registered
     authority using the **Haversine formula** (`geo.py`) — nothing is
     hardcoded.
  3. Notifies the nearest **3 hospitals**, **3 police stations**, and
     **3 traffic police stations**.
  4. Persists incidents, authorities, and notifications in
     `emergency_response/emergency.db` (SQLite).

## Run it

```bash
cd vista_accident
python -m vista_accident.emergency_response.server --port 8890
```

Then open:
- `http://localhost:8890/` — home / links to everything
- `http://localhost:8890/report` — report an incident (grant location
  permission when prompted)
- `http://localhost:8890/dashboard/hospital`
- `http://localhost:8890/dashboard/police`
- `http://localhost:8890/dashboard/traffic_police`

Open a dashboard in one tab and `/report` in another to watch an incident
appear on the dashboard within a few seconds (dashboards poll every 8s).

## Data model (SQLite)

- `authorities(id, type, name, lat, lon, address, contact)` — seeded on
  first run from `data/seed_authorities.json` (5 hospitals, 5 police
  stations, 5 traffic police stations around Indore, MP — same city as the
  existing `recipients.json` demo data).
- `incidents(id, incident_type, lat, lon, timestamp, severity, meta)`
- `notifications(id, incident_id, authority_id, authority_type,
  distance_km, notified_at, status)`

## API

- `GET  /api/authorities?type=hospital|police|traffic_police`
- `POST /api/incidents` — body `{incident_type, severity, lat, lon}` →
  stores the incident, computes nearest 3 per authority type, creates
  notifications, returns the incident with all notifications attached.
- `GET  /api/incidents?authority_type=hospital&authority_id=HOSP-01` —
  incidents routed to a given authority (or all authorities of that type
  if `authority_id` is omitted) — this is what each dashboard polls.
- `GET  /api/incidents/<id>` — one incident with all its notifications.
- `PATCH /api/notifications/<id>` — body `{status}` — update a
  notification's status.

## Notes

- Distances are real great-circle distances (Haversine), computed against
  the live authority table — not a lookup table or a hardcoded "nearest 3".
- The map tiles come from the public OpenStreetMap tile server and the
  Leaflet library from a CDN, loaded client-side by the browser; no API key
  is required.
- This module is independent of `AccidentPipeline` / `gui_app.py` / the
  ML detection pipeline. It's meant to demonstrate the citizen-report →
  nearest-authority → dashboard workflow end to end; wiring an actual
  confirmed ML alert (`alert.py`'s `AlertPayload`) into `POST
  /api/incidents` instead of manual browser reporting is a small, separate
  follow-up (map `AlertPayload.lat/lon/timestamp/kind/severity` onto the
  same endpoint).
