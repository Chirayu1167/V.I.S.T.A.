"""
VISTA — Emergency Response demo suite.

Additive extension of the existing VISTA accident-detection project. This
package does not touch the ML pipeline (`vista_accident/`) at all — it adds
three browser dashboards (Traffic Police / Police / Hospital) plus a citizen
"report incident" page, all backed by a small stdlib-only HTTP+SQLite
service, so an incident's *real* GPS location (captured via the browser
Geolocation API) can be routed to the nearest registered authorities using
an actual Haversine distance calculation.

Run with:
    python -m vista_accident.emergency_response.server
"""
