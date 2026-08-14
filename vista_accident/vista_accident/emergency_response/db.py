"""SQLite persistence for the Emergency Response demo suite.

Three tables, matching the spec exactly:

    authorities   — registered hospitals / police stations / traffic police
                    stations (id, type, name, lat, lon, address, contact)
    incidents     — reported incidents with real captured GPS + timestamp
    notifications — the incident -> nearest-authority routing records
                    generated for every incident

Stdlib-only (sqlite3), consistent with the rest of the project's tooling.
"""

import json
import os
import sqlite3
import time
from contextlib import contextmanager

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(_HERE, "emergency.db")
SEED_AUTHORITIES_PATH = os.path.join(_HERE, "data", "seed_authorities.json")

SCHEMA = """
CREATE TABLE IF NOT EXISTS authorities (
    id       TEXT PRIMARY KEY,
    type     TEXT NOT NULL CHECK(type IN ('hospital', 'police', 'traffic_police')),
    name     TEXT NOT NULL,
    lat      REAL NOT NULL,
    lon      REAL NOT NULL,
    address  TEXT,
    contact  TEXT
);

CREATE TABLE IF NOT EXISTS incidents (
    id             TEXT PRIMARY KEY,
    incident_type  TEXT NOT NULL,
    lat            REAL NOT NULL,
    lon            REAL NOT NULL,
    timestamp      REAL NOT NULL,
    severity       TEXT,
    meta           TEXT
);

CREATE TABLE IF NOT EXISTS notifications (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id     TEXT NOT NULL REFERENCES incidents(id),
    authority_id    TEXT NOT NULL REFERENCES authorities(id),
    authority_type  TEXT NOT NULL,
    distance_km     REAL NOT NULL,
    notified_at     REAL NOT NULL,
    status          TEXT NOT NULL DEFAULT 'notified'
);

CREATE INDEX IF NOT EXISTS idx_notifications_authority
    ON notifications(authority_id, notified_at DESC);
CREATE INDEX IF NOT EXISTS idx_notifications_incident
    ON notifications(incident_id);
"""


@contextmanager
def connect(db_path: str = DEFAULT_DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Create tables if missing and seed authorities on first run only.

    Idempotent — safe to call on every server start.
    """
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        row = conn.execute("SELECT COUNT(*) AS n FROM authorities").fetchone()
        if row["n"] == 0:
            with open(SEED_AUTHORITIES_PATH, "r", encoding="utf-8") as f:
                authorities = json.load(f)
            conn.executemany(
                """INSERT INTO authorities (id, type, name, lat, lon, address, contact)
                   VALUES (:id, :type, :name, :lat, :lon, :address, :contact)""",
                authorities,
            )


def get_authorities(conn, authority_type: str = None):
    if authority_type:
        rows = conn.execute(
            "SELECT * FROM authorities WHERE type = ?", (authority_type,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM authorities").fetchall()
    return [dict(r) for r in rows]


def insert_incident(conn, incident: dict) -> None:
    conn.execute(
        """INSERT INTO incidents (id, incident_type, lat, lon, timestamp, severity, meta)
           VALUES (:id, :incident_type, :lat, :lon, :timestamp, :severity, :meta)""",
        {**incident, "meta": json.dumps(incident.get("meta") or {})},
    )


def insert_notification(conn, incident_id: str, authority: dict, notified_at: float) -> int:
    cur = conn.execute(
        """INSERT INTO notifications
               (incident_id, authority_id, authority_type, distance_km, notified_at, status)
           VALUES (?, ?, ?, ?, ?, 'notified')""",
        (
            incident_id,
            authority["id"],
            authority["type"],
            authority["distance_km"],
            notified_at,
        ),
    )
    return cur.lastrowid


def get_incident(conn, incident_id: str):
    row = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
    if not row:
        return None
    incident = dict(row)
    incident["meta"] = json.loads(incident["meta"] or "{}")
    notifs = conn.execute(
        """SELECT n.*, a.name AS authority_name, a.lat AS authority_lat,
                  a.lon AS authority_lon, a.address AS authority_address,
                  a.contact AS authority_contact
           FROM notifications n JOIN authorities a ON a.id = n.authority_id
           WHERE n.incident_id = ?
           ORDER BY n.distance_km ASC""",
        (incident_id,),
    ).fetchall()
    incident["notifications"] = [dict(r) for r in notifs]
    return incident


def list_incidents_for_authority(conn, authority_type: str, authority_id: str = None,
                                 max_age_s: float = None):
    """Incidents routed to a given authority type (optionally one specific
    authority), newest first — what a dashboard renders. Pass max_age_s to
    only return incidents newer than that many seconds (no limit if None)."""
    params = [authority_type]
    query = """
        SELECT n.id AS notification_id, n.distance_km, n.notified_at, n.status,
               n.authority_id, a.name AS authority_name,
               i.id AS incident_id, i.incident_type, i.lat, i.lon,
               i.timestamp, i.severity, i.meta
        FROM notifications n
        JOIN incidents i ON i.id = n.incident_id
        JOIN authorities a ON a.id = n.authority_id
        WHERE n.authority_type = ?
    """
    if authority_id:
        query += " AND n.authority_id = ?"
        params.append(authority_id)
    if max_age_s:
        params.append(time.time() - max_age_s)
        query += " AND i.timestamp >= ?"
    query += " ORDER BY i.timestamp DESC"
    rows = conn.execute(query, params).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["meta"] = json.loads(d["meta"] or "{}")
        out.append(d)
    return out


def update_notification_status(conn, notification_id: int, status: str) -> bool:
    cur = conn.execute(
        "UPDATE notifications SET status = ? WHERE id = ?", (status, notification_id)
    )
    return cur.rowcount > 0


def clear_incidents(conn) -> int:
    """Delete every incident + its notifications. Returns the number of
    incidents removed (for a demo this is the 'start from a clean slate'
    button)."""
    n = conn.execute("SELECT COUNT(*) AS n FROM incidents").fetchone()["n"]
    conn.execute("DELETE FROM notifications")
    conn.execute("DELETE FROM incidents")
    return n


def new_incident_id(conn) -> str:
    # Random suffix (not a count) so concurrent POSTs under the
    # ThreadingHTTPServer can never collide on the PRIMARY KEY.
    import uuid
    return f"INC-{int(time.time())}-{uuid.uuid4().hex[:6]}"
