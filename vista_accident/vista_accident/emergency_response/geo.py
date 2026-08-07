"""Geographical distance helper — Haversine formula.

Deliberately dependency-free (matches the rest of the project's
stdlib-only tooling). Used to rank registered authorities by real
distance from an incident's captured GPS coordinates.
"""

import math

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in kilometers."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def nearest(lat: float, lon: float, candidates, limit: int = 3):
    """Return the `limit` closest candidates to (lat, lon).

    `candidates` is an iterable of dicts each containing 'lat' and 'lon'
    keys. Returns the input dicts, each augmented with a 'distance_km' key,
    sorted ascending by distance.
    """
    ranked = []
    for c in candidates:
        d = haversine_km(lat, lon, c["lat"], c["lon"])
        ranked.append({**c, "distance_km": round(d, 3)})
    ranked.sort(key=lambda c: c["distance_km"])
    return ranked[:limit]
