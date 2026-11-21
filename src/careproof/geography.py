"""Geography: Haversine distance, PIN-level aggregation, and query-location resolution."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median
from typing import Iterable, Mapping, Optional, Sequence

from .config import PIN_MIN_INLIER_SHARE, PIN_OUTLIER_KM
from .normalization import (
    clean_str, in_india, normalize_district, normalize_pin, normalize_state, parse_coordinate,
    validate_coordinates,
)

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle (straight-line) distance in km. Not road distance or travel time."""
    for v in (lat1, lon1, lat2, lon2):
        if v is None:
            raise ValueError("haversine_km requires four non-null coordinates")
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


# ---------------------------------------------------------------------------
# PIN aggregation (post-office grain -> PIN grain)
# ---------------------------------------------------------------------------
@dataclass
class PinSummary:
    pincode: str
    post_office_count: int
    districts: list[str]
    states: list[str]
    n_offices_with_valid_coords: int
    n_offices_coord_outliers: int
    centroid_lat: Optional[float]
    centroid_lon: Optional[float]
    spread_km: Optional[float]
    coords_defensible: bool
    is_ambiguous: bool
    missing_coordinates: bool
    coord_issue_counts: dict = field(default_factory=dict)

    @property
    def primary_district(self) -> Optional[str]:
        return self.districts[0] if len(self.districts) == 1 else None

    @property
    def primary_state(self) -> Optional[str]:
        return self.states[0] if len(self.states) == 1 else None


def aggregate_pin(pincode: str, offices: Sequence[Mapping]) -> PinSummary:
    """Collapse post-office rows of one PIN into one defensible PIN record.

    Centroid = coordinate-wise median of in-India office coordinates after
    discarding offices > PIN_OUTLIER_KM from a first-pass median. The centroid
    is defensible only when at least PIN_MIN_INLIER_SHARE of the geocoded
    offices are inliers (observed: most offices of a PIN lie within ~3 km, with
    a minority of mis-geocoded strays tens of km away).
    """
    districts = sorted({d for d in (normalize_district(o.get("district")) for o in offices) if d})
    states = sorted({s for s in (normalize_state(o.get("statename")) for o in offices) if s})
    pts, issues = [], {}
    for o in offices:
        c = validate_coordinates(o.get("latitude"), o.get("longitude"))
        if c.issue:
            issues[c.issue] = issues.get(c.issue, 0) + 1
        else:
            pts.append((c.lat, c.lon))
    clat = clon = spread = None
    n_out = 0
    defensible = False
    if pts:
        mlat, mlon = median(p[0] for p in pts), median(p[1] for p in pts)
        kept = [p for p in pts if haversine_km(mlat, mlon, p[0], p[1]) <= PIN_OUTLIER_KM]
        n_out = len(pts) - len(kept)
        if kept:
            clat, clon = median(p[0] for p in kept), median(p[1] for p in kept)
            spread = max(haversine_km(clat, clon, p[0], p[1]) for p in kept)
            defensible = len(kept) / len(pts) >= PIN_MIN_INLIER_SHARE
    return PinSummary(
        pincode=pincode,
        post_office_count=len(offices),
        districts=districts,
        states=states,
        n_offices_with_valid_coords=len(pts),
        n_offices_coord_outliers=n_out,
        centroid_lat=clat if defensible else None,
        centroid_lon=clon if defensible else None,
        spread_km=round(spread, 3) if spread is not None else None,
        coords_defensible=defensible,
        is_ambiguous=len(districts) > 1 or len(states) > 1,
        missing_coordinates=not pts,
        coord_issue_counts=issues,
    )


# ---------------------------------------------------------------------------
# Query location resolution
# ---------------------------------------------------------------------------
class LocationError(ValueError):
    """Raised when a location cannot be resolved reliably. Message is user-facing."""


@dataclass
class ResolvedLocation:
    lat: float
    lon: float
    label: str
    method: str
    district: Optional[str] = None
    state: Optional[str] = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class PlaceIndex:
    """Lookup structures built from the PIN silver table."""
    pins: dict  # pincode -> PinSummary-like mapping
    districts: dict  # DISTRICT -> list of {state, lat, lon, n_pins}

    @classmethod
    def from_pin_rows(cls, rows: Iterable[Mapping]) -> "PlaceIndex":
        pins, acc = {}, {}
        for r in rows:
            pins[r["pincode"]] = r
            if r.get("coords_defensible") and not r.get("is_ambiguous"):
                d, s = r.get("primary_district"), r.get("primary_state")
                if d and s:
                    acc.setdefault((d, s), []).append((r["centroid_lat"], r["centroid_lon"]))
        districts = {}
        for (d, s), pts in acc.items():
            districts.setdefault(d, []).append({
                "state": s, "lat": median(p[0] for p in pts), "lon": median(p[1] for p in pts), "n_pins": len(pts),
            })
        return cls(pins=pins, districts=districts)


def _split_place(value: str) -> tuple[Optional[str], Optional[str]]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        return None, None
    state = normalize_state(parts[-1]) if len(parts) > 1 else None
    return normalize_district(parts[0]), state


def resolve_location(location_type: str, location_value: str = "", latitude=None, longitude=None,
                     index: Optional[PlaceIndex] = None) -> ResolvedLocation:
    lt = (location_type or "").strip().lower()
    if lt == "coordinates":
        lat, lon = parse_coordinate(latitude), parse_coordinate(longitude)
        if (lat is None or lon is None) and location_value and "," in location_value:
            a, b = location_value.split(",", 1)
            lat, lon = parse_coordinate(a), parse_coordinate(b)
        if lat is None or lon is None:
            raise LocationError("Enter latitude and longitude as decimal degrees, e.g. 28.6139 and 77.2090.")
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise LocationError(f"Coordinates ({lat}, {lon}) are outside valid latitude/longitude ranges.")
        if not in_india(lat, lon):
            hint = " They look swapped - try exchanging latitude and longitude." if in_india(lon, lat) else ""
            raise LocationError(f"Coordinates ({lat}, {lon}) are outside India.{hint}")
        return ResolvedLocation(lat, lon, f"{lat:.5f}, {lon:.5f}", "user_coordinates")

    if index is None:
        raise LocationError("PIN/city/district lookup requires the PIN reference table.")

    if lt == "pincode":
        p = normalize_pin(location_value)
        if p.pin is None:
            raise LocationError(f"'{location_value}' is not a valid Indian PIN code (six digits, first digit 1-9).")
        r = index.pins.get(p.pin)
        if r is None:
            raise LocationError(f"PIN {p.pin} is not in the India Post directory. Try a nearby PIN or coordinates.")
        warnings = []
        if r.get("is_ambiguous"):
            warnings.append(
                f"PIN {p.pin} spans multiple districts/states ({'; '.join(r.get('districts') or [])} | "
                f"{'; '.join(r.get('states') or [])}). Distances use the PIN's central post-office location; "
                "enter coordinates for precision.")
        if not r.get("coords_defensible"):
            n_ok = r.get('n_offices_with_valid_coords', 0) or 0
            n_out = r.get('n_offices_coord_outliers', 0) or 0
            n_po = r.get('post_office_count', 0) or 0
            why = (f"none of its {n_po} post offices have valid coordinates" if n_ok == 0 else
                   f"its {n_ok} geocoded post offices disagree ({n_out} lie more than 15 km from the median)")
            raise LocationError(f"PIN {p.pin} has no reliable location in the India Post directory: {why}. "
                                "Please enter latitude/longitude instead.")
        return ResolvedLocation(r["centroid_lat"], r["centroid_lon"], f"PIN {p.pin}", "pin_centroid",
                                r.get("primary_district"), r.get("primary_state"), warnings)

    if lt in {"district", "city"}:
        d, s = _split_place(location_value or "")
        if not d:
            raise LocationError("Enter a district or city name, optionally with the state, e.g. 'Patna, Bihar'.")
        cands = index.districts.get(d, [])
        if s:
            cands = [c for c in cands if c["state"] == s]
        if not cands:
            raise LocationError(
                f"Could not find '{location_value}' as an India Post district. Cities that are not district names "
                "cannot be resolved reliably - please use a PIN code or coordinates.")
        if len(cands) > 1:
            states = ", ".join(sorted(c["state"] for c in cands))
            raise LocationError(f"'{location_value}' matches districts in several states ({states}). "
                                f"Add the state, e.g. '{location_value}, {cands[0]['state'].title()}'.")
        c = cands[0]
        warn = [f"Using the median location of {c['n_pins']} PIN areas in {d.title()} district; "
                "distances are approximate. Use a PIN or coordinates for precision."]
        if lt == "city":
            warn.append("City was matched to the district of the same name.")
        return ResolvedLocation(c["lat"], c["lon"], f"{d.title()}, {c['state'].title()}", "district_centroid",
                                d, c["state"], warn)

    raise LocationError("location_type must be one of: pincode, coordinates, city, district.")


_LATLON_RX = __import__("re").compile(r"^\s*(-?\d{1,2}(?:\.\d+)?)\s*[, ]\s*(-?\d{1,3}(?:\.\d+)?)\s*$")


def resolve_free_text(text: str, index: PlaceIndex) -> ResolvedLocation:
    """One input box: a 6-digit PIN, 'lat, lon', or a district/city name (optionally ', State')."""
    t = (text or "").strip()
    if not t:
        raise LocationError("Enter a PIN code (e.g. 800001), a district (e.g. Patna, Bihar) or coordinates (e.g. 25.61, 85.14).")
    digits = t.replace(" ", "")
    if digits.isdigit():
        return resolve_location("pincode", digits, index=index)
    m = _LATLON_RX.match(t)
    if m:
        return resolve_location("coordinates", "", m.group(1), m.group(2), index)
    return resolve_location("district", t, index=index)
