"""Pure-Python normalization helpers. No Spark, no pandas required.

These functions accept values exactly as they arrive from either the local CSV
(everything is a string; missing is the literal "null") or Databricks (real
NULLs, possibly arrays as Python lists / numpy arrays).
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .config import INDIA_LAT_RANGE, INDIA_LON_RANGE

NULL_TOKENS = {"", "null", "none", "nan", "na", "n/a", "[]", "{}"}
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
PIN_RE = re.compile(r"^[1-9][0-9]{5}$")


# ---------------------------------------------------------------------------
# Generic
# ---------------------------------------------------------------------------
def is_missing(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and x != x:  # NaN
        return True
    if isinstance(x, str) and x.strip().lower() in NULL_TOKENS:
        return True
    return False


def clean_str(x: Any) -> Optional[str]:
    """Trimmed string, or None for null-like tokens. Does not change case."""
    if is_missing(x):
        return None
    s = str(x).strip()
    return None if s.lower() in NULL_TOKENS else s


def parse_float(x: Any) -> Optional[float]:
    s = clean_str(x)
    if s is None:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return None if v != v else v


def parse_int(x: Any) -> Optional[int]:
    v = parse_float(x)
    return int(v) if v is not None and float(v).is_integer() else None


def dedupe_preserve(items: Iterable[str]) -> list[str]:
    seen, out = set(), []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def parse_array(x: Any, dedupe: bool = True) -> list[str]:
    """Parse a JSON array string / list / numpy array into a list of non-empty strings.

    Non-string and empty elements are dropped (observed in the source data).
    Order is preserved; exact repeats are removed when ``dedupe`` is True
    (98% of specialty arrays contain repeats).
    """
    if x is None:
        return []
    if hasattr(x, "tolist") and not isinstance(x, str):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        items = list(x)
    else:
        s = clean_str(x)
        if s is None:
            return []
        try:
            parsed = json.loads(s)
        except (ValueError, TypeError):
            return [s]  # not JSON: treat the whole string as one claim
        if isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, str):
            items = [parsed]
        else:
            return []
    out = [i.strip() for i in items if isinstance(i, str) and i.strip()]
    return dedupe_preserve(out) if dedupe else out


# ---------------------------------------------------------------------------
# Names and places
# ---------------------------------------------------------------------------
def _ascii_fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def normalize_name(x: Any) -> Optional[str]:
    """Lower-case, ASCII-folded, punctuation-free, whitespace-collapsed name for matching."""
    s = clean_str(x)
    if s is None:
        return None
    s = _ascii_fold(s).lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def _place_key(x: Any) -> Optional[str]:
    s = clean_str(x)
    if s is None:
        return None
    s = _ascii_fold(s).upper().replace("&", " AND ")
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


CANONICAL_STATES = {
    "ANDAMAN AND NICOBAR ISLANDS", "ANDHRA PRADESH", "ARUNACHAL PRADESH", "ASSAM", "BIHAR",
    "CHANDIGARH", "CHHATTISGARH", "DELHI", "GOA", "GUJARAT", "HARYANA", "HIMACHAL PRADESH",
    "JAMMU AND KASHMIR", "JHARKHAND", "KARNATAKA", "KERALA", "LADAKH", "LAKSHADWEEP",
    "MADHYA PRADESH", "MAHARASHTRA", "MANIPUR", "MEGHALAYA", "MIZORAM", "NAGALAND", "ODISHA",
    "PUDUCHERRY", "PUNJAB", "RAJASTHAN", "SIKKIM", "TAMIL NADU", "TELANGANA",
    "THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU", "TRIPURA", "UTTAR PRADESH", "UTTARAKHAND",
    "WEST BENGAL",
}
# Observed variants in facilities / NFHS -> canonical (India Post spelling).
STATE_ALIASES = {
    "NCT OF DELHI": "DELHI", "NEW DELHI": "DELHI", "DELHI NCR": "DELHI",
    "MAHARASTRA": "MAHARASHTRA",
    "TAMILNADU": "TAMIL NADU", "TN": "TAMIL NADU",
    "ORISSA": "ODISHA",
    "UP": "UTTAR PRADESH", "U P": "UTTAR PRADESH",
    "MP": "MADHYA PRADESH", "M P": "MADHYA PRADESH",
    "UTTARANCHAL": "UTTARAKHAND",
    "PONDICHERRY": "PUDUCHERRY",
    "J AND K": "JAMMU AND KASHMIR", "JAMMU KASHMIR": "JAMMU AND KASHMIR",
    "ANDAMAN AND NICOBAR": "ANDAMAN AND NICOBAR ISLANDS",
    "DADRA AND NAGAR HAVELI AND DAMAN AND DIU": "THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
    "DADRA AND NAGAR HAVELI": "THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
    "DAMAN AND DIU": "THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
    "CHATTISGARH": "CHHATTISGARH",
    "TELENGANA": "TELANGANA",
}


def normalize_state(x: Any) -> Optional[str]:
    """Canonical state name, or None if the value is missing / not a recognisable state.

    Returns None (rather than guessing) for values such as a city placed in the
    state field ("Mumbai", "Thane") -- those are reported as data-quality issues.
    """
    k = _place_key(x)
    if k is None:
        return None
    k = STATE_ALIASES.get(k, k)
    return k if k in CANONICAL_STATES else None


DISTRICT_ALIASES = {
    "AHMADABAD": "AHMEDABAD",
    "BANGALORE": "BENGALURU URBAN", "BENGALURU": "BENGALURU URBAN", "BANGALORE URBAN": "BENGALURU URBAN",
    "GURGAON": "GURUGRAM",
    "BOMBAY": "MUMBAI",
    "CALCUTTA": "KOLKATA",
    "MADRAS": "CHENNAI",
}


def normalize_district(x: Any) -> Optional[str]:
    k = _place_key(x)
    if k is None:
        return None
    return DISTRICT_ALIASES.get(k, k)


# ---------------------------------------------------------------------------
# PIN codes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PinParse:
    pin: Optional[str]
    issue: Optional[str]  # None when valid


def normalize_pin(x: Any) -> PinParse:
    """Return a six-digit PIN *string* (leading zeroes preserved) or an issue code.

    Accepts '110 002', '110-002', 110002, 110002.0. Never converts to int.
    """
    if is_missing(x):
        return PinParse(None, "missing")
    if isinstance(x, float):
        if not x.is_integer():
            return PinParse(None, "invalid_format")
        s = str(int(x))
    else:
        s = str(x).strip()
    s = re.sub(r"[\s\-]", "", s)
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    if not s.isdigit():
        return PinParse(None, "invalid_format")
    if len(s) != 6:
        return PinParse(None, "wrong_length")
    if not PIN_RE.match(s):
        return PinParse(None, "invalid_first_digit")
    return PinParse(s, None)


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------
_COORD_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*°?\s*[NnEe]?\s*$")


def parse_coordinate(x: Any) -> Optional[float]:
    """Parse '28.43', '28.430354° N', '18.856818°'. Reject DMS / internally spaced numbers."""
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return None if x != x else float(x)
    s = clean_str(x)
    if s is None:
        return None
    m = _COORD_RE.match(s)
    return float(m.group(1)) if m else None


def in_india(lat: Optional[float], lon: Optional[float]) -> bool:
    return (
        lat is not None and lon is not None
        and INDIA_LAT_RANGE[0] <= lat <= INDIA_LAT_RANGE[1]
        and INDIA_LON_RANGE[0] <= lon <= INDIA_LON_RANGE[1]
    )


@dataclass(frozen=True)
class CoordParse:
    lat: Optional[float]
    lon: Optional[float]
    issue: Optional[str]  # None | missing | unparseable | swapped | outside_india


def validate_coordinates(lat_raw: Any, lon_raw: Any) -> CoordParse:
    """Validate a lat/lon pair against India's bounding box. Never auto-corrects.

    Swapped pairs are flagged (issue='swapped') and returned as None, because
    silently swapping would hide a source error.
    """
    if is_missing(lat_raw) or is_missing(lon_raw):
        return CoordParse(None, None, "missing")
    lat, lon = parse_coordinate(lat_raw), parse_coordinate(lon_raw)
    if lat is None or lon is None:
        return CoordParse(None, None, "unparseable")
    if in_india(lat, lon):
        return CoordParse(lat, lon, None)
    if in_india(lon, lat):
        return CoordParse(None, None, "swapped")
    return CoordParse(None, None, "outside_india")


# ---------------------------------------------------------------------------
# NFHS values
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NfhsValue:
    raw: Optional[str]
    value: Optional[float]
    suppressed: bool       # '*' : estimate suppressed (too few unweighted cases)
    small_sample: bool     # '(x)': based on a small number of unweighted cases
    parse_error: bool


def parse_nfhs_value(x: Any) -> NfhsValue:
    raw = None if x is None else str(x)
    s = clean_str(x)
    if s is None:
        return NfhsValue(raw, None, False, False, False)
    if s == "*":
        return NfhsValue(raw, None, True, False, False)
    small = False
    m = re.fullmatch(r"\(\s*(.*?)\s*\)", s)
    if m:
        small, s = True, m.group(1)
    s = s.replace(",", "")
    try:
        return NfhsValue(raw, float(s), False, small, False)
    except ValueError:
        return NfhsValue(raw, None, False, small, True)


def is_valid_uuid(x: Any) -> bool:
    s = clean_str(x)
    return bool(s and UUID_RE.match(s))
