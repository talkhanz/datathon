"""Transparent scoring. All weights/penalties come from config and are policy choices."""
from __future__ import annotations

from typing import Iterable, Mapping, Optional

from .config import (
    CONFIDENCE_RULES, DEFAULT_WEIGHTS, LOCATION_AGREE_KM, LOCATION_CONFLICT_KM, MAX_TOTAL_PENALTY, PENALTIES,
    ScoringWeights,
)

COMPONENTS = ("service_match_score", "evidence_confidence_score", "location_quality_score",
              "distance_score", "readiness_score")
_WEIGHT_KEY = {
    "service_match_score": "service_match",
    "evidence_confidence_score": "evidence_confidence",
    "location_quality_score": "location_quality",
    "distance_score": "distance",
    "readiness_score": "readiness",
}


def clamp01(x: Optional[float]) -> float:
    if x is None or x != x:
        return 0.0
    return max(0.0, min(1.0, float(x)))


def distance_score(distance_km: Optional[float], max_distance_km: float) -> float:
    """Linear decay: 1 at the query point, 0 at max_distance_km. None -> 0."""
    if distance_km is None or max_distance_km <= 0:
        return 0.0
    return clamp01(1.0 - distance_km / max_distance_km)


def location_quality_score(source: str, pin_distance_km: Optional[float], state_conflict: bool) -> float:
    """How much we trust the facility's position.

    source: 'facility_coordinates' | 'pin_centroid' | 'none'
    pin_distance_km: distance between facility coords and its own PIN centroid (None if not checkable)
    """
    if source == "none":
        return 0.0
    if source == "pin_centroid":
        q = 0.4
    elif pin_distance_km is None:
        q = 0.6
    elif pin_distance_km <= LOCATION_AGREE_KM:
        q = 1.0
    elif pin_distance_km <= LOCATION_CONFLICT_KM:
        q = 0.8
    else:
        q = 0.3
    if state_conflict:
        q -= 0.2
    return round(clamp01(q), 4)


def data_quality_penalty(flags: Iterable[str]) -> float:
    total = sum(PENALTIES.get(f, 0.0) for f in set(flags))
    return round(min(MAX_TOTAL_PENALTY, total), 4)


def final_score(components: Mapping[str, float], penalty: float,
                weights: ScoringWeights = DEFAULT_WEIGHTS) -> float:
    w = weights.normalized().as_dict()
    pos = sum(w[_WEIGHT_KEY[c]] * clamp01(components.get(c)) for c in COMPONENTS)
    return round(clamp01(pos - clamp01(penalty)), 4)


def confidence_label(c: Mapping[str, float], penalty: float) -> str:
    hi, lo = CONFIDENCE_RULES["high"], CONFIDENCE_RULES["low"]
    if (penalty >= lo["min_penalty"] or c["evidence_confidence_score"] < lo["max_evidence"]
            or c["location_quality_score"] < lo["max_location"]):
        return "Low"
    if (c["service_match_score"] >= hi["min_service"] and c["evidence_confidence_score"] >= hi["min_evidence"]
            and penalty <= hi["max_penalty"] and c["location_quality_score"] >= hi["min_location"]):
        return "High"
    return "Medium"


def rank_key(r: Mapping):
    """Deterministic ordering: score desc, distance asc, facility_id asc."""
    d = r.get("distance_km")
    return (-round(r["final_score"], 6), d if d is not None else float("inf"), str(r["facility_id"]))


def score_candidate(cand: Mapping, distance_km: float, max_distance_km: float,
                    weights: ScoringWeights = DEFAULT_WEIGHTS) -> dict:
    comps = {
        "service_match_score": clamp01(cand["service_match_score"]),
        "evidence_confidence_score": clamp01(cand["evidence_confidence_score"]),
        "location_quality_score": clamp01(cand["location_quality_score"]),
        "distance_score": round(distance_score(distance_km, max_distance_km), 4),
        "readiness_score": clamp01(cand["readiness_score"]),
    }
    flags = list(cand.get("flags") or [])
    pen = data_quality_penalty(flags)
    return {**comps, "data_quality_penalty": pen, "final_score": final_score(comps, pen, weights),
            "confidence": confidence_label(comps, pen)}
