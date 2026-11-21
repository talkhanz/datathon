"""Evidence categories, hard eligibility gates, and transparent ranking of ELIGIBLE candidates.

Seven concepts are kept separate (docs/confidence_semantics.md):
service relevance · evidence strength (category) · identity confidence · location confidence ·
readiness evidence · current availability (human-confirmed only, never inferred) · final ranking score.
All thresholds and weights live in config.py and are policy choices.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Optional

from .config import (
    DEFAULT_WEIGHTS, DISTANCE_HALF_SCORE_KM, EVIDENCE_CATEGORIES, EVIDENCE_STRENGTH_BASE, GATES,
    LOCATION_AGREE_KM, LOCATION_CONFLICT_KM, MAX_TOTAL_PENALTY, PENALTIES, ScoringWeights,
)
from .credibility import (
    HIGH, MEDIUM, MISSING_EVIDENCE_EXCERPT, MISSING_SOURCE_PROVENANCE, MULTIPLE_FACILITY_IDENTITIES,
    SERVICE_CONTRADICTION, SPECIALTY_PROFILE_MISMATCH, WEAK_SERVICE_EVIDENCE, finding,
)

COMPONENTS = ("service_relevance_score", "evidence_strength_score", "identity_confidence_score",
              "location_confidence_score", "readiness_evidence_score", "distance_score")
_WEIGHT_KEY = {
    "service_relevance_score": "service_relevance", "evidence_strength_score": "evidence_strength",
    "identity_confidence_score": "identity_confidence", "location_confidence_score": "location_confidence",
    "readiness_evidence_score": "readiness_evidence", "distance_score": "distance",
}


def clamp01(x: Optional[float]) -> float:
    if x is None or x != x:
        return 0.0
    return max(0.0, min(1.0, float(x)))


def distance_score(distance_km: Optional[float], half_score_km: float = DISTANCE_HALF_SCORE_KM) -> float:
    """Hyperbolic decay: 1 at 0 km, 0.5 at ``half_score_km``. Independent of the search radius."""
    if distance_km is None or distance_km != distance_km or half_score_km <= 0:
        return 0.0
    return clamp01(1.0 / (1.0 + max(0.0, float(distance_km)) / half_score_km))


def location_quality_score(source: str, pin_distance_km: Optional[float], state_conflict: bool) -> float:
    """Positional trust before findings: agreement of facility coordinates with its own PIN centroid."""
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


# ---------------------------------------------------------------------------
# Evidence category (docs/evidence_policy.md)
# ---------------------------------------------------------------------------
def evidence_category(match, facility_findings: list) -> tuple[str, list]:
    """Return (category, claim_findings). Category in strong|moderate|weak|conflicting|insufficient."""
    claim = []
    level = match.evidence_level
    if level != "explicit_text":
        if level == "negated_only":
            claim.append(finding(SERVICE_CONTRADICTION, HIGH, _neg_text(match)))
        elif level == "other_facility_only":
            claim.append(finding(MULTIPLE_FACILITY_IDENTITIES, HIGH, "service is attributed to another facility: " + _neg_text(match)))
        elif level == "specialty_tag_only":
            claim.append(finding(WEAK_SERVICE_EVIDENCE, MEDIUM, "only a related specialty tag: " + ", ".join(match.specialty_tags[:3])))
        claim.append(finding(MISSING_EVIDENCE_EXCERPT, MEDIUM, "no attributable sentence states this service"))
        return "insufficient", claim
    if "negated_mention" in match.conflicts:
        claim.append(finding(SERVICE_CONTRADICTION, HIGH, _neg_text(match, kinds=("negated", "planned", "third_party_or_historical"))))
    if "other_facility_attribution" in match.conflicts:
        claim.append(finding(MULTIPLE_FACILITY_IDENTITIES, HIGH, "some passages attribute the service to another facility: "
                             + _neg_text(match, kinds=("other_facility",))))
    if "specialty_mismatch" in match.conflicts:
        claim.append(finding(SPECIALTY_PROFILE_MISMATCH, HIGH, "facility name/type indicates a different specialty"))
    high_facility = [f for f in facility_findings if f["severity"] == HIGH]
    if claim or high_facility:
        return "conflicting", claim
    medium_facility = [f for f in facility_findings if f["severity"] == MEDIUM]
    provenance_ok = not any(f["code"] == MISSING_SOURCE_PROVENANCE for f in facility_findings)
    structured = [f for f in match.matched_fields if f not in ("description", "name", "specialties")]
    corroborated = bool(match.specialty_tags) or len(structured) >= 1
    if (match.has_description_statement and match.n_distinct_statements >= 2 and corroborated
            and not medium_facility and provenance_ok):
        return "strong", claim
    if match.has_description_statement or match.n_distinct_statements >= 2:
        return "moderate", claim
    claim.append(finding(WEAK_SERVICE_EVIDENCE, LOW_SEV, "service stated once, in a single structured field"))
    return "weak", claim


LOW_SEV = "low"


def _neg_text(match, kinds=None) -> str:
    items = [e for e in match.negated_excerpts if kinds is None or any(e.get("kind", "").startswith(k) for k in kinds)]
    return "; ".join(f'{e["field"]}: "{e["text"][:140]}"' for e in items[:2])


def evidence_strength_score(category: str, corroboration: float) -> float:
    """Category base (dominant) + a small continuous corroboration term for tie-breaking within a category."""
    base = EVIDENCE_STRENGTH_BASE[category]
    return round(clamp01(base + (0.1 * clamp01(corroboration) if base > 0 else 0.0)), 4)


# ---------------------------------------------------------------------------
# Hard eligibility gates
# ---------------------------------------------------------------------------
def eligibility(category: str, excerpt: Optional[str], identity_conf: float, location_conf: float,
                facility_findings: list, claim_findings: list, provenance: Mapping, gates: Mapping = GATES) -> list:
    """Return the list of FAILED gates (empty list = eligible)."""
    failed = []
    rank = EVIDENCE_CATEGORIES.index
    if category in ("conflicting", "insufficient") or rank(category) < rank(gates["min_evidence_category"]):
        failed.append(f"evidence_{category}_below_minimum_{gates['min_evidence_category']}")
    if not (excerpt or "").strip():
        failed.append("no_supporting_excerpt")
    if location_conf < gates["min_location_confidence"]:
        failed.append("location_confidence_below_minimum")
    if identity_conf < gates["min_identity_confidence"]:
        failed.append("identity_confidence_below_minimum")
    if any(f["severity"] == HIGH for f in facility_findings + claim_findings):
        failed.append("high_severity_finding")
    if any(f["code"] == MULTIPLE_FACILITY_IDENTITIES for f in facility_findings + claim_findings):
        failed.append("probable_merged_or_misattributed_record")
    if not all(provenance.get(k) for k in ("facility_id", "source_row_id", "evidence_field")):
        failed.append("missing_provenance_fields")
    return failed


# ---------------------------------------------------------------------------
# Ranking (eligible candidates only)
# ---------------------------------------------------------------------------
def data_quality_penalty(findings: Iterable[Mapping]) -> float:
    total = sum(PENALTIES.get(f["code"], 0.0) for f in findings if f["severity"] == "low")
    return round(min(MAX_TOTAL_PENALTY, total), 4)


def final_ranking_score(components: Mapping[str, float], penalty: float,
                        weights: ScoringWeights = DEFAULT_WEIGHTS) -> float:
    w = weights.normalized().as_dict()
    pos = sum(w[_WEIGHT_KEY[c]] * clamp01(components.get(c)) for c in COMPONENTS)
    return round(clamp01(pos - clamp01(penalty)), 4)


def rank_key(r: Mapping):
    """Deterministic ordering: score desc, distance asc, facility_id asc."""
    d = r.get("distance_km")
    return (-round(r["final_ranking_score"], 6), d if d is not None else float("inf"), str(r["facility_id"]))


def score_candidate(cand: Mapping, distance_km: float, weights: ScoringWeights = DEFAULT_WEIGHTS) -> dict:
    comps = {
        "service_relevance_score": clamp01(cand["service_relevance_score"]),
        "evidence_strength_score": clamp01(cand["evidence_strength_score"]),
        "identity_confidence_score": clamp01(cand["identity_confidence_score"]),
        "location_confidence_score": clamp01(cand["location_confidence_score"]),
        "readiness_evidence_score": clamp01(cand["readiness_evidence_score"]),
        "distance_score": round(distance_score(distance_km), 4),
    }
    pen = data_quality_penalty(cand.get("findings") or [])
    return {**comps, "data_quality_penalty": pen, "final_ranking_score": final_ranking_score(comps, pen, weights)}
