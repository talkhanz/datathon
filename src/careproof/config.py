"""Central configuration for CareProof India.

Every numeric weight and threshold below is a *policy choice* made by the
project team, not a learned parameter. They are kept here so they can be
inspected, argued about, and varied in sensitivity analysis (notebook 05).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

# ---------------------------------------------------------------------------
# Source tables (read-only Databricks Marketplace share). Never write to these.
# ---------------------------------------------------------------------------
SOURCE_CATALOG = "databricks_virtue_foundation_dataset_dais_2026"
SOURCE_SCHEMA = "virtue_foundation_dataset"
SOURCE_TABLES = {
    "facilities": f"`{SOURCE_CATALOG}`.`{SOURCE_SCHEMA}`.`facilities`",
    "pincode": f"`{SOURCE_CATALOG}`.`{SOURCE_SCHEMA}`.`india_post_pincode_directory`",
    "nfhs": f"`{SOURCE_CATALOG}`.`{SOURCE_SCHEMA}`.`nfhs_5_district_health_indicators`",
}

# ---------------------------------------------------------------------------
# Project-owned objects. Everything we create starts with this prefix.
# ---------------------------------------------------------------------------
PROJECT_PREFIX = "careproof_"
DEFAULT_TARGET_CATALOG = "workspace"          # Free Edition default catalog
DEFAULT_TARGET_SCHEMA = "careproof_india"
TABLES = {
    "facilities_silver": "careproof_facilities_silver",
    "facilities_quarantine": "careproof_facilities_quarantine",
    "pincode_silver": "careproof_pincode_silver",
    "nfhs_silver": "careproof_nfhs_silver",
    "facility_service_gold": "careproof_facility_service_gold",
    "join_report": "careproof_join_report",
    "eval_results": "careproof_eval_results",
}

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_identifier(name: str, kind: str = "identifier") -> str:
    """Reject anything that is not a plain SQL identifier (prevents injection)."""
    if not isinstance(name, str) or not _IDENT.match(name):
        raise ValueError(f"Invalid {kind}: {name!r}. Use letters, digits and underscores only.")
    return name


def fq_table(catalog: str, schema: str, table: str) -> str:
    """Three-part, back-quoted name for a *project-owned* table."""
    validate_identifier(catalog, "catalog")
    validate_identifier(schema, "schema")
    validate_identifier(table, "table")
    if not table.startswith(PROJECT_PREFIX):
        raise ValueError(f"Refusing to address non-project table {table!r}; names must start with {PROJECT_PREFIX!r}.")
    if catalog == SOURCE_CATALOG:
        raise ValueError("Refusing to target the read-only Marketplace catalog.")
    return f"`{catalog}`.`{schema}`.`{table}`"


# ---------------------------------------------------------------------------
# Geography
# ---------------------------------------------------------------------------
# Generous bounding box for India including Andaman & Nicobar and Lakshadweep.
INDIA_LAT_RANGE = (6.0, 37.6)
INDIA_LON_RANGE = (68.0, 97.5)
PIN_REGEX = r"[1-9][0-9]{5}"

PIN_OUTLIER_KM = 15.0          # offices farther than this from the PIN's median office are treated as outliers
PIN_MIN_INLIER_SHARE = 0.6     # centroid is "defensible" only if >= 60% of geocoded offices are inliers
LOCATION_AGREE_KM = 5.0        # facility coords within this of its PIN centroid = strong agreement
LOCATION_CONFLICT_KM = 25.0    # facility coords farther than this from its PIN centroid = conflict
DEFAULT_MAX_DISTANCE_KM = 100.0
# Distance decay (policy): distance_score = 1 / (1 + d / DISTANCE_HALF_SCORE_KM), i.e. 0.5 at this distance.
# It is deliberately independent of the user's search radius, so changing max_distance_km only filters
# candidates and never reorders the facilities that remain (verified by tests/test_scoring.py).
DISTANCE_HALF_SCORE_KM = 15.0
DEFAULT_TOP_K = 5


# ---------------------------------------------------------------------------
# Scoring policy
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScoringWeights:
    """Heuristic policy weights. They sum to 1.0 so the positive part is in [0, 1]."""
    service_match: float = 0.30
    evidence_confidence: float = 0.25
    location_quality: float = 0.10
    distance: float = 0.25
    readiness: float = 0.10

    def as_dict(self) -> dict:
        return asdict(self)

    def normalized(self) -> "ScoringWeights":
        total = sum(self.as_dict().values())
        if total <= 0:
            raise ValueError("Weights must sum to a positive number")
        return ScoringWeights(**{k: v / total for k, v in self.as_dict().items()})


DEFAULT_WEIGHTS = ScoringWeights()

# Scenarios for sensitivity analysis (notebook 05). All are policy alternatives.
WEIGHT_SCENARIOS = {
    "default": ScoringWeights(),
    "distance_first": ScoringWeights(0.20, 0.15, 0.10, 0.45, 0.10),
    "evidence_first": ScoringWeights(0.30, 0.40, 0.10, 0.10, 0.10),
    "equal": ScoringWeights(0.2, 0.2, 0.2, 0.2, 0.2),
    "no_readiness": ScoringWeights(0.33, 0.28, 0.11, 0.28, 0.0),
}

# Subtractive penalties for data-quality / contradiction flags (policy choices).
PENALTIES = {
    "negated_mention": 0.20,          # text says the service is NOT offered / not available
    "specialty_mismatch": 0.25,       # e.g. an eye or dental clinic claiming dialysis
    "location_conflict": 0.15,        # coordinates disagree with the facility's own PIN by > LOCATION_CONFLICT_KM
    "state_conflict": 0.10,           # stated state not among the states of the stated PIN
    "suspicious_source_url": 0.05,    # source URLs include login / payment pages unrelated to health
    "possible_duplicate": 0.03,       # same normalized name within 1 km of another record
    "coordinates_from_pin": 0.05,     # no usable facility coordinates; PIN centroid used instead
}
MAX_TOTAL_PENALTY = 0.60

# Confidence label thresholds (policy choices).
CONFIDENCE_RULES = {
    "high": {"min_service": 0.9, "min_evidence": 0.7, "max_penalty": 0.05, "min_location": 0.6},
    "low": {"max_evidence": 0.5, "min_penalty": 0.25, "max_location": 0.4},
}

VERIFICATION_WARNING = (
    "Availability, staffing, and hours are NOT verified. This listing is based on web-extracted "
    "text that may be outdated or wrong. Call the facility to confirm the service is available "
    "before travelling. In an emergency, call 108 (ambulance) or 112."
)
DISTANCE_LABEL = "approximate straight-line distance (not road distance or travel time)"


@dataclass
class RunConfig:
    target_catalog: str = DEFAULT_TARGET_CATALOG
    target_schema: str = DEFAULT_TARGET_SCHEMA
    requested_service: str = "dialysis"
    location_type: str = "pincode"
    location_value: str = "110001"
    latitude: str = ""
    longitude: str = ""
    max_distance_km: float = DEFAULT_MAX_DISTANCE_KM
    top_k: int = DEFAULT_TOP_K
    rebuild_tables: bool = False
    weights: ScoringWeights = field(default_factory=ScoringWeights)

    def validate(self) -> "RunConfig":
        validate_identifier(self.target_catalog, "target_catalog")
        validate_identifier(self.target_schema, "target_schema")
        if self.target_catalog == SOURCE_CATALOG:
            raise ValueError("target_catalog must not be the read-only Marketplace catalog")
        if not (0 < float(self.max_distance_km) <= 3000):
            raise ValueError("max_distance_km must be in (0, 3000]")
        if not (1 <= int(self.top_k) <= 50):
            raise ValueError("top_k must be between 1 and 50")
        if self.location_type not in {"pincode", "coordinates", "city", "district"}:
            raise ValueError("location_type must be one of pincode, coordinates, city, district")
        return self
