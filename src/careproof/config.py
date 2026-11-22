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
    "facilities_rejected_rows": "careproof_facilities_rejected_rows",
    "pincode_silver": "careproof_pincode_silver",
    "nfhs_silver": "careproof_nfhs_silver",
    "facility_service_gold": "careproof_facility_service_gold",
    "join_report": "careproof_join_report",
    "eval_results": "careproof_eval_results",
    # supervised service-profile model + GA (notebook 07 / scripts/run_local.py model)
    "model_scores": "careproof_model_scores",
    "model_feature_selection": "careproof_model_feature_selection",
    "model_metrics": "careproof_model_metrics",
    "model_ga_history": "careproof_model_ga_history",
    "model_optimal_profile": "careproof_model_optimal_profile",
    "model_profile_summary": "careproof_model_profile_summary",
    "model_calibration": "careproof_model_calibration",
    "model_features": "careproof_model_features",
    "model_facilities": "careproof_model_facilities",
    # independent reference data (notebook 08 / scripts/run_local.py external)
    "external_reference": "careproof_ext_pmjay_hospitals",
    "external_matches": "careproof_external_matches",
    "external_validation": "careproof_external_validation",
    "external_validation_detail": "careproof_external_validation_detail",
}
EXTERNAL_TABLES = ("external_matches", "external_validation", "external_validation_detail")
MODEL_TABLES = ("model_scores", "model_feature_selection", "model_metrics", "model_ga_history", "model_optimal_profile",
                "model_profile_summary", "model_calibration", "model_features", "model_facilities")

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
    """Ranking weights for ELIGIBLE candidates only. Policy choices, not learned. Normalised to sum to 1."""
    # Default = Bayesian-optimised (Optuna TPE, 150 trials on the dev states, safety constraints enforced) and
    # rounded; it dominated the hand-set policy on all three label-free KPIs on held-out test states, with small
    # effects (docs/policy_optimization.md). Still a policy choice: these KPIs are not ranking correctness.
    service_relevance: float = 0.05
    evidence_strength: float = 0.35
    identity_confidence: float = 0.14
    location_confidence: float = 0.20
    readiness_evidence: float = 0.02
    distance: float = 0.24

    def as_dict(self) -> dict:
        return asdict(self)

    def normalized(self) -> "ScoringWeights":
        total = sum(self.as_dict().values())
        if total <= 0:
            raise ValueError("Weights must sum to a positive number")
        return ScoringWeights(**{k: v / total for k, v in self.as_dict().items()})


DEFAULT_WEIGHTS = ScoringWeights()

# Sensitivity scenarios (notebook 05). All are policy alternatives.
WEIGHT_SCENARIOS = {
    "default": ScoringWeights(),
    "hand_set_v1": ScoringWeights(0.15, 0.30, 0.15, 0.10, 0.10, 0.20),
    "distance_first": ScoringWeights(0.10, 0.20, 0.10, 0.10, 0.05, 0.45),
    "evidence_first": ScoringWeights(0.15, 0.45, 0.15, 0.10, 0.05, 0.10),
    "equal": ScoringWeights(1, 1, 1, 1, 1, 1),
    "no_readiness": ScoringWeights(0.15, 0.30, 0.15, 0.10, 0.0, 0.20),
}

# ---------------------------------------------------------------------------
# Evidence policy and hard eligibility gates (docs/evidence_policy.md). Configurable, tested.
# ---------------------------------------------------------------------------
EVIDENCE_CATEGORIES = ("insufficient", "conflicting", "weak", "moderate", "strong")
EVIDENCE_LABELS = {
    "strong": "Strong documented evidence",
    "moderate": "Moderate documented evidence",
    "weak": "Weak documented evidence",
    "conflicting": "Conflicting evidence",
    "insufficient": "Insufficient evidence",
}
# Base evidence_strength_score by category: separated so a nearby weaker claim cannot overtake a slightly
# farther stronger one (tested: weak@0 km never beats moderate@10 km with otherwise equal components).
EVIDENCE_STRENGTH_BASE = {"strong": 0.9, "moderate": 0.6, "weak": 0.25, "conflicting": 0.0, "insufficient": 0.0}
GATES = {
    "min_evidence_category": "weak",       # weak | moderate | strong  (conflicting/insufficient never pass)
    "min_location_confidence": 0.6,        # rejects coordinate/PIN conflicts and PIN-centroid-only positions
    "min_identity_confidence": 0.5,        # rejects records with a high-severity identity finding
}

# Low-severity issues that remain eligible but cost a little rank (policy).
PENALTIES = {
    "POSSIBLE_DUPLICATE": 0.03,
    "SUSPICIOUS_SOURCE_URL": 0.05,
    "AMBIGUOUS_PIN": 0.02,
}
MAX_TOTAL_PENALTY = 0.30

VERIFICATION_WARNING = (
    "CareProof is a referral planning and facility-discovery tool. Public facility information may be incomplete "
    "or outdated. Contact the receiving facility to confirm service and admission availability before travel. "
    "CareProof does not provide medical advice or emergency dispatch."
)
AVAILABILITY_NOT_VERIFIED = "Current availability not verified"
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
