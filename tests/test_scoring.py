import pytest

from careproof.config import GATES, MAX_TOTAL_PENALTY, PENALTIES, ScoringWeights
from careproof.credibility import HIGH, LOW, MEDIUM, finding
from careproof.scoring import (
    COMPONENTS, data_quality_penalty, distance_score, eligibility, evidence_category, evidence_strength_score,
    final_ranking_score, location_quality_score, rank_key, score_candidate,
)
from careproof.service_matching import match_service

BASE = {"name": "City Hospital", "description": None, "procedure": "null", "equipment": "null",
        "capability": "null", "specialties": "null", "facilityTypeId": "hospital", "address_city": "Patna",
        "address_stateOrRegion": "Bihar"}


def m(**kw):
    return match_service({**BASE, **kw}, "dialysis")


GOOD_CAND = {"service_relevance_score": 1.0, "evidence_strength_score": 0.95, "identity_confidence_score": 1.0,
             "location_confidence_score": 1.0, "readiness_evidence_score": 1.0, "findings": []}
PROV = {"facility_id": "f", "source_row_id": "r", "evidence_field": "description"}


# ---- evidence categories -------------------------------------------------------------------------------------
def test_strong_evidence():
    x = m(description="City Hospital runs a 10-bed dialysis unit.", procedure='["Haemodialysis"]', specialties='["nephrology"]')
    assert evidence_category(x, [])[0] == "strong"


def test_strong_requires_clean_facility():
    x = m(description="City Hospital runs a 10-bed dialysis unit.", procedure='["Haemodialysis"]', specialties='["nephrology"]')
    assert evidence_category(x, [finding("SHARED_PLACEHOLDER_COORDINATE", MEDIUM, "x")])[0] == "moderate"


def test_moderate_evidence_two_structured_statements():
    x = m(procedure='["Hemodialysis with 6 machines"]', capability='["Dialysis unit open 24x7"]')
    assert evidence_category(x, [])[0] == "moderate"


def test_bare_term_contained_in_longer_statement_is_not_extra_corroboration():
    x = m(procedure='["Dialysis"]', capability='["Provides haemodialysis for CKD patients"]')
    assert x.n_distinct_statements == 1 and evidence_category(x, [])[0] == "weak"


def test_weak_evidence_single_structured_field():
    x = m(procedure='["Dialysis"]')
    cat, claim = evidence_category(x, [])
    assert cat == "weak" and any(f["code"] == "WEAK_SERVICE_EVIDENCE" for f in claim)


def test_conflicting_when_negated_alongside_positive():
    x = m(procedure='["Dialysis"]', capability='["Dialysis not available"]')
    cat, claim = evidence_category(x, [])
    assert cat == "conflicting" and any(f["code"] == "SERVICE_CONTRADICTION" for f in claim)


def test_conflicting_when_facility_has_high_finding():
    x = m(procedure='["Dialysis"]', capability='["Dialysis unit"]')
    assert evidence_category(x, [finding("COORDINATE_LOCATION_CONFLICT", HIGH, "x")])[0] == "conflicting"


def test_insufficient_when_tag_only_or_missing_excerpt():
    x = m(specialties='["nephrology"]')
    cat, claim = evidence_category(x, [])
    assert cat == "insufficient" and {f["code"] for f in claim} >= {"WEAK_SERVICE_EVIDENCE", "MISSING_EVIDENCE_EXCERPT"}


# ---- attribution / non-current language ------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "High-risk cases requiring dialysis are referred to Kalpana Chawla Government Medical College",
    "Dialysis available through tie-up with MIMS Hospital Calicut",
    "Dialysis was formerly offered here",
    "Dialysis services discontinued in 2019",
    "Proposed dialysis unit",
])
def test_non_current_or_third_party_is_not_support(text):
    x = m(capability=f'["{text}"]')
    assert not x.is_candidate


def test_other_facility_attribution():
    x = match_service({**BASE, "name": "Randeep Hospital",
                       "procedure": '["Dialysis services are provided at Amandeep Hospital, Pathankot"]'}, "dialysis")
    assert x.evidence_level == "other_facility_only" and not x.is_candidate


def test_hosted_unit_at_own_facility_is_attributable():
    x = match_service({**BASE, "name": "Upadhyay Hospital",
                       "procedure": '["Operates as NephroPlus Dialysis Clinic at Upadhyay Hospital in Agra"]'}, "dialysis")
    assert x.is_candidate


def test_spelling_variant_is_same_entity():
    x = match_service({**BASE, "name": "Udyan Hospital",
                       "description": "Udayan Hospital is a multispeciality hospital with a dialysis unit."}, "dialysis")
    assert x.is_candidate


# ---- hard gates ---------------------------------------------------------------------------------------------------
def test_gates_pass_for_clean_candidate():
    assert eligibility("moderate", "Dialysis unit", 1.0, 1.0, [], [], PROV) == []


@pytest.mark.parametrize("kw,gate", [
    (dict(category="conflicting"), "evidence_conflicting"),
    (dict(category="insufficient"), "evidence_insufficient"),
    (dict(excerpt=""), "no_supporting_excerpt"),
    (dict(location=0.3), "location_confidence_below_minimum"),
    (dict(identity=0.4), "identity_confidence_below_minimum"),
    (dict(ffind=[finding("MULTIPLE_FACILITY_IDENTITIES", HIGH, "x")]), "probable_merged_or_misattributed_record"),
    (dict(prov={"facility_id": "f", "source_row_id": None, "evidence_field": "x"}), "missing_provenance_fields"),
])
def test_each_gate_blocks(kw, gate):
    args = dict(category="moderate", excerpt="Dialysis unit", identity=1.0, location=1.0, ffind=[], prov=PROV)
    args.update(kw)
    failed = eligibility(args["category"], args["excerpt"], args["identity"], args["location"], args["ffind"], [], args["prov"])
    assert any(f.startswith(gate) for f in failed), failed


def test_min_evidence_threshold_is_configurable():
    strict = {**GATES, "min_evidence_category": "moderate"}
    assert eligibility("weak", "x", 1, 1, [], [], PROV, GATES) == []
    assert eligibility("weak", "x", 1, 1, [], [], PROV, strict)


# ---- ranking ----------------------------------------------------------------------------------------------------
def test_distance_score_radius_independent():
    assert distance_score(0) == 1.0 and distance_score(15) == 0.5 and distance_score(None) == 0.0
    assert distance_score(10) > distance_score(11)


def test_nearby_weak_does_not_beat_slightly_farther_moderate():
    weak = {**GOOD_CAND, "evidence_strength_score": evidence_strength_score("weak", 0.6)}
    mod = {**GOOD_CAND, "evidence_strength_score": evidence_strength_score("moderate", 0.6)}
    assert score_candidate(mod, 10)["final_ranking_score"] > score_candidate(weak, 0)["final_ranking_score"]


def test_score_bounds_and_penalty():
    s = score_candidate({**GOOD_CAND, "findings": [finding(c, LOW, "x") for c in PENALTIES] * 3}, 0)
    assert all(0 <= s[k] <= 1 for k in COMPONENTS + ("final_ranking_score", "data_quality_penalty"))
    assert data_quality_penalty([finding("POSSIBLE_DUPLICATE", LOW, "x")]) == PENALTIES["POSSIBLE_DUPLICATE"]
    assert data_quality_penalty([finding("POSSIBLE_DUPLICATE", HIGH, "x")]) == 0  # high severity gates, not penalises
    assert data_quality_penalty([finding(c, LOW, "x") for c in PENALTIES]) <= MAX_TOTAL_PENALTY


def test_weights_normalized():
    comps = dict.fromkeys(COMPONENTS, 1.0)
    assert final_ranking_score(comps, 0, ScoringWeights(2, 2, 2, 2, 2, 2)) == 1.0


def test_location_quality():
    assert location_quality_score("facility_coordinates", 1.0, False) == 1.0
    assert location_quality_score("facility_coordinates", 100.0, False) == 0.3
    assert location_quality_score("pin_centroid", None, False) == 0.4
    assert location_quality_score("none", None, False) == 0.0


def test_deterministic_tie_breaking():
    rows = [{"facility_id": "b", "final_ranking_score": 0.5, "distance_km": 3.0},
            {"facility_id": "a", "final_ranking_score": 0.5, "distance_km": 3.0},
            {"facility_id": "c", "final_ranking_score": 0.5, "distance_km": 1.0},
            {"facility_id": "d", "final_ranking_score": 0.9, "distance_km": 9.0}]
    assert [r["facility_id"] for r in sorted(rows, key=rank_key)] == ["d", "c", "a", "b"]
