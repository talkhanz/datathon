import pytest
from careproof.config import PENALTIES, MAX_TOTAL_PENALTY, ScoringWeights
from careproof.scoring import (
    data_quality_penalty, distance_score, final_score, location_quality_score, rank_key, score_candidate,
    confidence_label,
)

GOOD = {"service_match_score": 1.0, "evidence_confidence_score": 0.9, "location_quality_score": 1.0,
        "readiness_score": 1.0, "flags": []}


def test_score_bounds_extremes():
    for comps in [dict.fromkeys(GOOD, 0.0), {**GOOD, "service_match_score": 5, "readiness_score": -3}]:
        s = score_candidate({**comps, "flags": ["negated_mention", "specialty_mismatch", "location_conflict",
                                                "state_conflict", "suspicious_source_url"]}, 0.0)
        for k, v in s.items():
            if k != "confidence":
                assert 0.0 <= v <= 1.0, k


def test_distance_score():
    assert distance_score(0) == 1.0 and distance_score(15) == 0.5 and distance_score(45) == 0.25
    assert distance_score(None) == 0.0 and 0 < distance_score(5000) < 0.01
    assert distance_score(10) > distance_score(11)  # strictly decreasing


def test_penalty_application_and_cap():
    assert data_quality_penalty([]) == 0.0
    assert data_quality_penalty(["specialty_mismatch"]) == PENALTIES["specialty_mismatch"]
    assert data_quality_penalty(["specialty_mismatch", "specialty_mismatch"]) == PENALTIES["specialty_mismatch"]
    assert data_quality_penalty(list(PENALTIES)) == MAX_TOTAL_PENALTY
    clean = score_candidate(GOOD, 5)
    bad = score_candidate({**GOOD, "flags": ["specialty_mismatch"]}, 5)
    assert bad["final_score"] == pytest.approx(clean["final_score"] - PENALTIES["specialty_mismatch"], abs=1e-4)
    assert bad["confidence"] != "High"


def test_weights_are_normalized():
    comps = {"service_match_score": 1, "evidence_confidence_score": 1, "location_quality_score": 1,
             "distance_score": 1, "readiness_score": 1}
    assert final_score(comps, 0, ScoringWeights(2, 2, 2, 2, 2)) == 1.0


def test_location_quality():
    assert location_quality_score("facility_coordinates", 1.0, False) == 1.0
    assert location_quality_score("facility_coordinates", 100.0, False) == 0.3
    assert location_quality_score("facility_coordinates", None, False) == 0.6
    assert location_quality_score("pin_centroid", None, False) == 0.4
    assert location_quality_score("none", None, False) == 0.0
    assert location_quality_score("facility_coordinates", 1.0, True) == 0.8


def test_confidence_labels():
    hi = {"service_match_score": 1, "evidence_confidence_score": 0.9, "location_quality_score": 1}
    assert confidence_label(hi, 0.0) == "High"
    assert confidence_label({**hi, "evidence_confidence_score": 0.4}, 0.0) == "Low"
    assert confidence_label(hi, 0.3) == "Low"
    assert confidence_label({**hi, "evidence_confidence_score": 0.6}, 0.0) == "Medium"


def test_deterministic_tie_breaking():
    rows = [{"facility_id": "b", "final_score": 0.5, "distance_km": 3.0},
            {"facility_id": "a", "final_score": 0.5, "distance_km": 3.0},
            {"facility_id": "c", "final_score": 0.5, "distance_km": 1.0},
            {"facility_id": "d", "final_score": 0.9, "distance_km": 9.0}]
    order = [r["facility_id"] for r in sorted(rows, key=rank_key)]
    assert order == ["d", "c", "a", "b"]
    assert order == [r["facility_id"] for r in sorted(reversed(rows), key=rank_key)]
