import pandas as pd
import pytest

from careproof import optimization as O
from careproof.config import WEIGHT_SCENARIOS, ScoringWeights


def test_default_and_hand_set_are_feasible():
    assert O.is_feasible(WEIGHT_SCENARIOS["default"]) and O.is_feasible(WEIGHT_SCENARIOS["hand_set_v1"])


def test_distance_heavy_policy_violates_safety_constraint():
    assert not O.is_feasible(ScoringWeights(0.02, 0.05, 0.02, 0.02, 0.02, 0.87))


def test_pareto_front_and_knee():
    df = pd.DataFrame([
        {"feasible": True, "proximity_top1_km": 10, "evidence_topk": 0.5, "consistency_topk": 0.9, "w_x": 1},
        {"feasible": True, "proximity_top1_km": 12, "evidence_topk": 0.4, "consistency_topk": 0.8, "w_x": 2},  # dominated
        {"feasible": True, "proximity_top1_km": 15, "evidence_topk": 0.7, "consistency_topk": 0.9, "w_x": 3},
        {"feasible": False, "proximity_top1_km": 1, "evidence_topk": 0.9, "consistency_topk": 1.0, "w_x": 4},  # infeasible
    ])
    front = O.pareto_front(df)
    assert set(front.w_x) == {1, 3}
    assert O.knee_point(front).w_x in (1, 3)


def test_paired_bootstrap_detects_consistent_improvement():
    a = pd.DataFrame({"top1_km": [10.0, 20, 30, 40], "evidence": [0.5] * 4, "consistency": [0.9] * 4})
    b = a.assign(top1_km=a.top1_km - 2)
    r = O.paired_bootstrap(a, b, n_boot=500).set_index("kpi")
    assert r.loc["top1_km", "ci95_high"] < 0 and r.loc["top1_km", "queries_improved"] == 4


def test_optimise_runs_deterministically_on_synthetic():
    pytest.importorskip("optuna")
    from test_pipeline import make_synthetic
    from careproof import pipeline as P
    from careproof.geography import ResolvedLocation
    gold = make_synthetic()[2]
    qs = O.QuerySet([ResolvedLocation(25.61, 85.14, "t", "user_coordinates")], ["dialysis"], [100.0], [3])
    idx = P.RecommendationIndex(gold)
    s1, s2 = O.optimise(idx, qs, "tpe", 12), O.optimise(idx, qs, "tpe", 12)
    assert [t.params for t in s1.trials] == [t.params for t in s2.trials]
    assert len(O.optimise(idx, qs, "nsga2", 12).trials) == 12
