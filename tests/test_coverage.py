"""Capacity planner: metrics, solver quality and candidate hygiene on a synthetic grid."""
import numpy as np
import pandas as pd

from careproof import coverage as CV


def toy(threshold=20.0):
    rng = np.random.default_rng(0)
    pins = pd.DataFrame({"pincode": [f"{800000 + i}" for i in range(120)], "lat": 25 + rng.uniform(0, 2, 120),
                         "lon": 85 + rng.uniform(0, 2, 120), "district": [f"D{i % 6}" for i in range(120)],
                         "weight": rng.integers(1, 10, 120)})
    providers = pd.DataFrame({"facility_id": ["p1"], "name": ["P"], "lat": [25.2], "lon": [85.2]})
    cands = pd.DataFrame({"facility_id": [f"c{i}" for i in range(25)], "name": [f"C{i}" for i in range(25)],
                          "lat": 25 + rng.uniform(0, 2, 25), "lon": 85 + rng.uniform(0, 2, 25),
                          "address_city": "x", "model_score": 0.3})
    return CV.Problem(pins, providers, cands, threshold)


def test_metrics_monotone_and_bounded():
    pr = toy()
    b = pr.metrics([])
    a = pr.metrics(list(range(pr.D.shape[1])))
    assert 0 <= b["coverage"] <= a["coverage"] <= 1 and a["mean_km"] <= b["mean_km"]
    assert a["districts_without_access"] <= b["districts_without_access"]
    assert abs(pr.w.sum() - 1) < 1e-9 and abs(pr.w_district.sum() - 1) < 1e-9


def test_candidates_without_new_coverage_are_dropped():
    pr = toy()
    assert (pr.cand_gain > 0).all() and pr.D.shape[1] == len(pr.cands)


def test_hybrid_ga_never_worse_than_greedy_and_beats_random():
    pr = toy()
    for eq in (0.0, 0.5, 1.0):
        out = CV.plan(pr, 4, eq)
        o = out["objective"]
        assert o["ga"] >= o["greedy"] - 1e-12 >= o["random_mean"]
        assert len(set(out["selected"])) == len(out["selected"]) <= 4     # distinct sites, within budget
        assert abs(pr.objective(out["selected"], eq) - max(o["ga"], o["greedy"])) < 1e-12


def test_ga_matches_bruteforce_on_small_instance():
    from itertools import combinations
    pr = toy(threshold=15.0)
    m = pr.D.shape[1]
    best = max(pr.objective(list(c), 0.5) for c in combinations(range(m), 2))
    sel, _ = CV.ga_plan(pr, 2, 0.5, n_gen=80)
    assert abs(pr.objective(sel, 0.5) - best) < 1e-9


def test_budget_curve_is_nondecreasing():
    c = CV.budget_curve(toy(), 6, 0.5)
    assert (np.diff(c.objective) >= -1e-12).all() and c.sites_added.iloc[0] == 0
