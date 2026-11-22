"""Service-profile model: GA correctness, leakage guard, split hygiene, inverse-design constraints, lead safety."""
import json

import numpy as np
import pandas as pd
import pytest

from careproof import modeling as M
from careproof.ga import Gene, run_ga

pytest.importorskip("sklearn")
from test_pipeline import make_synthetic  # noqa: E402


@pytest.fixture(scope="module")
def built():
    return make_synthetic()


# ------------------------------------------------------------------------------------------------ GA
def test_ga_solves_onemax_and_is_seeded():
    genes = [Gene(f"b{i}", "binary") for i in range(30)]
    r1 = run_ga(lambda P: P.sum(axis=1), genes, pop_size=30, n_gen=60, seed=1)
    r2 = run_ga(lambda P: P.sum(axis=1), genes, pop_size=30, n_gen=60, seed=1)
    assert r1.best_fitness >= 28 and np.array_equal(r1.best_x, r2.best_x)
    best = [h["best"] for h in r1.history]
    assert all(b2 >= b1 for b1, b2 in zip(best, best[1:]))  # elitism: best never gets worse


def test_ga_continuous_respects_bounds_and_repair():
    genes = [Gene("x", "float", -2, 2), Gene("y", "float", -2, 2)]

    def repair(P):  # feasible set: x <= 1
        P[:, 0] = np.minimum(P[:, 0], 1)
        return P
    r = run_ga(lambda P: -((P[:, 0] - 1.5) ** 2) - (P[:, 1] + 0.5) ** 2, genes, pop_size=40, n_gen=80, seed=3, repair=repair)
    assert r.best_x[0] <= 1 + 1e-9 and abs(r.best_x[0] - 1) < 0.05 and abs(r.best_x[1] + 0.5) < 0.1


# ------------------------------------------------------------------------------------------------ leakage
def test_features_never_use_label_text(built):
    silver, _, _ = built
    X1, meta = M.build_features(silver)
    scrambled = silver.copy()
    for c in ("name", "description", "procedure", "equipment", "capability"):
        scrambled[c] = "dialysis haemodialysis unit " + scrambled[c].astype(str).str[::-1]
    X2, _ = M.build_features(scrambled)
    pd.testing.assert_frame_equal(X1, X2)
    assert not set(meta.source) & set(M.FORBIDDEN_SOURCES)


def test_labels_are_explicit_text_only(built):
    silver, _, gold = built
    X, _ = M.build_features(silver)
    y = M.build_labels(gold, X.index)["dialysis"]
    lv = gold.set_index("facility_id").evidence_level
    for fid, v in y.items():
        assert v == int(lv.get(fid) == "explicit_text")
    assert y.sum() >= 1 and (y == 0).sum() >= 1  # negated and tag-only facilities are negatives


def test_state_split_keeps_states_disjoint():
    s = pd.DataFrame({"facility_id": [str(i) for i in range(400)],
                      "state_norm": [f"S{i % 15}" for i in range(400)]})
    split, groups = M.state_split(s, test_share=0.3)
    by_state = pd.DataFrame({"split": split, "g": groups}).groupby("g").split.nunique()
    assert (by_state == 1).all() and 0.2 <= (split == "test").mean() <= 0.45


# ------------------------------------------------------------------------------------------------ inverse design
def test_optimal_profile_constraints():
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame({"ft_hospital": rng.integers(0, 2, n).astype(float),
                      "num_doctors_log": rng.normal(2, 1, n), "num_doctors_missing": 0.0,
                      **{f"spec_t{i}": rng.integers(0, 2, n).astype(float) for i in range(6)}})
    X["ft_clinic"] = 1 - X.ft_hospital
    X["n_specialty_tags"] = X[[f"spec_t{i}" for i in range(6)]].sum(axis=1)
    y = ((X.ft_hospital + X.spec_t0 + X.spec_t1 + X.spec_t2 + X.spec_t3 + (X.num_doctors_log > 2)) >= 4).astype(int).to_numpy()
    meta = pd.DataFrame({"feature": X.columns, "kind": ["binary" if X[c].isin([0, 1]).all() and c != "n_specialty_tags"
                                                        else "numeric" for c in X.columns]})
    model = M.make_model("logreg").fit(X, y)
    out = M.optimal_profile(model, X, y, list(X.columns), meta, pop_size=30, n_gen=25)
    p = out["profile"]
    assert p.ft_hospital + p.ft_clinic <= 1                     # one-hot group stays one-hot
    tags = sum(p[f"spec_t{i}"] for i in range(6))
    assert tags <= out["tag_cap"] and p.n_specialty_tags >= tags  # realistic and self-consistent tag count
    assert p.num_doctors_missing == 0
    assert out["score"] > np.median(model.predict_proba(X[y == 1])[:, 1])


# ------------------------------------------------------------------------------------------------ leads safety
def test_unverified_leads_never_include_documented_negated_or_untrusted(built):
    silver, _, gold = built
    fac = M.lead_locations(silver)
    ids = fac.facility_id.tolist()
    scores = pd.DataFrame({"facility_id": ids, "service": "dialysis", "model_score": 0.9,
                           "documented_label": M.build_labels(gold, ids)["dialysis"].values})
    leads = M.unverified_leads(scores, fac, gold, "dialysis", 25.61, 85.14, 50)
    g = gold.set_index("facility_id").evidence_level
    for fid in leads.facility_id:
        assert g.get(fid, "none") in ("none", "specialty_tag_only")
    names = set(leads.name)
    assert "Negated Hospital" not in names and "Far Strong Dialysis Centre" not in names
    assert (leads.location_confidence_score >= 0.6).all() and not leads.has_high_finding.any()
    low = scores.assign(model_score=0.1)
    assert M.unverified_leads(low, fac, gold, "dialysis", 25.61, 85.14, 50).empty
