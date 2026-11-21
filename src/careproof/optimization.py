"""KPI-driven policy optimisation of the ranking weights (Bayesian TPE and NSGA-II genetic algorithm, via Optuna).

What is optimised: the six ranking weights in `config.ScoringWeights`. They act ONLY on candidates that already
passed the hard eligibility gates. The gates, evidence categories and quarantine rules are NOT tunable, so the
optimiser cannot buy a KPI gain with an unsafe recommendation.

KPIs (label-free, measured on the dev split of the frozen random query sample, reported on the held-out test split):
  * proximity    - mean straight-line km of the top-1 recommendation (minimise)
  * evidence     - mean evidence_strength_score of the top-K (maximise)
  * consistency  - mean identity_confidence x location_confidence of the top-K (maximise)
Coverage and the unsupported-recommendation rate do not depend on the weights (the gates fix them); they are
reported to show this.

Hard constraints (trials violating them are infeasible):
  C1  a weak claim at 0 km must not outrank an otherwise-equal moderate claim at 10 km
  C2  a moderate claim at 0 km must not outrank an otherwise-equal strong claim at 10 km
  C3  every component keeps weight >= MIN_WEIGHT (all six stay visible and meaningful)

The true objective, ranking correctness against human labels (NDCG@K), plugs into the same loop once labels
exist (`ndcg_objective`). Until then the result is a **Pareto front** of policy trade-offs, not a claim of accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DEFAULT_WEIGHTS, DISTANCE_HALF_SCORE_KM, EVIDENCE_STRENGTH_BASE, ScoringWeights
from .geography import LocationError, resolve_location
from .service_matching import resolve_service

WEIGHT_NAMES = ("service_relevance", "evidence_strength", "identity_confidence", "location_confidence",
                "readiness_evidence", "distance")
MIN_WEIGHT = 0.02
TIER_GAP_KM = 10.0


def _ds(km: float) -> float:
    return 1.0 / (1.0 + km / DISTANCE_HALF_SCORE_KM)


def constraint_violations(w: ScoringWeights) -> list[float]:
    """Optuna convention: values <= 0 are feasible."""
    n = w.normalized()
    dist_gain = n.distance * (_ds(0) - _ds(TIER_GAP_KM))
    b = EVIDENCE_STRENGTH_BASE
    c1 = dist_gain - n.evidence_strength * (b["moderate"] - b["weak"])
    c2 = dist_gain - n.evidence_strength * (b["strong"] - b["moderate"])
    c3 = MIN_WEIGHT - min(n.as_dict().values())
    return [c1, c2, c3]


def is_feasible(w: ScoringWeights) -> bool:
    return all(v <= 0 for v in constraint_violations(w))


@dataclass
class QuerySet:
    locs: list
    services: list
    max_km: list
    top_k: list

    @classmethod
    def from_frame(cls, q: pd.DataFrame, places) -> "QuerySet":
        locs, svcs, kms, ks = [], [], [], []
        for r in q.to_dict("records"):
            try:
                loc = resolve_location(r["location_type"], str(r.get("location_value") or ""), r.get("latitude"),
                                       r.get("longitude"), places)
            except LocationError:
                continue
            locs.append(loc); svcs.append(resolve_service(r["service_input"]).key)
            kms.append(float(r["max_distance_km"])); ks.append(int(r["top_k"]))
        return cls(locs, svcs, kms, ks)


def kpis(index, qs: QuerySet) -> dict:
    top1_km, ev, cons, answered, n = [], [], [], 0, len(qs.locs)
    for loc, svc, km, k in zip(qs.locs, qs.services, qs.max_km, qs.top_k):
        res, _ = index.recommend(loc, svc, km, k)
        if not res:
            continue
        answered += 1
        top1_km.append(res[0]["distance_km"])
        ev.append(np.mean([r["evidence_strength_score"] for r in res]))
        cons.append(np.mean([r["identity_confidence_score"] * r["location_confidence_score"] for r in res]))
    return {"queries": n, "answered": answered,
            "proximity_top1_km": float(np.mean(top1_km)) if top1_km else float("nan"),
            "evidence_topk": float(np.mean(ev)) if ev else float("nan"),
            "consistency_topk": float(np.mean(cons)) if cons else float("nan")}


def utility(k: dict, km_scale: float = 25.0) -> float:
    """Single scalar for Bayesian optimisation (a stated policy: 1 point of evidence ~ 25 km of proximity)."""
    return k["evidence_topk"] + k["consistency_topk"] - k["proximity_top1_km"] / km_scale


def _suggest(trial) -> ScoringWeights:
    raw = {n: trial.suggest_float(n, MIN_WEIGHT, 1.0) for n in WEIGHT_NAMES}
    return ScoringWeights(**raw).normalized()


def optimise(index, dev: QuerySet, method: str = "tpe", n_trials: int = 150, seed: int = 20261121):
    """method='tpe' (Bayesian, single objective = utility) or 'nsga2' (genetic, multi-objective Pareto)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cache = {}

    def evaluate(w: ScoringWeights):
        key = tuple(round(v, 6) for v in w.as_dict().values())
        if key not in cache:
            cache[key] = kpis(index.reweighted(w), dev)
        return cache[key]

    def constraints(trial):
        return trial.user_attrs["violations"]

    if method == "tpe":
        sampler = optuna.samplers.TPESampler(seed=seed, constraints_func=constraints, n_startup_trials=20)
        study = optuna.create_study(direction="maximize", sampler=sampler)
    else:
        sampler = optuna.samplers.NSGAIISampler(seed=seed, constraints_func=constraints, population_size=30)
        study = optuna.create_study(directions=["minimize", "maximize", "maximize"], sampler=sampler)
    study.enqueue_trial(dict(zip(WEIGHT_NAMES, DEFAULT_WEIGHTS.as_dict().values())))  # include the default policy

    def objective(trial):
        w = _suggest(trial)
        trial.set_user_attr("violations", constraint_violations(w))
        trial.set_user_attr("weights", w.as_dict())
        k = evaluate(w)
        trial.set_user_attr("kpis", k)
        if method == "tpe":
            return utility(k)
        return k["proximity_top1_km"], k["evidence_topk"], k["consistency_topk"]

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def feasible_trials(study) -> pd.DataFrame:
    rows = []
    for t in study.trials:
        if t.state.name != "COMPLETE" or "kpis" not in t.user_attrs:
            continue
        feas = all(v <= 0 for v in t.user_attrs["violations"])
        rows.append({"trial": t.number, "feasible": feas, **t.user_attrs["kpis"],
                     **{f"w_{k}": round(v, 4) for k, v in t.user_attrs["weights"].items()}})
    return pd.DataFrame(rows)


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """Non-dominated feasible trials (min proximity, max evidence, max consistency)."""
    f = df[df.feasible].copy()
    keep = []
    vals = f[["proximity_top1_km", "evidence_topk", "consistency_topk"]].to_numpy()
    for i, a in enumerate(vals):
        dominated = any((b[0] <= a[0] and b[1] >= a[1] and b[2] >= a[2]) and
                        (b[0] < a[0] or b[1] > a[1] or b[2] > a[2]) for j, b in enumerate(vals) if j != i)
        keep.append(not dominated)
    return f[keep].sort_values("proximity_top1_km")


def knee_point(front: pd.DataFrame) -> pd.Series:
    """Front member closest to the utopia point after min-max scaling each KPI."""
    cols = {"proximity_top1_km": -1, "evidence_topk": 1, "consistency_topk": 1}
    z = pd.DataFrame({c: (front[c] - front[c].min()) / ((front[c].max() - front[c].min()) or 1) * s
                      for c, s in cols.items()})
    z["proximity_top1_km"] += 1  # flip so 1 is best for all
    dist = np.sqrt(((1 - z) ** 2).sum(axis=1))
    return front.loc[dist.idxmin()]


def weights_from_row(row) -> ScoringWeights:
    return ScoringWeights(**{n: float(row[f"w_{n}"]) for n in WEIGHT_NAMES})


def top1_agreement(index_a, index_b, qs: QuerySet) -> float:
    same = tot = 0
    for loc, svc, km, k in zip(qs.locs, qs.services, qs.max_km, qs.top_k):
        a, _ = index_a.recommend(loc, svc, km, k)
        b, _ = index_b.recommend(loc, svc, km, k)
        if a or b:
            tot += 1
            same += bool(a and b and a[0]["facility_id"] == b[0]["facility_id"])
    return same / tot if tot else float("nan")


def ndcg_objective(index, labels: pd.DataFrame, qs_frame: pd.DataFrame, places, k: int = 5) -> float:
    """Future objective once human labels exist: mean NDCG@k where unlabeled shown items count as non-relevant
    (conservative). Requires labels keyed by (query_id, facility_id) with yes/no in `label_service_supported`."""
    lab = labels.assign(rel=(labels["label_service_supported"].astype(str).str.lower() == "yes").astype(int))
    rel = {(r.query_id, r.facility_id): r.rel for r in lab.itertuples()}
    scores = []
    for r in qs_frame.to_dict("records"):
        try:
            loc = resolve_location(r["location_type"], str(r.get("location_value") or ""), r.get("latitude"),
                                   r.get("longitude"), places)
        except LocationError:
            continue
        res, _ = index.recommend(loc, resolve_service(r["service_input"]).key, float(r["max_distance_km"]), k)
        g = [rel.get((r["query_id"], x["facility_id"]), 0) for x in res]
        dcg = sum(v / np.log2(i + 2) for i, v in enumerate(g))
        ideal = sorted([v for (q, _), v in rel.items() if q == r["query_id"]], reverse=True)[:k]
        idcg = sum(v / np.log2(i + 2) for i, v in enumerate(ideal))
        if idcg:
            scores.append(dcg / idcg)
    return float(np.mean(scores)) if scores else float("nan")


def per_query_kpis(index, qs: QuerySet) -> pd.DataFrame:
    rows = []
    for i, (loc, svc, km, k) in enumerate(zip(qs.locs, qs.services, qs.max_km, qs.top_k)):
        res, _ = index.recommend(loc, svc, km, k)
        if res:
            rows.append({"q": i, "top1_km": res[0]["distance_km"],
                         "evidence": float(np.mean([r["evidence_strength_score"] for r in res])),
                         "consistency": float(np.mean([r["identity_confidence_score"] * r["location_confidence_score"] for r in res]))})
    return pd.DataFrame(rows).set_index("q")


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 5000, seed: int = 20261121) -> pd.DataFrame:
    """95% CI of mean(b - a) per KPI over queries answered by both policies (b = candidate, a = baseline)."""
    common = a.index.intersection(b.index)
    d = (b.loc[common] - a.loc[common])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    out = []
    for c in d.columns:
        boots = d[c].to_numpy()[idx].mean(axis=1)
        out.append({"kpi": c, "mean_delta": d[c].mean(), "ci95_low": np.percentile(boots, 2.5),
                    "ci95_high": np.percentile(boots, 97.5), "n_queries": len(d),
                    "queries_improved": int((d[c] < 0).sum() if c == "top1_km" else (d[c] > 0).sum()),
                    "queries_worse": int((d[c] > 0).sum() if c == "top1_km" else (d[c] < 0).sum())})
    return pd.DataFrame(out)
