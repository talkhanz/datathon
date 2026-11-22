"""Capacity planner: where would adding service S at existing hospitals close the most access gaps?

A maximal-covering location problem (Church & ReVelle 1974) solved with the project's genetic algorithm, compared
against the classic greedy heuristic and random selection. Planning aid only, not a referral tool.

DEMAND   PIN-code areas of one state (defensible centroids only). The dataset has no population counts, so each PIN is
         weighted either by its number of post offices (a rough settlement proxy) or equally. Both are proxies and the
         app says so.
SUPPLY   facilities that already credibly document S (eligible in the gold table), including those just across the
         state border, because patients cross borders.
CANDIDATES  existing hospitals that do NOT document S, with a trustworthy location and no high-severity finding.
         Optionally only those whose structured profile already resembles documented providers (model score).
         "Adding S" is hypothetical: the planner shows where it would matter, not whether it is feasible.
OBJECTIVE  (1 - equity) x people-weighted coverage + equity x district-equal coverage, where coverage = share of
         demand within `threshold_km` (straight line) of a credible provider. District-equal coverage gives each district
         the same weight, so the equity setting pushes capacity towards sparse, under-served districts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ga import Gene, run_ga
from .service_matching import TAXONOMY, name_profiles

SEED = 20261121
DIST_CAP_KM = 300.0


def _hav_matrix(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Pairwise great-circle km between point sets (n1 x n2)."""
    p1, p2 = np.radians(lat1)[:, None], np.radians(lat2)[None, :]
    dl = np.radians(lon2)[None, :] - np.radians(lon1)[:, None]
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.minimum(1.0, np.sqrt(a)))


class Problem:
    def __init__(self, pins: pd.DataFrame, providers: pd.DataFrame, cands: pd.DataFrame, threshold_km: float):
        self.pins, self.providers, self.threshold = pins.reset_index(drop=True), providers.reset_index(drop=True), threshold_km
        w = self.pins.weight.to_numpy(float)
        self.w = w / w.sum()
        codes, self.districts = pd.factorize(self.pins.district.fillna("UNKNOWN"))
        self.dcode = codes
        dw = np.bincount(codes, weights=self.w)
        self.w_district = self.w / dw[codes] / len(self.districts)    # each district sums to 1/n_districts
        plat, plon = self.pins.lat.to_numpy(float), self.pins.lon.to_numpy(float)
        if len(self.providers):
            self.d0 = _hav_matrix(plat, plon, self.providers.lat.to_numpy(float), self.providers.lon.to_numpy(float)).min(axis=1)
        else:
            self.d0 = np.full(len(self.pins), np.inf)
        D = _hav_matrix(plat, plon, cands.lat.to_numpy(float), cands.lon.to_numpy(float)) if len(cands) else np.zeros((len(plat), 0))
        gain = ((D <= threshold_km) & (self.d0[:, None] > threshold_km)).T @ self.w   # demand each candidate could newly cover
        keep = gain > 0
        self.cands, self.D = cands[keep].reset_index(drop=True), D[:, keep]
        self.cand_gain = gain[keep]

    # -------------------------------------------------------------------------------------------- metrics
    def distances(self, sel) -> np.ndarray:
        sel = [int(i) for i in sel]
        return np.minimum(self.d0, self.D[:, sel].min(axis=1)) if sel else self.d0

    def metrics(self, sel) -> dict:
        d = self.distances(sel)
        cov = d <= self.threshold
        dist_cov = np.bincount(self.dcode, weights=self.w * cov) / np.bincount(self.dcode, weights=self.w)
        return {"coverage": float(self.w @ cov), "district_coverage": float(dist_cov.mean()),
                "districts_without_access": int((dist_cov == 0).sum()), "n_districts": len(self.districts),
                "mean_km": float(self.w @ np.minimum(d, DIST_CAP_KM))}

    def objective(self, sel, equity: float) -> float:
        cov = self.distances(sel) <= self.threshold
        return float((1 - equity) * (self.w @ cov) + equity * (self.w_district @ cov))

    def district_table(self, sel) -> pd.DataFrame:
        d0, d1 = self.d0 <= self.threshold, self.distances(sel) <= self.threshold
        df = pd.DataFrame({"district": self.districts[self.dcode], "w": self.w, "before": self.w * d0, "after": self.w * d1})
        g = df.groupby("district").sum()
        return pd.DataFrame({"before": g.before / g.w, "after": g.after / g.w, "demand_share": g.w}).reset_index()


# ------------------------------------------------------------------------------------------------ solvers
def greedy(prob: Problem, k: int, equity: float):
    """Classic greedy for maximal covering: (1 - 1/e)-optimal for this monotone submodular objective."""
    sel, gains, cur = [], [], prob.objective([], equity)
    for _ in range(min(k, prob.D.shape[1])):
        best, best_v = None, cur
        for j in range(prob.D.shape[1]):
            if j in sel:
                continue
            v = prob.objective(sel + [j], equity)
            if v > best_v + 1e-12:
                best, best_v = j, v
        if best is None:
            break
        sel.append(best); gains.append(best_v - cur); cur = best_v
    return sel, gains


def random_baseline(prob: Problem, k: int, equity: float, n: int = 200, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = prob.D.shape[1]
    if m == 0:
        return np.array([prob.objective([], equity)])
    return np.array([prob.objective(list(rng.choice(m, min(k, m), replace=False)), equity) for _ in range(n)])


def ga_plan(prob: Problem, k: int, equity: float, pop_size: int = 40, n_gen: int = 60, seed: int = SEED,
            init_solutions=()):
    """GA over K integer genes (candidate indices). Duplicates are repaired to unused candidates."""
    m = prob.D.shape[1]
    k = min(k, m)
    rng = np.random.default_rng(seed)
    genes = [Gene(f"site_{i}", "int", 0, m - 1) for i in range(k)]
    cov_mask = prob.D <= prob.threshold
    need = prob.d0 > prob.threshold
    wt = (1 - equity) * prob.w + equity * prob.w_district
    base = float(wt @ ~need)

    def fitness(P):
        idx = P.astype(int)
        covered = cov_mask[:, idx].any(axis=2) & need[:, None]  # (pins, population)
        return base + wt @ covered

    def repair(P):
        for row in P:
            seen = set()
            for j in range(len(row)):
                v = int(row[j])
                while v in seen:
                    v = int(rng.integers(0, m))
                seen.add(v); row[j] = v
        return P

    init = np.array([list(s)[:k] + list(rng.choice(m, k - len(s[:k]), replace=False)) for s in init_solutions if len(s)]) \
        if init_solutions else None
    res = run_ga(fitness, genes, pop_size=pop_size, n_gen=n_gen, seed=seed, mutation_rate=1.0 / max(k, 1),
                 sigma=0.3, repair=repair, init=init, patience=20)
    return sorted(int(v) for v in res.best_x), res


def plan(prob: Problem, k: int, equity: float, seed: int = SEED) -> dict:
    """Run greedy, GA (seeded with greedy and random starts) and random; return everything the app shows."""
    g_sel, g_gain = greedy(prob, k, equity)
    ga_sel, ga_res = ga_plan(prob, k, equity, seed=seed, init_solutions=[g_sel] if g_sel else ())
    v_ga, v_g = prob.objective(ga_sel, equity), prob.objective(g_sel, equity)
    chosen = ga_sel if v_ga >= v_g - 1e-12 else g_sel
    ga0_sel, ga0_res = ga_plan(prob, k, equity, seed=seed + 1)   # independent run, no greedy seed
    rnd = random_baseline(prob, k, equity, seed=seed)
    sites = prob.cands.iloc[chosen].copy()
    sites["new_demand_share"] = [float(prob.w @ ((prob.D[:, j] <= prob.threshold) & (prob.d0 > prob.threshold))) for j in chosen]
    sites["marginal_gain"] = [prob.objective(chosen, equity) - prob.objective([c for c in chosen if c != j], equity) for j in chosen]
    return {"selected": chosen, "sites": sites.sort_values("marginal_gain", ascending=False),
            "before": prob.metrics([]), "after": prob.metrics(chosen),
            "objective": {"ga": v_ga, "greedy": v_g, "ga_random_start": prob.objective(ga0_sel, equity), "random_mean": float(rnd.mean()), "random_p95": float(np.percentile(rnd, 95)),
                          "before": prob.objective([], equity)},
            "greedy_gains": g_gain, "history": ga_res.history, "history_random_start": ga0_res.history, "districts": prob.district_table(chosen)}


def equity_frontier(prob: Problem, k: int, steps=(0.0, 0.25, 0.5, 0.75, 1.0)) -> pd.DataFrame:
    """Trade-off between people-weighted and district-equal coverage as the equity weight varies (greedy solver)."""
    rows = []
    for e in steps:
        sel, _ = greedy(prob, k, e)
        rows.append({"equity_weight": e, **prob.metrics(sel), "sites": len(sel)})
    return pd.DataFrame(rows)


def budget_curve(prob: Problem, k_max: int, equity: float) -> pd.DataFrame:
    """Coverage after adding 0..k_max sites (greedy order): how fast do returns diminish?"""
    sel, _ = greedy(prob, k_max, equity)
    rows = []
    for k in range(len(sel) + 1):
        m = prob.metrics(sel[:k])
        rows.append({"sites_added": k, **m, "objective": prob.objective(sel[:k], equity)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ assembly
def build_problem(pin_silver: pd.DataFrame, gold: pd.DataFrame, facilities: pd.DataFrame, service: str, state: str,
                  threshold_km: float, weighting: str = "post_offices", features: pd.DataFrame | None = None,
                  scores: pd.DataFrame | None = None, min_score: float | None = None,
                  min_location_conf: float = 0.6) -> Problem:
    """facilities: `modeling.lead_locations` table. features: model feature matrix (for hospital type)."""
    p = pin_silver[(pin_silver.primary_state == state) & pin_silver.coords_defensible.astype(bool)]
    pins = pd.DataFrame({"pincode": p.pincode.values, "lat": p.centroid_lat.astype(float).values,
                         "lon": p.centroid_lon.astype(float).values, "district": p.primary_district.values,
                         "weight": (p.post_office_count.astype(float).values if weighting == "post_offices"
                                    else np.ones(len(p)))})
    pad = threshold_km / 100.0 + 0.2                          # degrees: include neighbours across the border
    box = lambda df: df[df.lat.between(pins.lat.min() - pad, pins.lat.max() + pad) &  # noqa: E731
                        df.lon.between(pins.lon.min() - pad, pins.lon.max() + pad)]
    el = gold[(gold.service == service) & gold.eligible.astype(bool)]
    providers = box(pd.DataFrame({"facility_id": el.facility_id.values, "name": el["name"].values,
                                  "lat": pd.to_numeric(el.location_lat, errors="coerce").values,
                                  "lon": pd.to_numeric(el.location_lon, errors="coerce").values}).dropna(subset=["lat", "lon"]))
    f = facilities.rename(columns={"location_lat": "lat", "location_lon": "lon"}).dropna(subset=["lat", "lon"])
    f = f[(f.location_confidence_score >= min_location_conf) & ~f.has_high_finding.astype(bool)
          & ~f.facility_id.isin(set(el.facility_id))]
    exempt = TAXONOMY[service].profile_exempt   # e.g. eye / dental / IVF hospitals are implausible sites
    f = f[[not [p for p in name_profiles(n) if p not in exempt] for n in f["name"]]]
    if features is not None and "ft_hospital" in features:
        hosp = set(features.index[features.ft_hospital.astype(float) > 0.5])
        f = f[f.facility_id.isin(hosp)]
    if scores is not None:
        s = scores[scores.service == service].set_index("facility_id").model_score
        f = f.assign(model_score=f.facility_id.map(s))
        if min_score is not None:
            f = f[f.model_score >= min_score]
    return Problem(pins, providers, box(f), threshold_km)
