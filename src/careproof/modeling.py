"""Supervised "service-profile" model, feature selection and GA inverse design.

QUESTION THE MODEL ANSWERS
    "Given only a facility's *structured* attributes (type, operator, size, staff, specialty tags, digital footprint,
    local facility density), how likely is it that the facility's own text documents service S?"

LABEL (silver, not ground truth)
    documented = 1 if the deterministic evidence engine found an *attributable, non-negated* statement of S in the
    facility's own text (`evidence_level == "explicit_text"`), else 0. It is a label about the RECORD, not about
    whether the service is really delivered today. Human labels (PENDING_HUMAN_REVIEW) would replace it.

LEAKAGE CONTROL
    The label is computed from name / description / procedure / equipment / capability. None of those fields, and
    nothing derived from them (item counts included), is a feature. `FORBIDDEN_SOURCES` is enforced at build time and
    by a test that scrambles those fields and checks the feature matrix is unchanged.

HOW IT IS USED (never as a recommendation)
    * "profile fit" for documented, eligible facilities (an eye hospital "documenting" cardiac surgery fits poorly);
    * "unverified leads": nearby facilities with NO documented evidence whose structured profile looks like
      documented providers -> "call to verify", shown separately and never ranked. Distance and model score never
      rescue an ineligible candidate into the recommendation list.

PROTOCOL
    states are split dev/test (seeded, ~30% of facilities in test states). Filter selection, the GA wrapper and model
    choice use the dev states only (inner GroupKFold by state); the reported test metrics are on untouched states.
"""
from __future__ import annotations

import json
import re
from collections import Counter

import numpy as np
import pandas as pd

from .ga import Gene, run_ga
from .normalization import parse_array
from .service_matching import _COMPILED, TAXONOMY

SEED = 20261121
FORBIDDEN_SOURCES = ("name", "facility_name_norm", "description", "procedure", "equipment", "capability",
                     "procedure_list_json", "equipment_list_json", "capability_list_json", "procedure_n_items",
                     "equipment_n_items", "capability_n_items")
N_TOP_TAGS = 60
DENSITY_KM = 10.0
BANDS = ((0.5, "high"), (0.2, "medium"), (0.0, "low"))
LEAD_MIN_SCORE = 0.5

NUMERIC = {  # feature: (source column, clip_max, transform)
    "num_doctors_log": ("numberDoctors", 2000, "log1p"),
    "capacity_beds_log": ("capacity", 5000, "log1p"),
    "facility_age_years": ("yearEstablished", None, "age"),
    "social_media_channels": ("distinct_social_media_presence_count", 20, None),
    "facts_count_log": ("number_of_facts_about_the_organization", 1000, "log1p"),
    "followers_log": ("engagement_metrics_n_followers", 1e7, "log1p"),
    "posts_log": ("post_metrics_post_count", 1e5, "log1p"),
    "n_specialty_tags": ("specialties_n_items", 200, None),
}
LABELS = {
    "num_doctors_log": "Doctors listed", "capacity_beds_log": "Beds (capacity)", "facility_age_years": "Years operating",
    "social_media_channels": "Social-media channels", "facts_count_log": "Facts on profile",
    "followers_log": "Social followers", "posts_log": "Social posts", "n_specialty_tags": "Number of specialty tags",
    "density_10km_log": "Facilities within 10 km", "num_doctors_missing": "Doctors not listed",
    "capacity_beds_missing": "Beds not listed", "facility_age_missing": "Year not listed",
    "ft_hospital": "Type: hospital", "ft_clinic": "Type: clinic", "ft_dentist": "Type: dentist",
    "op_public": "Operator: public/government", "op_private": "Operator: private",
    "aff_government": "Affiliation: government", "aff_academic": "Affiliation: academic",
    "aff_philanthropy": "Affiliation: philanthropy", "aff_community": "Affiliation: community",
    "aff_faith": "Affiliation: faith-based", "has_official_phone": "Has official phone",
    "has_official_website": "Has official website", "has_email": "Has e-mail",
    "affiliated_staff": "Staff listed on profile", "custom_logo": "Custom logo",
}
SERVICE_WORDS = {"dialysis": "dialysis", "c_section": "obstetric", "nicu": "neonatal", "mri": "radiology/MRI",
                 "cardiac_surgery": "cardiac", "burn_care": "burn", "trauma_emergency": "emergency/trauma"}
ONE_HOT_GROUPS = (("ft_hospital", "ft_clinic", "ft_dentist"), ("op_public", "op_private"))
LOG_FEATURES = {f for f, (_, _, t) in NUMERIC.items() if t == "log1p"} | {"density_10km_log"}


def feature_label(f: str) -> str:
    if f in LABELS:
        return LABELS[f]
    if f.startswith("related_tag_"):
        return f"Any {SERVICE_WORDS.get(f[12:], f[12:])}-related tag"
    if f.startswith("spec_"):
        return "Tag: " + re.sub(r"(?<!^)(?=[A-Z])", " ", f[5:]).lower()
    return f


def to_natural(f: str, v: float) -> float:
    """Model-scale value -> human units (log features back to counts)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return float("nan")
    return float(np.expm1(v)) if f in LOG_FEATURES else float(v)


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Column or an all-missing column (synthetic / partial extracts may lack optional fields)."""
    return df[name] if name in df.columns else pd.Series([None] * len(df), index=df.index, dtype=object)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.where(~s.astype(str).str.lower().isin(["", "null", "none", "nan"])), errors="coerce")


def _bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().eq("true").astype(float)


def _density(lat: np.ndarray, lon: np.ndarray, km: float = DENSITY_KM) -> np.ndarray:
    from sklearn.neighbors import BallTree
    out = np.full(len(lat), np.nan)
    ok = ~(np.isnan(lat) | np.isnan(lon))
    if ok.sum() < 2:
        return out
    pts = np.radians(np.c_[lat[ok], lon[ok]])
    tree = BallTree(pts, metric="haversine")
    out[ok] = tree.query_radius(pts, r=km / 6371.0088, count_only=True) - 1
    return out


def top_tags(silver: pd.DataFrame, n: int = N_TOP_TAGS) -> list[str]:
    cnt = Counter(t for v in silver["specialties"] for t in set(parse_array(v)))
    return [t for t, _ in cnt.most_common(n)]


def build_features(silver: pd.DataFrame, tags: list[str] | None = None, ref_year: int = 2026):
    """Return (X indexed by facility_id, meta). Uses only structured, non-text fields (see FORBIDDEN_SOURCES)."""
    s = silver.reset_index(drop=True)
    s = s.assign(**{c: _col(s, c) for c in {v[0] for v in NUMERIC.values()} | {
        "facilityTypeId", "operatorTypeId", "affiliationTypeIds", "officialPhone", "officialWebsite", "email",
        "affiliated_staff_presence", "custom_logo_presence", "specialties", "location_lat", "location_lon"}})
    X = pd.DataFrame(index=s["facility_id"].astype(str).values)
    meta = []

    def add(name, values, kind, source):
        assert not any(source == f or source.startswith(f + "_") for f in FORBIDDEN_SOURCES), source
        X[name] = np.asarray(values, dtype=float)
        meta.append({"feature": name, "kind": kind, "source": source, "group": _group(name)})

    for f, (src, cap, tr) in NUMERIC.items():
        v = _num(s[src])
        if tr == "age":
            v = (ref_year - v).where((v >= 1800) & (v <= ref_year)).clip(0, 150)
        if cap is not None:
            v = v.clip(0, cap)
        if tr == "log1p":
            v = np.log1p(v)
        add(f, v, "numeric", src)
    for f, src in (("num_doctors_missing", "numberDoctors"), ("capacity_beds_missing", "capacity"),
                   ("facility_age_missing", "yearEstablished")):
        add(f, _num(s[src]).isna().astype(float), "binary", src)
    lat = pd.to_numeric(s["location_lat"], errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(s["location_lon"], errors="coerce").to_numpy(dtype=float)
    add("density_10km_log", np.log1p(_density(lat, lon)), "numeric", "location_lat")
    ft = s["facilityTypeId"].astype(str).str.lower()
    for k in ("hospital", "clinic", "dentist"):
        add(f"ft_{k}", (ft == k).astype(float), "binary", "facilityTypeId")
    op = s["operatorTypeId"].astype(str).str.lower()
    add("op_public", op.isin(["public", "government"]).astype(float), "binary", "operatorTypeId")
    add("op_private", (op == "private").astype(float), "binary", "operatorTypeId")
    aff = s["affiliationTypeIds"].astype(str).str.lower()
    for k in ("government", "academic", "philanthropy", "community", "faith"):
        add(f"aff_{k}", aff.str.contains(k, regex=False).astype(float), "binary", "affiliationTypeIds")
    for f, src in (("has_official_phone", "officialPhone"), ("has_official_website", "officialWebsite"),
                   ("has_email", "email")):
        add(f, (~s[src].astype(str).str.lower().isin(["", "null", "none", "nan"])).astype(float), "binary", src)
    add("affiliated_staff", _bool(s["affiliated_staff_presence"]), "binary", "affiliated_staff_presence")
    add("custom_logo", _bool(s["custom_logo_presence"]), "binary", "custom_logo_presence")
    tags = tags if tags is not None else top_tags(s)
    sets = [set(parse_array(v)) for v in s["specialties"]]
    for svc, (_, tag_rx) in _COMPILED.items():  # the service's own related-tag patterns (structured field)
        add(f"related_tag_{svc}", [any(tag_rx.search(t) for t in z) for z in sets], "binary", "specialties")
    for t in tags:
        add("spec_" + t, [t in z for z in sets], "binary", "specialties")
    return X, pd.DataFrame(meta)


def _group(name: str) -> str:
    if name.startswith(("spec_", "related_tag_")):
        return "specialty tag"
    if name.startswith(("ft_", "op_", "aff_")):
        return "type / operator"
    if name in ("followers_log", "posts_log", "social_media_channels", "facts_count_log", "has_official_website",
                "has_email", "custom_logo", "affiliated_staff", "has_official_phone"):
        return "digital footprint"
    if name == "density_10km_log":
        return "location context"
    return "size / staff"


def build_labels(gold: pd.DataFrame, ids) -> pd.DataFrame:
    """documented[facility, service] = 1 iff explicit, attributable text evidence exists (silver label)."""
    y = pd.DataFrame(0, index=pd.Index(ids, dtype=str), columns=list(TAXONOMY), dtype=int)
    pos = gold[gold.evidence_level == "explicit_text"]
    for svc, grp in pos.groupby("service"):
        y.loc[y.index.intersection(grp.facility_id.astype(str)), svc] = 1
    return y


def tag_rule_scores(gold: pd.DataFrame, ids) -> pd.DataFrame:
    """Baseline B0: 1 if the facility carries a specialty tag related to the service, else 0."""
    out = pd.DataFrame(0.0, index=pd.Index(ids, dtype=str), columns=list(TAXONOMY))
    g = gold[gold.specialty_tags_json.fillna("[]").map(lambda v: len(json.loads(v)) > 0)]
    for svc, grp in g.groupby("service"):
        out.loc[out.index.intersection(grp.facility_id.astype(str)), svc] = 1.0
    return out


def state_split(silver: pd.DataFrame, test_share: float = 0.3, seed: int = SEED) -> pd.Series:
    """Assign whole states to dev or test (seeded) so no state appears in both. Unknown state -> dev."""
    st = silver.set_index(silver.facility_id.astype(str))["state_norm"].fillna("UNKNOWN").astype(str)
    sizes = st[st != "UNKNOWN"].value_counts()
    rng = np.random.default_rng(seed)
    order = list(sizes.index[rng.permutation(len(sizes))])
    test, n = set(), 0
    for s in order:
        if n / len(st) >= test_share:
            break
        if sizes[s] > 0.2 * len(st):    # never put a single giant state in test alone
            continue
        test.add(s); n += sizes[s]
    return st.map(lambda s: "test" if s in test else "dev").rename("split"), st.rename("group")


# ----------------------------------------------------------------------------------------------- metrics
def ece(y, p, bins: int = 10) -> float:
    y, p = np.asarray(y), np.asarray(p)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def metrics(y, p) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    ok = len(np.unique(y)) == 2
    return {"average_precision": float(average_precision_score(y, p)) if ok else float("nan"),
            "roc_auc": float(roc_auc_score(y, p)) if ok else float("nan"),
            "brier": float(brier_score_loss(y, np.clip(p, 0, 1))), "ece": ece(y, np.clip(p, 0, 1)),
            "base_rate": float(np.mean(y)), "n": int(len(y)), "n_pos": int(np.sum(y))}


# ----------------------------------------------------------------------------------------------- models
def make_model(kind: str, seed: int = SEED):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.5, max_iter=500))
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                              l2_regularization=1.0, random_state=seed)
    raise ValueError(kind)


def oof_predict(kind, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, n_splits: int = 5) -> np.ndarray:
    from sklearn.model_selection import GroupKFold
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=min(n_splits, len(np.unique(groups)))).split(X, y, groups):
        if y[tr].sum() == 0:
            continue
        m = make_model(kind).fit(X.iloc[tr], y[tr])
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
    return p


# ----------------------------------------------------------------------------------------------- feature selection
def filter_features(X: pd.DataFrame, y: np.ndarray, meta: pd.DataFrame, min_prevalence: float = 0.005,
                    corr_max: float = 0.9, seed: int = SEED) -> pd.DataFrame:
    """Stage 1 (filter): drop near-constant features, rank by mutual information, prune |Spearman| > corr_max."""
    from sklearn.feature_selection import mutual_info_classif
    kinds = meta.set_index("feature")["kind"]
    rep = []
    Xi = X.fillna(X.median())
    for f in X.columns:
        prev = float((X[f] > 0).mean()) if kinds[f] == "binary" else float(X[f].notna().mean())
        reason = None
        if kinds[f] == "binary" and not (min_prevalence <= prev <= 1 - min_prevalence):
            reason = "near-constant"
        elif kinds[f] == "numeric" and X[f].nunique() < 3:
            reason = "near-constant"
        rep.append({"feature": f, "prevalence_or_coverage": prev, "drop_reason": reason})
    rep = pd.DataFrame(rep).set_index("feature")
    keep = rep.index[rep.drop_reason.isna()]
    mi = mutual_info_classif(Xi[keep], y, discrete_features=[kinds[f] == "binary" for f in keep], random_state=seed)
    rep["mutual_info"] = pd.Series(mi, index=keep)
    order = rep.loc[keep].sort_values("mutual_info", ascending=False).index
    corr = Xi[order].rank().corr().abs()
    kept = []
    for f in order:
        hit = next((k for k in kept if corr.loc[f, k] > corr_max), None)
        if hit:
            rep.loc[f, "drop_reason"] = f"correlated with {hit}"
        else:
            kept.append(f)
    rep["filter_kept"] = rep.drop_reason.isna()
    return rep.reset_index()


def ga_select(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, candidates: list[str], mi: pd.Series,
              pop_size: int = 24, n_gen: int = 18, parsimony: float = 0.001, seed: int = SEED):
    """Stage 2 (wrapper): GA over a binary feature mask. Fitness = inner state-grouped 3-fold CV average precision
    of a logistic regression on the masked features, minus `parsimony` per selected feature."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import GroupKFold
    Z = X[candidates].fillna(X[candidates].median())
    Z = ((Z - Z.mean()) / Z.std(ddof=0).replace(0, 1)).to_numpy()
    folds = list(GroupKFold(n_splits=3).split(Z, y, groups))
    cache = {}

    def score(mask):
        key = mask.astype(bool).tobytes()
        if key not in cache:
            cols = np.flatnonzero(mask)
            if len(cols) == 0:
                cache[key] = 0.0
            else:
                aps = []
                for tr, te in folds:
                    if y[tr].sum() == 0 or y[te].sum() == 0:
                        continue
                    m = LogisticRegression(C=0.5, max_iter=300).fit(Z[np.ix_(tr, cols)], y[tr])
                    aps.append(average_precision_score(y[te], m.decision_function(Z[np.ix_(te, cols)])))
                cache[key] = float(np.mean(aps)) - parsimony * len(cols) if aps else 0.0
        return cache[key]

    genes = [Gene(f, "binary") for f in candidates]
    ranked = mi.reindex(candidates).fillna(0).rank(ascending=False).to_numpy()
    init = np.array([(ranked <= k).astype(float) for k in (5, 10, 15, 25, len(candidates))])
    res = run_ga(lambda P: np.array([score(m) for m in P]), genes, pop_size=pop_size, n_gen=n_gen, seed=seed,
                 init=init, patience=6)
    selected = [f for f, b in zip(candidates, res.best_x) if b > 0.5]
    return selected, res


# ----------------------------------------------------------------------------------------------- inverse design
def optimal_profile(model, X: pd.DataFrame, y: np.ndarray, features: list[str], meta: pd.DataFrame,
                    realism: float = 0.25, pop_size: int = 60, n_gen: int = 60, seed: int = SEED):
    """GA inverse design: feature values that maximise the model score, penalised for straying from the cloud of real
    documented facilities (nearest-neighbour distance beyond the typical one). Missing-value flags are fixed to 0
    ("value is listed"); one-hot groups stay one-hot; the specialty-tag count stays realistic and consistent."""
    from sklearn.neighbors import NearestNeighbors
    kinds = meta.set_index("feature")["kind"]
    tunable = [f for f in features if not f.endswith("_missing")]
    fixed = {f: 0.0 for f in features if f.endswith("_missing")}
    genes = []
    for f in tunable:
        if kinds[f] == "binary":
            genes.append(Gene(f, "binary"))
        else:
            v = X[f].dropna()
            genes.append(Gene(f, "float", float(v.quantile(0.01)), float(v.quantile(0.99))))
    pos = X[y == 1]
    med = X[tunable].median()
    sd = X[tunable].std(ddof=0).replace(0, 1)
    P_std = ((pos[tunable].fillna(med) - X[tunable].mean()) / sd).to_numpy()
    nn = NearestNeighbors(n_neighbors=2).fit(P_std)
    d_typ = float(np.median(nn.kneighbors(P_std)[0][:, 1])) or 1.0
    spec_idx = [j for j, f in enumerate(tunable) if f.startswith("spec_")]
    tag_cap = max(1, int(np.ceil(np.percentile(pos[[tunable[j] for j in spec_idx]].sum(axis=1), 90)))) if spec_idx else 0
    n_tag_col = tunable.index("n_specialty_tags") if "n_specialty_tags" in tunable else None
    groups = [[tunable.index(f) for f in g if f in tunable] for g in ONE_HOT_GROUPS]
    rng = np.random.default_rng(seed)

    def repair(P):
        for g in groups:
            if len(g) > 1:
                for row in P:
                    on = [j for j in g if row[j] > 0.5]
                    for j in on[1:] if len(on) > 1 else []:
                        row[j] = 0.0
        if spec_idx:
            for row in P:
                on = [j for j in spec_idx if row[j] > 0.5]
                if len(on) > tag_cap:
                    for j in rng.choice(on, len(on) - tag_cap, replace=False):
                        row[j] = 0.0
                if n_tag_col is not None:
                    row[n_tag_col] = max(row[n_tag_col], sum(row[j] > 0.5 for j in spec_idx))
        return P

    def frame(P):
        df = pd.DataFrame(P, columns=tunable)
        for f, v in fixed.items():
            df[f] = v
        return df[features]

    def fitness(P):
        p = model.predict_proba(frame(P))[:, 1]
        d = nn.kneighbors(((pd.DataFrame(P, columns=tunable) - X[tunable].mean()) / sd).to_numpy(), n_neighbors=1)[0][:, 0]
        return p - realism * np.maximum(0.0, d / d_typ - 1.0)

    # seed the population with real documented facilities so the search starts on the data manifold
    init = pos[tunable].fillna(med).sample(min(len(pos), pop_size // 2), random_state=seed).to_numpy()
    res = run_ga(fitness, genes, pop_size=pop_size, n_gen=n_gen, seed=seed, init=init, repair=repair, patience=15)
    best = frame(res.best_x[None, :]).iloc[0]
    d_best, i_best = nn.kneighbors(((best[tunable] - X[tunable].mean()) / sd).to_numpy()[None, :], n_neighbors=1)
    return {"profile": best, "score": float(model.predict_proba(frame(res.best_x[None, :]))[:, 1][0]),
            "fitness": res.best_fitness, "nearest_index": pos.index[int(i_best[0][0])],
            "nearest_distance_rel": float(d_best[0][0] / d_typ), "history": res.history, "tag_cap": tag_cap}


# ----------------------------------------------------------------------------------------------- orchestration
def band(p: float) -> str:
    return next(name for t, name in BANDS if p >= t)


def train_all(silver: pd.DataFrame, gold: pd.DataFrame, services=None, seed: int = SEED, log=print) -> dict:
    """Run the full protocol for every service and return the output tables (pandas DataFrames)."""
    from sklearn.inspection import permutation_importance
    services = list(services or TAXONOMY)
    X, meta = build_features(silver)
    Y = build_labels(gold, X.index)
    B0 = tag_rule_scores(gold, X.index)
    split, groups = state_split(silver)
    split, groups = split.reindex(X.index), groups.reindex(X.index)
    dev, test = (split == "dev").to_numpy(), (split == "test").to_numpy()
    g = groups.to_numpy()
    out = {k: [] for k in ("selection", "metrics", "history", "profile", "summary", "scores", "calibration")}

    for svc in services:
        y = Y[svc].to_numpy()
        log(f"[{svc}] positives={y.sum()} ({y.mean():.1%}) dev={dev.sum()} test={test.sum()}")
        filt = filter_features(X[dev], y[dev], meta, seed=seed)
        cand = filt.loc[filt.filter_kept, "feature"].tolist()
        sel, ga_res = ga_select(X[dev], y[dev], g[dev], cand, filt.set_index("feature")["mutual_info"], seed=seed)
        log(f"[{svc}] filter kept {len(cand)}/{X.shape[1]}, GA selected {len(sel)}")
        for h in ga_res.history:
            out["history"].append({"service": svc, "task": "feature_selection", **h})

        sets = {"all": list(X.columns), "filter": cand, "ga": sel}
        rows, best = [], None
        for kind in ("logreg", "hgb"):
            for fs, cols in sets.items():
                if kind == "logreg" and fs == "all":
                    continue
                p_dev = oof_predict(kind, X.loc[dev, cols], y[dev], g[dev])
                m_dev = metrics(y[dev], p_dev)
                model = make_model(kind).fit(X.loc[dev, cols], y[dev])
                m_test = metrics(y[test], model.predict_proba(X.loc[test, cols])[:, 1])
                rows.append({"model": kind, "feature_set": fs, "n_features": len(cols), "dev": m_dev, "test": m_test,
                             "fitted": model})
        rows.append({"model": "specialty_tag_rule", "feature_set": "rule", "n_features": 1,
                     "dev": metrics(y[dev], B0.loc[dev, svc]), "test": metrics(y[test], B0.loc[test, svc]), "fitted": None})
        best = max(rows, key=lambda r: r["dev"]["average_precision"])  # the tag rule competes too (dev states only)
        for r in rows:
            for part in ("dev", "test"):
                out["metrics"].append({"service": svc, "model": r["model"], "feature_set": r["feature_set"],
                                       "n_features": r["n_features"], "split": "dev_cv" if part == "dev" else "test",
                                       "chosen": r is best, **r[part]})
        rule_row = rows[-1]
        log(f"[{svc}] chosen {best['model']}/{best['feature_set']} test AP={best['test']['average_precision']:.3f} "
            f"(tag rule {rule_row['test']['average_precision']:.3f}, base {best['test']['base_rate']:.3f})")
        if best["fitted"] is None:  # a simple rule beat every model on dev: do not use a model for this service
            filt = filt.assign(service=svc, ga_selected=filt.feature.isin(sel), in_final_model=False,
                               permutation_importance=np.nan, importance_std=np.nan,
                               feature_label=filt.feature.map(feature_label),
                               group=filt.feature.map(meta.set_index("feature")["group"]))
            out["selection"].append(filt)
            out["summary"].append({"service": svc, "model": best["model"], "feature_set": "rule", "model_used": False,
                                   "n_features": 0, "test_average_precision": best["test"]["average_precision"],
                                   "test_base_rate": best["test"]["base_rate"]})
            log(f"[{svc}] model NOT used: the specialty-tag rule is better on dev states")
            continue
        cols = sets[best["feature_set"]]
        imp = permutation_importance(best["fitted"], X.loc[test, cols], y[test], scoring="average_precision",
                                     n_repeats=5, random_state=seed)
        imp_s = pd.Series(imp.importances_mean, index=cols)
        imp_sd = pd.Series(imp.importances_std, index=cols)
        filt = filt.assign(service=svc, ga_selected=filt.feature.isin(sel), in_final_model=filt.feature.isin(cols),
                           permutation_importance=filt.feature.map(imp_s), importance_std=filt.feature.map(imp_sd),
                           feature_label=filt.feature.map(feature_label),
                           group=filt.feature.map(meta.set_index("feature")["group"]))
        out["selection"].append(filt)

        # calibration on test
        p_test = best["fitted"].predict_proba(X.loc[test, cols])[:, 1]
        bins = np.minimum((p_test * 10).astype(int), 9)
        for b in range(10):
            mask = bins == b
            if mask.any():
                out["calibration"].append({"service": svc, "bin": b, "mean_predicted": float(p_test[mask].mean()),
                                           "observed_rate": float(y[test][mask].mean()), "n": int(mask.sum())})

        # scores for every facility: out-of-fold over all states (no facility is scored by a model that saw its state)
        p_all = oof_predict(best["model"], X[cols], y, g)
        out["scores"].append(pd.DataFrame({"facility_id": X.index, "service": svc, "documented_label": y,
                                           "model_score": np.round(p_all, 4), "split": split.values}))

        # inverse design on a model refit on all data
        final = make_model(best["model"]).fit(X[cols], y)
        prof = optimal_profile(final, X[cols], y, cols, meta, seed=seed)
        for h in prof["history"]:
            out["history"].append({"service": svc, "task": "optimal_profile", **h})
        pos_med, neg_med = X.loc[y == 1, cols].median(), X.loc[y == 0, cols].median()
        for f in cols:
            out["profile"].append({"service": svc, "feature": f, "feature_label": feature_label(f),
                                   "kind": meta.set_index("feature").loc[f, "kind"],
                                   "optimal_value": to_natural(f, prof["profile"][f]),
                                   "documented_median": to_natural(f, pos_med[f]),
                                   "undocumented_median": to_natural(f, neg_med[f]),
                                   "documented_rate": float(X.loc[y == 1, f].mean()) if f in meta.feature[meta.kind == "binary"].values else float("nan"),
                                   "undocumented_rate": float(X.loc[y == 0, f].mean()) if f in meta.feature[meta.kind == "binary"].values else float("nan"),
                                   "permutation_importance": float(imp_s.get(f, np.nan))})
        nearest = silver.set_index(silver.facility_id.astype(str)).loc[prof["nearest_index"]]
        out["summary"].append({"service": svc, "model": best["model"], "feature_set": best["feature_set"], "model_used": True,
                               "n_features": len(cols), "profile_score": prof["score"],
                               "documented_median_score": float(np.median(p_all[y == 1])) if y.sum() else float("nan"),
                               "undocumented_median_score": float(np.median(p_all[y == 0])),
                               "nearest_real_facility_id": prof["nearest_index"],
                               "nearest_real_facility_name": str(nearest["name"]),
                               "nearest_distance_relative": prof["nearest_distance_rel"], "tag_cap": prof["tag_cap"],
                               "test_average_precision": best["test"]["average_precision"],
                               "test_base_rate": best["test"]["base_rate"]})
        log(f"[{svc}] GA optimal profile score={prof['score']:.3f} nearest real={nearest['name']!s:.40}")

    scores = pd.concat(out["scores"], ignore_index=True)
    scores["score_band"] = scores.model_score.map(band)
    feats = X.reset_index().rename(columns={"index": "facility_id"})
    return {"model_scores": scores, "model_feature_selection": pd.concat(out["selection"], ignore_index=True),
            "model_metrics": pd.DataFrame(out["metrics"]), "model_ga_history": pd.DataFrame(out["history"]),
            "model_optimal_profile": pd.DataFrame(out["profile"]), "model_profile_summary": pd.DataFrame(out["summary"]),
            "model_calibration": pd.DataFrame(out["calibration"]), "model_features": feats,
            "model_facilities": lead_locations(silver)}


def unverified_leads(scores: pd.DataFrame, silver_loc: pd.DataFrame, gold: pd.DataFrame, service: str, lat: float,
                     lon: float, max_km: float, k: int = 5, min_location_conf: float = 0.6) -> pd.DataFrame:
    """Nearby facilities with NO attributable text evidence for `service` but a high model score.
    Excludes anything whose text negates the service or attributes it to another facility, any facility with a
    high-severity finding, and any facility whose location is not trustworthy. These are NOT recommendations."""
    from .pipeline import _haversine_vec
    s = scores[(scores.service == service) & (scores.model_score >= LEAD_MIN_SCORE) & (scores.documented_label == 0)]
    if s.empty:
        return s.assign(distance_km=[])
    g = gold[gold.service == service].set_index("facility_id")
    bad = set(g.index[g.evidence_level.isin(["negated_only", "other_facility_only", "explicit_text"])])
    s = s[~s.facility_id.isin(bad)].merge(silver_loc, on="facility_id")
    s = s[(s.location_confidence_score >= min_location_conf) & ~s.has_high_finding]
    if s.empty:
        return s.assign(distance_km=[])
    s = s.assign(distance_km=_haversine_vec(lat, lon, s.location_lat.to_numpy(), s.location_lon.to_numpy()))
    return s[s.distance_km <= max_km].sort_values(["model_score", "distance_km"], ascending=[False, True]).head(k)


def lead_locations(silver: pd.DataFrame) -> pd.DataFrame:
    """Slim facility table for `unverified_leads`."""
    silver = silver.reset_index(drop=True)
    f = silver.facility_findings_json.fillna("[]").map(json.loads)
    return pd.DataFrame({"facility_id": silver.facility_id.astype(str), "name": silver["name"],
                         "location_lat": pd.to_numeric(silver.location_lat, errors="coerce"),
                         "location_lon": pd.to_numeric(silver.location_lon, errors="coerce"),
                         "location_confidence_score": pd.to_numeric(silver.location_confidence_score, errors="coerce"),
                         "has_high_finding": f.map(lambda z: any(x.get("severity") == "high" for x in z)),
                         "phone": _col(silver, "officialPhone"), "address_city": _col(silver, "address_city")})
