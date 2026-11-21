"""Benchmark runner and automatic metrics. Human-judged metrics are exported as a labeling template."""
from __future__ import annotations

import json
from typing import Mapping

import numpy as np
import pandas as pd

from .config import WEIGHT_SCENARIOS, INDIA_LAT_RANGE, INDIA_LON_RANGE
from .geography import LocationError, haversine_km, resolve_location
from .normalization import clean_str, parse_array
from .pipeline import keyword_baseline, recommend
from .service_matching import resolve_service

CONTRA = {"specialty_mismatch", "negated_mention", "location_conflict", "state_conflict"}


def excerpt_is_exact(silver_row: Mapping, field: str, excerpt: str) -> bool:
    if not excerpt or not field:
        return False
    if field in ("description", "name"):
        return excerpt in (clean_str(silver_row.get(field)) or "")
    return any(excerpt in item for item in parse_array(silver_row.get(field)))


def run_benchmark(queries: pd.DataFrame, gold: pd.DataFrame, silver: pd.DataFrame, index):
    by_id = silver.set_index("facility_id").to_dict("index")
    qrows, rrows, label_rows = [], [], []
    for q in queries.to_dict("records"):
        svc = resolve_service(q["service_input"]).key
        base = {"query_id": q["query_id"], "category": q["category"], "service": svc,
                "expected_outcome": q["expected_outcome"]}
        try:
            loc = resolve_location(q["location_type"], str(q.get("location_value") or ""),
                                   q.get("latitude"), q.get("longitude"), index)
        except LocationError as e:
            qrows.append({**base, "status": "location_error", "message": str(e), "n_results": 0,
                          "outcome_as_expected": None if q["expected_outcome"] == "unknown"
                          else q["expected_outcome"] == "location_error",
                          "location_warning": None})
            continue
        maxkm, k = float(q["max_distance_km"]), int(q["top_k"])
        full = {name: recommend(gold, loc, svc, maxkm, k, w) for name, w in WEIGHT_SCENARIOS.items()}
        runs = {name: v[0] for name, v in full.items()}
        res, summ = full["default"]
        bl = keyword_baseline(gold, loc, svc, maxkm, k)
        exp_ok = None if q["expected_outcome"] == "unknown" else (q["expected_outcome"] == "abstain" and not res) or (
            (q["expected_outcome"] == "results" and len(res) > 0) or
            (q["expected_outcome"] == "results_with_warning" and len(res) > 0 and bool(loc.warnings)))
        default_ids = [r["facility_id"] for r in res]
        stab = {}
        for name, rr in runs.items():
            ids = [r["facility_id"] for r in rr]
            inter, union = len(set(ids) & set(default_ids)), len(set(ids) | set(default_ids))
            stab[f"jaccard_{name}"] = round(inter / union, 3) if union else 1.0
            stab[f"top1_same_{name}"] = (ids[:1] == default_ids[:1])
        cites = [excerpt_is_exact(by_id[r["facility_id"]], r["evidence_field"], r["evidence_excerpt"]) for r in res]
        geo = [(r["distance_km"] <= maxkm + 1e-6) and _in_india(by_id[r["facility_id"]]) and
               abs(haversine_km(loc.lat, loc.lon, by_id[r["facility_id"]]["location_lat"],
                                by_id[r["facility_id"]]["location_lon"]) - r["distance_km"]) < 0.05 for r in res]
        bl_weak = [b for b in bl if b["evidence_level"] != "explicit_text" or CONTRA & set(b["flags"])]
        qrows.append({**base, "status": "ok", "message": " | ".join(loc.warnings) or None, "n_results": len(res),
                      "outcome_as_expected": exp_ok, "location_warning": bool(loc.warnings),
                      "citations_exact": sum(cites), "citations_total": len(cites),
                      "geo_valid": sum(geo), "geo_total": len(geo),
                      "n_strong": sum(r["evidence_category"] == "strong" for r in res),
                      "n_moderate": sum(r["evidence_category"] == "moderate" for r in res),
                      "n_weak": sum(r["evidence_category"] == "weak" for r in res),
                      "retrieved_mentions": summ["retrieved_mentions_in_radius"],
                      "held_back": summ["held_back_in_radius"],
                      "cp_with_contradiction_flags": sum(bool(CONTRA & set(r["flags"])) for r in res),
                      "baseline_n": len(bl), "baseline_weak_or_contradictory": len(bl_weak),
                      "baseline_top1_weak": bool(bl and bl[0] in bl_weak),
                      "top1_differs_from_baseline": bool(bl and res and bl[0]["facility_id"] != res[0]["facility_id"]),
                      **stab})
        for system, items in (("careproof", res), ("baseline_nearest_keyword", bl)):
            for r in items:
                row = by_id[r["facility_id"]]
                g = gold[(gold.facility_id == r["facility_id"]) & (gold.service == svc)].iloc[0]
                label_rows.append({
                    "query_id": q["query_id"], "service": svc, "location": loc.label, "system": system,
                    "rank": r["rank"], "facility_id": r["facility_id"], "facility_name": r["facility_name"],
                    "distance_km": r["distance_km"], "evidence_category": g["evidence_category"],
                    "eligible_under_policy": bool(g["eligible"]),
                    "evidence_field": g["evidence_field"], "evidence_excerpt": g["evidence_excerpt"],
                    "flags": json.dumps(r["flags"]), "source_url": row.get("primary_source_url"),
                    "label_service_supported": "", "reviewer": "", "notes": "",
                    "review_status": "PENDING_HUMAN_REVIEW",
                })
        for r in res:
            rrows.append({"query_id": q["query_id"], **{k: r[k] for k in (
                "rank", "facility_id", "source_row_id", "facility_name", "distance_km", "final_ranking_score",
                "evidence_category", "service_relevance_score", "evidence_strength_score", "identity_confidence_score",
                "location_confidence_score", "readiness_evidence_score", "distance_score", "data_quality_penalty",
                "evidence_field", "evidence_excerpt", "pipeline_version")}})
    return pd.DataFrame(qrows), pd.DataFrame(rrows), pd.DataFrame(label_rows)


def _in_india(row) -> bool:
    lat, lon = row.get("location_lat"), row.get("location_lon")
    return lat is not None and INDIA_LAT_RANGE[0] <= lat <= INDIA_LAT_RANGE[1] and INDIA_LON_RANGE[0] <= lon <= INDIA_LON_RANGE[1]


def summarize(qdf: pd.DataFrame) -> dict:
    ok = qdf[qdf.status == "ok"]
    ans = qdf[qdf.expected_outcome.isin(["results", "results_with_warning"])]
    out = {
        "queries_total": int(len(qdf)),
        "outcome_as_expected": (f"{int(qdf.outcome_as_expected.dropna().astype(bool).sum())}/"
                                f"{int(qdf.outcome_as_expected.notna().sum())}" if qdf.outcome_as_expected.notna().any()
                                else "n/a (no expected outcomes: unbiased random sample)"),
        "coverage_queries_with_results": f"{int((ok.n_results > 0).sum())}/{len(qdf)}",
        "coverage_answerable_queries_with_results": f"{int((ans.n_results > 0).sum())}/{len(ans)}",
        "abstained_no_eligible_in_radius": int(((qdf.status == "ok") & (qdf.n_results == 0)).sum()),
        "abstentions_with_mentions_held_back": int(((qdf.status == "ok") & (qdf.n_results == 0) & (qdf.get("retrieved_mentions", 0) > 0)).sum()),
        "citation_exact_substring_rate": _rate(ok.citations_exact.sum(), ok.citations_total.sum()),
        "geographic_validity_rate": _rate(ok.geo_valid.sum(), ok.geo_total.sum()),
        "careproof_results_with_contradiction_flags": f"{int(ok.cp_with_contradiction_flags.sum())}/{int(ok.n_results.sum())}",
        "careproof_results_by_category": {c: int(ok[f"n_{c}"].sum()) for c in ("strong", "moderate", "weak") if f"n_{c}" in ok},
        "baseline_results_weak_or_contradictory": f"{int(ok.baseline_weak_or_contradictory.sum())}/{int(ok.baseline_n.sum())}",
        "baseline_top1_weak_queries": f"{int(ok.baseline_top1_weak.sum())}/{len(ok)}",
        "top1_differs_from_baseline": f"{int(ok.top1_differs_from_baseline.sum())}/{len(ok)}",
        "precision_at_k": "PENDING_HUMAN_REVIEW (see labeling template)",
    }
    for c in [c for c in qdf.columns if c.startswith("jaccard_") and c != "jaccard_default"]:
        out[f"mean_{c}_vs_default"] = round(float(ok[c].mean()), 3)
    for c in [c for c in qdf.columns if c.startswith("top1_same_") and c != "top1_same_default"]:
        out[f"{c}_vs_default"] = f"{int(ok[c].sum())}/{len(ok)}"
    return out


def _rate(a, b):
    return f"{int(a)}/{int(b)} = {a / b:.1%}" if b else "n/a"


# ---------------------------------------------------------------------------
# Baseline comparison (B1-B4) with latency. Label-free metrics only; label metrics below.
# ---------------------------------------------------------------------------
import time as _time

from .pipeline import nearest_any_baseline

SYSTEMS = ("B1_nearest_any", "B2_nearest_exact_word", "B3_nearest_synonym", "B4_careproof")


def _run_system(name, gold, silver, loc, svc, maxkm, k):
    if name == "B1_nearest_any":
        return nearest_any_baseline(silver, loc, maxkm, k)
    if name == "B2_nearest_exact_word":
        return keyword_baseline(gold, loc, svc, maxkm, k, column="baseline_exact_term_hit")
    if name == "B3_nearest_synonym":
        return keyword_baseline(gold, loc, svc, maxkm, k)
    return recommend(gold, loc, svc, maxkm, k)[0]


def compare_baselines(queries: pd.DataFrame, gold: pd.DataFrame, silver: pd.DataFrame, index, repeats: int = 3):
    """Per-system label-free metrics over answerable queries + latency (p50/p95 over repeats x queries)."""
    gsvc = {s: gold[gold.service == s] for s in gold.service.unique()}
    status = {(r.facility_id, r.service): (r.evidence_level, json.loads(r.flags_json), bool(r.eligible))
              for r in gold.itertuples()}
    rows, lat = [], {s: [] for s in SYSTEMS}
    for q in queries.to_dict("records"):
        svc = resolve_service(q["service_input"]).key
        try:
            loc = resolve_location(q["location_type"], str(q.get("location_value") or ""), q.get("latitude"),
                                   q.get("longitude"), index)
        except LocationError:
            continue
        maxkm, k = float(q["max_distance_km"]), int(q["top_k"])
        g = gsvc.get(svc, gold.iloc[0:0])
        for name in SYSTEMS:
            res = None
            for _ in range(repeats):
                t0 = _time.perf_counter()
                res = _run_system(name, g, silver, loc, svc, maxkm, k)
                lat[name].append((_time.perf_counter() - t0) * 1000)
            st = [status.get((r["facility_id"], svc), ("no_signal", [], False)) for r in res]
            lv, fl, el = [x[0] for x in st], [x[1] for x in st], [x[2] for x in st]
            rows.append({"query_id": q["query_id"], "system": name, "n_results": len(res),
                         "n_explicit_evidence": sum(v == "explicit_text" for v in lv),
                         "n_no_or_tag_only_evidence": sum(v in ("no_signal", "none", "specialty_tag_only") for v in lv),
                         "n_negated": sum(v == "negated_only" for v in lv),
                         "n_contradiction_flag": sum(bool(CONTRA & set(f)) for f in fl),
                         "n_unsupported": sum(not e for e in el),
                         "top1_explicit_clean": bool(res) and el[0]})
    per = pd.DataFrame(rows)
    agg = per.groupby("system").agg(queries=("query_id", "nunique"), results=("n_results", "sum"),
                                    coverage=("n_results", lambda x: (x > 0).mean()),
                                    explicit=("n_explicit_evidence", "sum"), weak=("n_no_or_tag_only_evidence", "sum"),
                                    negated=("n_negated", "sum"), contra=("n_contradiction_flag", "sum"),
                                    unsupported=("n_unsupported", "sum"),
                                    top1_clean=("top1_explicit_clean", "mean")).reindex(list(SYSTEMS))
    agg["unsupported_recommendation_rate_pct"] = (100 * agg.unsupported / agg.results).round(1)
    agg["pct_results_with_explicit_evidence"] = (100 * agg.explicit / agg.results).round(1)
    agg["pct_results_no_or_tag_only_evidence"] = (100 * agg.weak / agg.results).round(1)
    agg["pct_results_with_contradiction"] = (100 * agg.contra / agg.results).round(1)
    agg["pct_top1_passes_evidence_gates"] = (100 * agg.top1_clean).round(1)
    agg["coverage_pct"] = (100 * agg.coverage).round(1)
    agg["latency_ms_p50"] = [round(float(np.percentile(lat[s], 50)), 2) for s in agg.index]
    agg["latency_ms_p95"] = [round(float(np.percentile(lat[s], 95)), 2) for s in agg.index]
    keep = ["queries", "results", "coverage_pct", "unsupported_recommendation_rate_pct",
            "pct_results_with_explicit_evidence", "pct_results_no_or_tag_only_evidence", "negated",
            "pct_results_with_contradiction", "pct_top1_passes_evidence_gates", "latency_ms_p50", "latency_ms_p95"]
    return agg[keep].reset_index(), per


# ---------------------------------------------------------------------------
# Label-based metrics: computed ONLY from human labels; returns PENDING if none exist.
# ---------------------------------------------------------------------------
LABEL_COL = "label_service_supported"


def label_metrics(labels: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Precision@K and NDCG@K per system from a filled labeling template (yes=1, no=0, unclear excluded)."""
    lab = labels[labels[LABEL_COL].astype(str).str.strip().str.lower().isin(["yes", "no"])].copy()
    if lab.empty:
        return pd.DataFrame([{"status": "PENDING_HUMAN_REVIEW", "labeled_rows": 0}])
    lab["rel"] = (lab[LABEL_COL].str.strip().str.lower() == "yes").astype(int)
    lab = lab[lab["rank"].astype(int) <= k]
    out = []
    for (sysname, qid), d in lab.groupby(["system", "query_id"]):
        d = d.sort_values("rank")
        rel = d.rel.tolist()
        dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rel))
        idcg = sum(r / np.log2(i + 2) for i, r in enumerate(sorted(rel, reverse=True)))
        out.append({"system": sysname, "query_id": qid, "precision_at_k": np.mean(rel),
                    "ndcg_at_k_within_labeled": dcg / idcg if idcg else 0.0, "n_labeled": len(rel)})
    o = pd.DataFrame(out)
    return o.groupby("system").agg(queries=("query_id", "nunique"), labeled=("n_labeled", "sum"),
                                   precision_at_k=("precision_at_k", "mean"),
                                   ndcg_at_k=("ndcg_at_k_within_labeled", "mean")).round(3).reset_index()


# ---------------------------------------------------------------------------
# Unbiased query sample for labeling (frozen before results are inspected) + group-aware split.
# ---------------------------------------------------------------------------
LABELING_SEED = 20261121
SAMPLE_SERVICES = ("dialysis", "c-section", "nicu", "mri", "cardiac surgery", "burn care", "trauma")


def make_labeling_queries(pin_silver: pd.DataFrame, pin_raw: pd.DataFrame, per_cell: int = 2,
                          seed: int = LABELING_SEED, max_distance_km: float = 100.0, top_k: int = 5) -> pd.DataFrame:
    """Stratified random sample: service x district-rurality quartile x `per_cell` PINs.

    Only PINs with a defensible, unambiguous centroid are eligible (so the sample tests ranking, not geocoding;
    geocoding failures are covered by the curated benchmark). Split is by STATE (group-aware): all queries of a
    state fall in the same split, which prevents leakage through shared facilities and nearby locations.
    """
    from .eda import district_rurality_proxy
    rur = district_rurality_proxy(pin_raw)
    rur["quartile"] = pd.qcut(rur.bo_share.rank(method="first"), 4, labels=["Q1_urban", "Q2", "Q3", "Q4_rural"])
    p = pin_silver[pin_silver.coords_defensible.astype(bool) & ~pin_silver.is_ambiguous.astype(bool)]
    p = p.merge(rur, left_on=["primary_district", "primary_state"], right_on=["district", "state"])
    rng = np.random.default_rng(seed)
    states = sorted(p.primary_state.unique())
    rng.shuffle(states)
    test_states = set(states[: max(1, int(round(0.3 * len(states))))])
    rows, n = [], 0
    for svc in SAMPLE_SERVICES:
        for qt in ["Q1_urban", "Q2", "Q3", "Q4_rural"]:
            cell = p[p.quartile == qt].sort_values("pincode")
            pick = cell.iloc[rng.choice(len(cell), size=min(per_cell, len(cell)), replace=False)]
            for r in pick.itertuples():
                n += 1
                rows.append({"query_id": f"L{n:03d}", "category": f"random|{qt}", "service_input": svc,
                             "location_type": "pincode", "location_value": r.pincode, "latitude": "", "longitude": "",
                             "max_distance_km": max_distance_km, "top_k": top_k, "expected_outcome": "unknown",
                             "state": r.primary_state, "district": r.primary_district,
                             "split": "test" if r.primary_state in test_states else "dev",
                             "notes": f"seed={seed}; frozen before inspecting results"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Facility-service PAIR benchmark (manual review of evidence, identity, location, eligibility, quarantine)
# ---------------------------------------------------------------------------
PAIR_LABEL_COLUMNS = ["label_service_relevant", "label_evidence_strength", "label_identity_consistent",
                      "label_location_consistent", "label_eligible_for_recommendation", "label_quarantine_reason",
                      "reviewer_confidence", "reviewer", "reviewer_2", "notes"]


def _stratum(r) -> str:
    f = r["facility_findings_json"] + r["claim_findings_json"]
    if '"MULTIPLE_FACILITY_IDENTITIES"' in f:
        return "merged_or_misattributed"
    if '"SERVICE_CONTRADICTION"' in f:
        return "negation_or_non_current"
    if '"COORDINATE_LOCATION_CONFLICT"' in f or '"MULTIPLE_INCOMPATIBLE_LOCATIONS"' in f:
        return "geographic_contradiction"
    if '"UNRESOLVED_LOCATION"' in f:
        return "missing_coordinates"
    if '"AMBIGUOUS_PIN"' in f:
        return "ambiguous_pin"
    if r["evidence_level"] == "specialty_tag_only":
        return "hard_negative_tag_only"
    if r["evidence_level"] == "none":
        return "hard_negative_keyword_only"
    return f"evidence_{r['evidence_category']}"


def make_pair_benchmark(gold: pd.DataFrame, per_cell: int = 2, seed: int = LABELING_SEED) -> pd.DataFrame:
    """Seeded stratified sample of (facility, service) pairs: service x stratum, `per_cell` each.

    Returns IDs + stratum + split only (no source text) so the file can be committed; the review sheet with
    excerpts is produced at runtime by `pair_review_sheet`.
    """
    g = gold.copy()
    g["stratum"] = g.apply(_stratum, axis=1)
    rng = np.random.default_rng(seed)
    rows = []
    for (svc, stratum), d in sorted(g.groupby(["service", "stratum"]), key=lambda x: x[0]):
        d = d.sort_values("facility_id")
        pick = d.iloc[rng.choice(len(d), size=min(per_cell, len(d)), replace=False)]
        for r in pick.itertuples():
            rows.append({"pair_id": f"P{len(rows) + 1:04d}", "facility_id": r.facility_id, "service": svc,
                         "stratum": stratum, "system_eligible_at_sampling": bool(r.eligible)})
    out = pd.DataFrame(rows)
    st = sorted(gold.address_stateOrRegion.fillna("").unique())
    rng.shuffle(st)
    test_states = set(st[: int(round(0.3 * len(st)))])
    state = gold.set_index("facility_id").address_stateOrRegion.to_dict()
    out["split"] = out.facility_id.map(lambda f: "test" if state.get(f) in test_states else "dev")
    return out


def pair_review_sheet(pairs: pd.DataFrame, gold: pd.DataFrame, silver: pd.DataFrame) -> pd.DataFrame:
    """Blind review sheet: source evidence + system outputs hidden in separate columns reviewers should not see first."""
    g = gold.set_index(["facility_id", "service"])
    s = silver.set_index("facility_id")
    rows = []
    for p in pairs.itertuples():
        gr, sr = g.loc[(p.facility_id, p.service)], s.loc[p.facility_id]
        rows.append({"pair_id": p.pair_id, "split": p.split, "service": p.service, "facility_name": sr["name"],
                     "address": ", ".join(str(x) for x in [sr["address_line1"], sr["address_city"],
                                                          sr["address_stateOrRegion"], sr["address_zipOrPostcode"]]),
                     "description": sr["description"], "evidence_excerpt": gr["evidence_excerpt"],
                     "evidence_field": gr["evidence_field"], "source_url": sr["primary_source_url"],
                     **{c: "" for c in PAIR_LABEL_COLUMNS},
                     "HIDDEN_system_category": gr["evidence_category"], "HIDDEN_system_eligible": bool(gr["eligible"]),
                     "HIDDEN_system_failed_gates": gr["failed_gates_json"], "HIDDEN_stratum": p.stratum})
    return pd.DataFrame(rows).sample(frac=1, random_state=LABELING_SEED)


def pair_metrics(sheet: pd.DataFrame) -> dict:
    """Safety metrics from a reviewed pair sheet. Returns PENDING until labels exist."""
    lab = sheet[sheet["label_eligible_for_recommendation"].astype(str).str.lower().isin(["yes", "no"])]
    if lab.empty:
        return {"status": "PENDING_HUMAN_REVIEW", "labeled_pairs": 0}
    y = lab["label_eligible_for_recommendation"].str.lower() == "yes"
    p = lab["HIDDEN_system_eligible"].astype(str).str.lower() == "true"
    rel = lab["label_service_relevant"].astype(str).str.lower() == "yes"
    retrieved = lab["evidence_excerpt"].notna()
    return {
        "labeled_pairs": int(len(lab)),
        "unsupported_recommendation_rate": round(float((p & ~y).sum() / max(1, p.sum())), 3),
        "eligibility_precision": round(float((p & y).sum() / max(1, p.sum())), 3),
        "eligibility_recall": round(float((p & y).sum() / max(1, y.sum())), 3),
        "quarantine_precision": round(float((~p & ~y).sum() / max(1, (~p).sum())), 3),
        "quarantine_recall": round(float((~p & ~y).sum() / max(1, (~y).sum())), 3),
        "retrieval_recall_of_relevant": round(float((rel & retrieved).sum() / max(1, rel.sum())), 3),
    }
