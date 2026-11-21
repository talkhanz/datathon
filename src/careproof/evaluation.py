"""Benchmark runner and automatic metrics. Human-judged metrics are exported as a labeling template."""
from __future__ import annotations

import json
from typing import Mapping

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
                          "outcome_as_expected": q["expected_outcome"] == "location_error",
                          "location_warning": None})
            continue
        maxkm, k = float(q["max_distance_km"]), int(q["top_k"])
        runs = {name: recommend(gold, loc, svc, maxkm, k, w)[0] for name, w in WEIGHT_SCENARIOS.items()}
        res = runs["default"]
        bl = keyword_baseline(gold, loc, svc, maxkm, k)
        exp_ok = (q["expected_outcome"] == "results" and len(res) > 0) or \
                 (q["expected_outcome"] == "results_with_warning" and len(res) > 0 and bool(loc.warnings))
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
                      "n_high": sum(r["confidence"] == "High" for r in res),
                      "n_low": sum(r["confidence"] == "Low" for r in res),
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
                    "distance_km": r["distance_km"], "confidence": r.get("confidence", "n/a (baseline)"),
                    "evidence_field": g["evidence_field"], "evidence_excerpt": g["evidence_excerpt"],
                    "flags": json.dumps(r["flags"]), "source_url": row.get("primary_source_url"),
                    "label_service_supported (yes/no/unclear)": "", "reviewer": "", "notes": "",
                    "review_status": "PENDING_HUMAN_REVIEW",
                })
        for r in res:
            rrows.append({"query_id": q["query_id"], **{k: r[k] for k in (
                "rank", "facility_id", "facility_name", "distance_km", "final_score", "confidence",
                "service_match_score", "evidence_confidence_score", "location_quality_score", "distance_score",
                "readiness_score", "data_quality_penalty", "evidence_field", "evidence_excerpt")}})
    return pd.DataFrame(qrows), pd.DataFrame(rrows), pd.DataFrame(label_rows)


def _in_india(row) -> bool:
    lat, lon = row.get("location_lat"), row.get("location_lon")
    return lat is not None and INDIA_LAT_RANGE[0] <= lat <= INDIA_LAT_RANGE[1] and INDIA_LON_RANGE[0] <= lon <= INDIA_LON_RANGE[1]


def summarize(qdf: pd.DataFrame) -> dict:
    ok = qdf[qdf.status == "ok"]
    ans = qdf[qdf.expected_outcome.isin(["results", "results_with_warning"])]
    out = {
        "queries_total": int(len(qdf)),
        "outcome_as_expected": f"{int(qdf.outcome_as_expected.sum())}/{len(qdf)}",
        "coverage_answerable_queries_with_results": f"{int((ans.n_results > 0).sum())}/{len(ans)}",
        "citation_exact_substring_rate": _rate(ok.citations_exact.sum(), ok.citations_total.sum()),
        "geographic_validity_rate": _rate(ok.geo_valid.sum(), ok.geo_total.sum()),
        "careproof_results_with_contradiction_flags": f"{int(ok.cp_with_contradiction_flags.sum())}/{int(ok.n_results.sum())}",
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
