"""Local end-to-end run on the CSV copies in data/raw/ (development only).

Usage:
    python scripts/run_local.py build                      # build silver + gold into data/derived/
    python scripts/run_local.py query dialysis pincode 800001 [max_km] [top_k]
    python scripts/run_local.py query mri coordinates 28.6139,77.2090
    python scripts/run_local.py model                      # train service-profile models + GA (after build)
    python scripts/run_local.py external                   # PM-JAY external check (after scripts/fetch_external.py)
"""
import json, os, sys, time
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from careproof import pipeline as P  # noqa: E402
from careproof.geography import resolve_location, LocationError  # noqa: E402
from careproof.service_matching import resolve_service  # noqa: E402

RAW, DER = os.path.join(ROOT, "data", "raw"), os.path.join(ROOT, "data", "derived")


def read_csv(name):
    # keep everything as strings; the literal "null" is handled by normalization
    return pd.read_csv(os.path.join(RAW, f"{name}.csv"), dtype=str, keep_default_na=False)


def to_parquet(df, path):
    """Parquet with object columns coerced to string/None (mixed bool/NaN columns break Arrow)."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            nn = out[c].dropna()
            if len(nn) and nn.map(lambda v: isinstance(v, bool)).all():
                out[c] = out[c].map(lambda v: None if v is None or v != v else bool(v)).astype("boolean")
            elif len(nn) and nn.map(lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)).all():
                out[c] = pd.to_numeric(out[c], errors="coerce")
            else:
                out[c] = out[c].map(lambda v: None if v is None or (isinstance(v, float) and v != v) else str(v))
    out.to_parquet(path, index=False)


def build():
    os.makedirs(DER, exist_ok=True)
    t = time.time()
    pin = P.build_pincode_silver(read_csv("india_post_pincode_directory"))
    print(f"pincode_silver rows={len(pin)} ambiguous={int(pin.is_ambiguous.sum())} "
          f"defensible={int(pin.coords_defensible.sum())} ({time.time()-t:.0f}s)")
    nfhs = P.build_nfhs_silver(read_csv("nfhs_5_district_health_indicators"))
    print(f"nfhs_silver rows={len(nfhs)} suppressed={int(nfhs.is_suppressed.sum())} small_sample={int(nfhs.is_small_sample.sum())}")
    fac, rejected, rep = P.build_facility_silver(read_csv("facilities"), pin)
    print(json.dumps({k: v for k, v in rep.items()}, default=str, indent=1))
    gold = P.build_facility_service_gold(fac)
    quar = P.build_quarantine(fac, gold, rejected)
    print("quarantine rows by reason/severity:\n", quar.groupby(["reason_code", "severity"]).size().to_string())
    print("gold eligible by service x category:\n",
          gold[gold.evidence_level != "none"].groupby(["service", "evidence_category"]).size().unstack(fill_value=0).to_string())
    print("eligible per service:", gold[gold.eligible].groupby("service").size().to_dict())
    print(f"gold rows={len(gold)} ({time.time()-t:.0f}s)")
    print(gold.groupby(["service", "evidence_level"]).size().unstack(fill_value=0))
    for n, d in [("pincode_silver", pin), ("nfhs_silver", nfhs), ("facilities_silver", fac),
                 ("facilities_rejected_rows", rejected), ("facilities_quarantine", quar),
                 ("facility_service_gold", gold)]:
        to_parquet(d, os.path.join(DER, f"careproof_{n}.parquet"))


def load(n):
    return pd.read_parquet(os.path.join(DER, f"careproof_{n}.parquet"))


def query(service, ltype, lvalue, max_km=100, top_k=5):
    sdef = resolve_service(service)
    pin = load("pincode_silver")
    idx = P.place_index_from_pin_silver(pin)
    lat = lon = None
    if ltype == "coordinates":
        lat, lon = lvalue.split(",")
    try:
        loc = resolve_location(ltype, lvalue, lat, lon, idx)
    except LocationError as e:
        print("LOCATION ERROR:", e); return
    gold = load("facility_service_gold")
    res, summ = P.recommend(gold, loc, sdef.key, float(max_km), int(top_k))
    print(json.dumps({k: v for k, v in summ.items() if k != "held_back"}, indent=1))
    for h in summ["held_back"][:5]:
        print("   HELD BACK:", h["facility_name"], h["distance_km"], "km", h["evidence_category"], h["failed_gates"], h["reasons"][:1])
    for r in res:
        print(f"\n#{r['rank']} {r['facility_name']} | {r['distance_km']} km | rank_score={r['final_ranking_score']} [{r['evidence_label']}]")
        print("  ", r["explanation"])
        for w in r["identity_warnings"] + r["location_warnings"] + r["missing_data_warnings"]:
            print("   !", w)
        print("   url:", r["source_url"])
    print("\nBASELINE nearest keyword match:")
    for b in P.keyword_baseline(gold, loc, sdef.key, float(max_km), int(top_k)):
        print("  ", b)


def model():
    from careproof import modeling as M
    t = time.time()
    out = M.train_all(load("facilities_silver"), load("facility_service_gold"))
    for name, df in out.items():
        to_parquet(df, os.path.join(DER, f"careproof_{name}.parquet"))
    m = out["model_metrics"]
    pd.set_option("display.width", 250)
    print(m[m.split == "test"].pivot_table(index="service", columns=["model", "feature_set"], values="average_precision").round(3))
    print(out["model_profile_summary"][["service", "model", "feature_set", "n_features", "test_average_precision",
                                        "test_base_rate", "profile_score", "nearest_real_facility_name"]].to_string())
    print(f"model tables written ({time.time()-t:.0f}s)")


def external():
    """Independent check against official PM-JAY empanelment lists (run scripts/fetch_external.py first)."""
    from careproof import external as E
    ref = pd.read_parquet(os.path.join(ROOT, "data", "external", "careproof_ext_pmjay_hospitals.parquet"))
    out = E.run(ref, load("facilities_silver"), load("facility_service_gold"), load("model_scores"))
    print("matches:", out["match_summary"])
    for k in ("external_matches", "external_validation", "external_validation_detail"):
        to_parquet(out[k], os.path.join(DER, f"careproof_{k}.parquet"))
    v = out["external_validation"]
    pd.set_option("display.width", 250)
    print(v[["service", "system", "n_flagged", "empanelled_rate", "ci95_low", "ci95_high", "base_rate"]].round(2).to_string())


if __name__ == "__main__":
    if sys.argv[1] in ("eval", "model", "external"):
        if sys.argv[1] == "model":
            model()
        elif sys.argv[1] == "external":
            external()
    elif sys.argv[1] == "build":
        build()
    else:
        query(*sys.argv[2:])


def evaluate():
    from careproof.evaluation import (compare_baselines, label_metrics, make_labeling_queries, run_benchmark,
                                      summarize)
    pin, gold, silver = load("pincode_silver"), load("facility_service_gold"), load("facilities_silver")
    idx = P.place_index_from_pin_silver(pin)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70); pd.set_option("display.max_columns", 20)
    # (1) curated smoke benchmark: behaviour on hand-picked edge cases (NOT an unbiased estimate)
    q = pd.read_csv(os.path.join(ROOT, "evaluation", "benchmark_queries.csv"), dtype=str, keep_default_na=False)
    qdf, rdf, ldf = run_benchmark(q, gold, silver, idx)
    print(qdf[["query_id", "category", "service", "status", "n_results", "outcome_as_expected", "n_strong", "n_moderate", "n_weak", "held_back",
               "baseline_top1_weak", "message"]].to_string())
    print("CURATED BENCHMARK SUMMARY:", json.dumps(summarize(qdf), indent=1))
    # (2) frozen, seeded, stratified random sample (the unbiased set for label-free metrics + human labels)
    lq_path = os.path.join(ROOT, "evaluation", "labeling_queries_v1.csv")
    if not os.path.exists(lq_path):
        make_labeling_queries(pin, read_csv("india_post_pincode_directory")).to_csv(lq_path, index=False)
        print("froze", lq_path)
    lq = pd.read_csv(lq_path, dtype=str, keep_default_na=False)
    for name, qs in (("CURATED", q), ("RANDOM_ALL", lq), ("RANDOM_DEV", lq[lq.split == "dev"]),
                     ("RANDOM_TEST", lq[lq.split == "test"])):
        agg, _ = compare_baselines(qs, gold, silver, idx)
        print(f"\nBASELINE COMPARISON [{name}] ({len(qs)} queries)\n", agg.to_string(index=False))
    lqdf, _, lldf = run_benchmark(lq, gold, silver, idx)
    print("\nRANDOM SAMPLE status counts:", lqdf.status.value_counts().to_dict(),
          "| zero-result queries by quartile:",
          lqdf.assign(q=lqdf.category.str.split("|").str[1]).groupby("q").n_results.apply(lambda x: int((x == 0).sum())).to_dict())
    print("RANDOM SAMPLE SUMMARY:", json.dumps(summarize(lqdf), indent=1))
    out = os.path.join(ROOT, "evaluation", "labeling_template_LOCAL.csv")
    lldf.sample(frac=1, random_state=1).to_csv(out, index=False)
    print("labeling template (shuffled, for blind review) ->", out, len(lldf), "rows")
    print("label metrics:", label_metrics(lldf).to_dict("records"))
    from careproof.evaluation import make_pair_benchmark, pair_metrics, pair_review_sheet
    pb_path = os.path.join(ROOT, "evaluation", "pair_benchmark_v1.csv")
    if not os.path.exists(pb_path):
        make_pair_benchmark(gold).to_csv(pb_path, index=False)
        print("froze", pb_path)
    pairs = pd.read_csv(pb_path, dtype=str, keep_default_na=False)
    sheet = pair_review_sheet(pairs, gold, silver)
    sheet.to_csv(os.path.join(ROOT, "evaluation", "pair_review_sheet_LOCAL.csv"), index=False)
    print(f"pair benchmark: {len(pairs)} pairs; strata:", pairs.stratum.value_counts().to_dict())
    print("pair metrics:", pair_metrics(sheet))


if __name__ == "__main__" and sys.argv[1] == "eval":
    evaluate()
