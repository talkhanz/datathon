"""Local end-to-end run on the CSV copies in data/raw/ (development only).

Usage:
    python scripts/run_local.py build                      # build silver + gold into data/derived/
    python scripts/run_local.py query dialysis pincode 800001 [max_km] [top_k]
    python scripts/run_local.py query mri coordinates 28.6139,77.2090
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


def build():
    os.makedirs(DER, exist_ok=True)
    t = time.time()
    pin = P.build_pincode_silver(read_csv("india_post_pincode_directory"))
    print(f"pincode_silver rows={len(pin)} ambiguous={int(pin.is_ambiguous.sum())} "
          f"defensible={int(pin.coords_defensible.sum())} ({time.time()-t:.0f}s)")
    nfhs = P.build_nfhs_silver(read_csv("nfhs_5_district_health_indicators"))
    print(f"nfhs_silver rows={len(nfhs)} suppressed={int(nfhs.is_suppressed.sum())} small_sample={int(nfhs.is_small_sample.sum())}")
    fac, quar, rep = P.build_facility_silver(read_csv("facilities"), pin)
    print(json.dumps({k: v for k, v in rep.items()}, default=str, indent=1))
    gold = P.build_facility_service_gold(fac)
    print(f"gold rows={len(gold)} ({time.time()-t:.0f}s)")
    print(gold.groupby(["service", "evidence_level"]).size().unstack(fill_value=0))
    for n, d in [("pincode_silver", pin), ("nfhs_silver", nfhs), ("facilities_silver", fac),
                 ("facilities_quarantine", quar), ("facility_service_gold", gold)]:
        d.to_pickle(os.path.join(DER, f"careproof_{n}.pkl"))


def load(n):
    return pd.read_pickle(os.path.join(DER, f"careproof_{n}.pkl"))


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
    print(json.dumps(summ, indent=1))
    for r in res:
        print(f"\n#{r['rank']} {r['facility_name']} | {r['distance_km']} km | final={r['final_score']} [{r['confidence']}]")
        print("  ", r["explanation"])
        for w in r["missing_data_warnings"] + r["contradiction_warnings"]:
            print("   !", w)
        print("   url:", r["source_url"])
    print("\nBASELINE nearest keyword match:")
    for b in P.keyword_baseline(gold, loc, sdef.key, float(max_km), int(top_k)):
        print("  ", b)


if __name__ == "__main__":
    if sys.argv[1] == "eval":
        pass
    elif sys.argv[1] == "build":
        build()
    else:
        query(*sys.argv[2:])


def evaluate():
    from careproof.evaluation import run_benchmark, summarize
    pin, gold, silver = load("pincode_silver"), load("facility_service_gold"), load("facilities_silver")
    idx = P.place_index_from_pin_silver(pin)
    q = pd.read_csv(os.path.join(ROOT, "evaluation", "benchmark_queries.csv"), dtype=str, keep_default_na=False)
    qdf, rdf, ldf = run_benchmark(q, gold, silver, idx)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
    print(qdf[["query_id", "category", "service", "status", "n_results", "outcome_as_expected", "n_high", "n_low",
               "baseline_top1_weak", "top1_differs_from_baseline", "message"]].to_string())
    print(json.dumps(summarize(qdf), indent=1))
    out = os.path.join(ROOT, "evaluation", "labeling_template_LOCAL.csv")
    ldf.to_csv(out, index=False)
    print("labeling template ->", out, len(ldf), "rows")


if __name__ == "__main__" and sys.argv[1] == "eval":
    evaluate()
