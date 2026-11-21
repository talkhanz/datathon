"""Local KPI-driven weight optimisation (Bayesian TPE + NSGA-II GA). Usage: python scripts/run_optimization.py [n_trials]"""
import json, os, sys, time
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from careproof import optimization as O, pipeline as P  # noqa: E402
from careproof.config import DEFAULT_WEIGHTS  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
D = os.path.join(ROOT, "data", "derived")
gold = P.coerce_gold_types(pd.read_parquet(os.path.join(D, "careproof_facility_service_gold.parquet")))
places = P.place_index_from_pin_silver(P.coerce_pin_types(pd.read_parquet(os.path.join(D, "careproof_pincode_silver.parquet"))))
q = pd.read_csv(os.path.join(ROOT, "evaluation", "labeling_queries_v1.csv"), dtype=str, keep_default_na=False)
dev, test = O.QuerySet.from_frame(q[q.split == "dev"], places), O.QuerySet.from_frame(q[q.split == "test"], places)
base = P.RecommendationIndex(gold)
out = {}
for method in ("tpe", "nsga2"):
    t = time.time(); study = O.optimise(base, dev, method, n); dt = time.time() - t
    df = O.feasible_trials(study); front = O.pareto_front(df)
    out[method] = (study, df, front)
    print(f"{method}: {n} trials in {dt:.0f}s, feasible {int(df.feasible.sum())}/{len(df)}, pareto front {len(front)}")
tpe_best = O.weights_from_row(out["tpe"][1][out["tpe"][1].feasible].assign(u=lambda d: d.evidence_topk + d.consistency_topk - d.proximity_top1_km / 25).sort_values("u").iloc[-1])
allf = pd.concat([out["tpe"][1], out["nsga2"][1]]); front = O.pareto_front(allf); knee = O.weights_from_row(O.knee_point(front))
rows = []
for name, w in (("default (hand-set)", DEFAULT_WEIGHTS), ("bayesian_tpe_best_utility", tpe_best), ("nsga2_pareto_knee", knee)):
    idx = base.reweighted(w)
    for split, qs in (("dev", dev), ("test", test)):
        k = O.kpis(idx, qs)
        rows.append({"policy": name, "split": split, **{kk: round(v, 4) if isinstance(v, float) else v for kk, v in k.items()},
                     "utility": round(O.utility(k), 4), "top1_same_as_default": round(O.top1_agreement(idx, base, qs), 3),
                     "feasible": O.is_feasible(w)})
res = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(res.to_string(index=False))
print("\nweights:", json.dumps({"default": DEFAULT_WEIGHTS.normalized().as_dict(), "tpe_best": tpe_best.normalized().as_dict(),
                                "pareto_knee": knee.normalized().as_dict()}, indent=1, default=lambda x: round(x, 3)))
print(f"\ncombined Pareto front: {len(front)} policies")
print(front[["proximity_top1_km", "evidence_topk", "consistency_topk"] + [c for c in front if c.startswith("w_")]].round(3).head(12).to_string(index=False))
os.makedirs(os.path.join(ROOT, "evaluation"), exist_ok=True)
front.to_csv(os.path.join(ROOT, "evaluation", "pareto_front_LOCAL.csv"), index=False)
res.to_csv(os.path.join(ROOT, "evaluation", "policy_comparison_LOCAL.csv"), index=False)
