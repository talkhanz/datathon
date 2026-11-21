# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · KPI-driven policy optimisation (Bayesian TPE + NSGA-II genetic algorithm)
# MAGIC Tunes the six ranking weights **only among candidates that already passed the hard eligibility gates**.
# MAGIC Gates, evidence categories and quarantine are not tunable, and safety constraints (stronger evidence must beat
# MAGIC slightly-closer weaker evidence) are enforced as hard constraints. Weights are optimised on the **dev** states of
# MAGIC the frozen random query sample and reported on the held-out **test** states with paired-bootstrap 95% CIs.
# MAGIC KPIs are label-free (proximity, evidence strength, identity × location consistency); they are **not** ranking
# MAGIC correctness. `optimization.ndcg_objective` plugs human labels into the same loop when they exist.

# COMMAND ----------

# MAGIC %pip install optuna --quiet

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

create_widgets()
cfg = get_config()
from careproof import optimization as O
from careproof.config import DEFAULT_WEIGHTS, WEIGHT_SCENARIOS
config.TABLES["policy_optimization"] = "careproof_policy_optimization"
config.TABLES["pareto_front"] = "careproof_pareto_front"

gold = pipeline.coerce_gold_types(spark_to_pandas_str(spark.table(tbl(cfg, "facility_service_gold").replace("`", ""))))
places = pipeline.place_index_from_pin_silver(pipeline.coerce_pin_types(
    spark_to_pandas_str(spark.table(tbl(cfg, "pincode_silver").replace("`", "")))))
q = pd.read_csv(os.path.join(os.path.dirname(SRC), "evaluation", "labeling_queries_v1.csv"), dtype=str, keep_default_na=False)
dev, test = O.QuerySet.from_frame(q[q.split == "dev"], places), O.QuerySet.from_frame(q[q.split == "test"], places)
base = pipeline.RecommendationIndex(gold, WEIGHT_SCENARIOS["hand_set_v1"])
print(f"dev queries={len(dev.locs)} test queries={len(test.locs)}")

# COMMAND ----------

t0 = time.time()
studies = {m: O.optimise(base, dev, m, n_trials=150) for m in ("tpe", "nsga2")}
trials = pd.concat([O.feasible_trials(s).assign(method=m) for m, s in studies.items()], ignore_index=True)
front = O.pareto_front(trials)
print(f"optimisation: {time.time()-t0:.0f}s, feasible {int(trials.feasible.sum())}/{len(trials)}, Pareto front {len(front)}")
display(pandas_to_spark(front.round(4)))

# COMMAND ----------

# MAGIC %md ## Held-out test: hand-set vs optimised policies (paired bootstrap)

# COMMAND ----------

tpe = trials[(trials.method == "tpe") & trials.feasible].assign(u=lambda d: d.evidence_topk + d.consistency_topk - d.proximity_top1_km / 25)
candidates = {"hand_set_v1": WEIGHT_SCENARIOS["hand_set_v1"], "current_default (rounded TPE)": DEFAULT_WEIGHTS,
              "tpe_best_this_run": O.weights_from_row(tpe.sort_values("u").iloc[-1]),
              "nsga2_knee_this_run": O.weights_from_row(O.knee_point(front))}
ref = O.per_query_kpis(base, test)
rows = []
for name, w in candidates.items():
    idx = base.reweighted(w)
    k = O.kpis(idx, test)
    boot = O.paired_bootstrap(ref, O.per_query_kpis(idx, test)).set_index("kpi")
    rows.append({"policy": name, "feasible": O.is_feasible(w), **{kk: round(v, 4) if isinstance(v, float) else v for kk, v in k.items()},
                 "delta_top1_km": round(boot.loc["top1_km", "mean_delta"], 3),
                 "delta_top1_km_ci95": f"[{boot.loc['top1_km','ci95_low']:.2f}, {boot.loc['top1_km','ci95_high']:.2f}]",
                 "delta_evidence_ci95": f"[{boot.loc['evidence','ci95_low']:.3f}, {boot.loc['evidence','ci95_high']:.3f}]",
                 "delta_consistency_ci95": f"[{boot.loc['consistency','ci95_low']:.3f}, {boot.loc['consistency','ci95_high']:.3f}]",
                 "top1_same_as_hand_set": round(O.top1_agreement(idx, base, test), 3),
                 **{f"w_{n}": round(v, 3) for n, v in w.normalized().as_dict().items()}})
res = pd.DataFrame(rows)
display(pandas_to_spark(res))
safe_write(pandas_to_spark(res.astype(str)), cfg, "policy_optimization", "Policy optimisation results on held-out test states.")
safe_write(pandas_to_spark(front.round(5)), cfg, "pareto_front", "Feasible Pareto-optimal ranking-weight policies (dev states).")
