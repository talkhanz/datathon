# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Evaluation, baselines, latency
# MAGIC Two query sets, used for different purposes:
# MAGIC * `evaluation/benchmark_queries.csv` is **curated** edge cases (ambiguous PIN, swapped coordinates, merged record…).
# MAGIC   It tests *behaviour* and is **not** an unbiased performance estimate.
# MAGIC * `evaluation/labeling_queries_v1.csv` is a **seeded, stratified random sample** (service × district rurality),
# MAGIC   frozen before results were inspected and split dev/test **by state**. Unbiased label-free metrics and the human
# MAGIC   labeling template come from this set.
# MAGIC
# MAGIC Systems: **B1** nearest facility (any), **B2** nearest exact-word match, **B3** nearest synonym match,
# MAGIC **B4** CareProof.
# MAGIC Label-free metrics measure **evidence-policy compliance and behaviour**, not correctness. Precision@K / NDCG@K
# MAGIC need human labels and are reported as **PENDING_HUMAN_REVIEW** until the template is filled in.

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

create_widgets()
cfg = get_config()
from careproof.evaluation import compare_baselines, label_metrics, run_benchmark, summarize
for k, v in {"eval_labeling": "careproof_eval_labeling_template", "eval_results_detail": "careproof_eval_results_detail",
             "baseline_comparison": "careproof_baseline_comparison"}.items():
    config.TABLES[k] = v

root = os.path.dirname(SRC)
curated = pd.read_csv(os.path.join(root, "evaluation", "benchmark_queries.csv"), dtype=str, keep_default_na=False)
random_q = pd.read_csv(os.path.join(root, "evaluation", "labeling_queries_v1.csv"), dtype=str, keep_default_na=False)
print(f"curated={len(curated)} random={len(random_q)} (dev={int((random_q.split=='dev').sum())}, test={int((random_q.split=='test').sum())})")

t0 = time.time()
pin = spark_to_pandas_str(spark.table(tbl(cfg, "pincode_silver").replace("`", "")))
for c in ("centroid_lat", "centroid_lon", "n_offices_with_valid_coords", "n_offices_coord_outliers", "post_office_count"):
    pin[c] = pd.to_numeric(pin[c])
for c in ("coords_defensible", "is_ambiguous"):
    pin[c] = pin[c] == "true"
index = pipeline.place_index_from_pin_silver(pin)
gold = spark_to_pandas_str(spark.table(tbl(cfg, "facility_service_gold").replace("`", "")))
for c in ("location_lat", "location_lon", "location_quality_score", "service_match_score", "evidence_confidence_score",
          "readiness_score", "facility_to_pin_km"):
    gold[c] = pd.to_numeric(gold[c], errors="coerce")
for c in ("is_candidate", "baseline_keyword_hit", "baseline_exact_term_hit"):
    gold[c] = gold[c] == "true"
silver = spark_to_pandas_str(spark.table(tbl(cfg, "facilities_silver").replace("`", "")))
for c in ("location_lat", "location_lon"):
    silver[c] = pd.to_numeric(silver[c], errors="coerce")
print(f"cold load of reference tables: {time.time()-t0:.1f}s")

# COMMAND ----------

# MAGIC %md ## 1. Curated behaviour benchmark

# COMMAND ----------

qdf, rdf, _ = run_benchmark(curated, gold, silver, index)
display(pandas_to_spark(qdf.astype(str)))
print(json.dumps(summarize(qdf), indent=1))

# COMMAND ----------

# MAGIC %md ## 2. Unbiased random sample: CareProof vs baselines (label-free) + latency (p50/p95, warm, per query)

# COMMAND ----------

comparisons = []
for name, qs in (("random_all", random_q), ("random_dev", random_q[random_q.split == "dev"]),
                 ("random_test", random_q[random_q.split == "test"]), ("curated", curated)):
    agg, _ = compare_baselines(qs, gold, silver, index)
    agg.insert(0, "query_set", name)
    comparisons.append(agg)
cmp_df = pd.concat(comparisons, ignore_index=True)
display(pandas_to_spark(cmp_df))
lqdf, lrdf, lldf = run_benchmark(random_q, gold, silver, index)
print("RANDOM SAMPLE:", json.dumps(summarize(lqdf), indent=1))

# COMMAND ----------

# MAGIC %md ## 3. Persist results and the blind labeling template

# COMMAND ----------

run_at = pd.Timestamp.now(tz="UTC").isoformat()
safe_write(pandas_to_spark(pd.concat([qdf.assign(query_set="curated"), lqdf.assign(query_set="random")]).astype(str)
                           .assign(run_at=run_at)), cfg, "eval_results", "CareProof per-query evaluation (automatic metrics only).")
safe_write(pandas_to_spark(pd.concat([rdf.assign(query_set="curated"), lrdf.assign(query_set="random")])),
           cfg, "eval_results_detail", "CareProof recommended rows with component scores.")
safe_write(pandas_to_spark(cmp_df.assign(run_at=run_at)), cfg, "baseline_comparison",
           "B1-B4 label-free comparison and latency. Not a correctness measure.")
lab = lldf.merge(random_q[["query_id", "split"]], on="query_id").sample(frac=1, random_state=1)
safe_write(pandas_to_spark(lab), cfg, "eval_labeling", "Blind labeling template (random sample). PENDING_HUMAN_REVIEW.")
print("label metrics:", label_metrics(lab).to_dict("records"))

# COMMAND ----------

# MAGIC %md ## 4. Weight sensitivity on the random sample (does the top result survive alternative policies?)

# COMMAND ----------

stab_cols = ["query_id", "service"] + [c for c in lqdf.columns if c.startswith(("jaccard_", "top1_same_"))]
display(pandas_to_spark(lqdf[lqdf.status == "ok"][stab_cols]))
