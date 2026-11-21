# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Evaluation
# MAGIC Runs the benchmark in `evaluation/benchmark_queries.csv` against CareProof and the
# MAGIC **nearest-facility-with-a-keyword-match** baseline, under several weight scenarios.
# MAGIC
# MAGIC * **Measured automatically:** citation correctness (excerpt is an exact substring of the cited field),
# MAGIC   geographic validity, coverage, expected handling of bad/ambiguous locations, ranking stability under
# MAGIC   weight changes, and how often the baseline's picks are weak or contradictory.
# MAGIC * **Not measured (needs humans):** precision@K. A labeling template is written to
# MAGIC   `careproof_eval_labeling_template` with `review_status = PENDING_HUMAN_REVIEW`. No precision number is
# MAGIC   reported until humans fill it in.

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

create_widgets()
cfg = get_config()
from careproof.evaluation import run_benchmark, summarize
config.TABLES["eval_labeling"] = "careproof_eval_labeling_template"
config.TABLES["eval_results_detail"] = "careproof_eval_results_detail"

bench_path = os.path.join(os.path.dirname(SRC), "evaluation", "benchmark_queries.csv")
queries = pd.read_csv(bench_path, dtype=str, keep_default_na=False)
print(f"{len(queries)} benchmark queries from {bench_path}")

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
for c in ("is_candidate", "baseline_keyword_hit"):
    gold[c] = gold[c] == "true"
silver = spark_to_pandas_str(spark.table(tbl(cfg, "facilities_silver").replace("`", "")))
for c in ("location_lat", "location_lon"):
    silver[c] = pd.to_numeric(silver[c], errors="coerce")

# COMMAND ----------

t0 = time.time()
qdf, rdf, ldf = run_benchmark(queries, gold, silver, index)
print(f"benchmark ran in {time.time()-t0:.0f}s")
display(pandas_to_spark(qdf))

# COMMAND ----------

# MAGIC %md ## Summary (measured values only)

# COMMAND ----------

summary = summarize(qdf)
print(json.dumps(summary, indent=1))

# COMMAND ----------

qdf["run_at"] = pd.Timestamp.now(tz="UTC").isoformat()
safe_write(pandas_to_spark(qdf), cfg, "eval_results", "CareProof benchmark per-query results (automatic metrics only).")
safe_write(pandas_to_spark(rdf), cfg, "eval_results_detail", "CareProof benchmark recommended rows with component scores.")
safe_write(pandas_to_spark(ldf), cfg, "eval_labeling", "Labeling template for precision@K. PENDING_HUMAN_REVIEW.")

# COMMAND ----------

# MAGIC %md ## Weight sensitivity: does the top result change when policy weights change?

# COMMAND ----------

stab_cols = ["query_id", "service"] + [c for c in qdf.columns if c.startswith(("jaccard_", "top1_same_"))]
display(pandas_to_spark(qdf[qdf.status == "ok"][stab_cols]))
