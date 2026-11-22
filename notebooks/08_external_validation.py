# Databricks notebook source
# MAGIC %md
# MAGIC # 08 · External validation: official AB-PMJAY empanelment lists (Haryana State Health Agency)
# MAGIC An **independent, real** reference. Hospitals are empanelled for specific specialities after an inspection by
# MAGIC the State Health Agency. We match Virtue Foundation records to these lists by name and district (precision first)
# MAGIC and measure how often each system's flagged hospitals are empanelled for the matching speciality.
# MAGIC "Not empanelled" is weak negative evidence, so results are **support rates**, not accuracy. See `docs/external_validation.md`.
# MAGIC
# MAGIC **Input:** `careproof_ext_pmjay_hospitals.parquet`, produced by `scripts/fetch_external.py` (official PDFs, contact
# MAGIC columns removed) and uploaded to the project-owned volume `careproof_external`.

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

create_widgets()
cfg = get_config()
from careproof import external as E
vol = f"{cfg.target_catalog}.{cfg.target_schema}.careproof_external"
spark.sql(f"CREATE VOLUME IF NOT EXISTS {vol}")  # project-owned, inside the target schema
path = f"/Volumes/{cfg.target_catalog}/{cfg.target_schema}/careproof_external/careproof_ext_pmjay_hospitals.parquet"
ref = pd.read_parquet(path)
print(f"reference hospitals: {len(ref)} ({int(ref.has_specialities.sum())} with specialities)")

silver = spark_to_pandas_str(spark.table(tbl(cfg, "facilities_silver").replace("`", "")))
gold = pipeline.coerce_gold_types(spark_to_pandas_str(spark.table(tbl(cfg, "facility_service_gold").replace("`", ""))))
scores = spark_to_pandas_str(spark.table(tbl(cfg, "model_scores").replace("`", "")))
scores["model_score"] = pd.to_numeric(scores.model_score)

# COMMAND ----------

out = E.run(ref, silver, gold, scores)
print("matches:", out["match_summary"])
v = out["external_validation"]
display(pandas_to_spark(v.round(4)))

# COMMAND ----------

safe_write(pandas_to_spark(ref), cfg, "external_reference", "AB-PMJAY Haryana empanelment reference (official lists; no contact data).")
safe_write(pandas_to_spark(out["external_matches"]), cfg, "external_matches", "Precision-first name+district matches to PM-JAY hospitals.")
safe_write(pandas_to_spark(v), cfg, "external_validation", "Support rates vs PM-JAY empanelment per system, Wilson 95% CIs.")
safe_write(pandas_to_spark(out["external_validation_detail"]), cfg, "external_validation_detail", "Per matched hospital x service detail.")
