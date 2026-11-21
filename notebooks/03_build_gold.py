# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Build gold: `careproof_facility_service_gold`
# MAGIC One row per (facility, service) where the facility has *any* signal for the service: explicit text
# MAGIC evidence, a related specialty tag only, a negated mention only, or a naive keyword hit (kept so the
# MAGIC baseline comparison is fair). Only `is_candidate = true` rows (explicit, non-negated text) can be
# MAGIC recommended. Matching is deterministic and identical to the locally tested `careproof.service_matching`.

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

create_widgets()
cfg = get_config()
if not cfg.rebuild_tables and table_exists(tbl(cfg, "facility_service_gold")):
    dbutils.notebook.exit("Gold table already exists; set rebuild_tables=yes to rebuild. SKIPPED.")

# COMMAND ----------

t0 = time.time()
silver = spark_to_pandas_str(spark.table(tbl(cfg, "facilities_silver").replace("`", "")))
for c in ("location_lat", "location_lon", "location_quality_score", "facility_to_pin_km"):
    silver[c] = pd.to_numeric(silver[c], errors="coerce")
print("silver rows:", len(silver))
gold = pipeline.build_facility_service_gold(silver)
print(f"gold rows={len(gold)} built in {time.time()-t0:.0f}s")
display(gold.groupby(["service", "evidence_level"]).size().unstack(fill_value=0).reset_index())

# COMMAND ----------

# Fanout / integrity checks before writing
assert len(gold.drop_duplicates(["facility_id", "service"])) == len(gold), "duplicate (facility_id, service)"
assert set(gold.facility_id) <= set(silver.facility_id), "gold references unknown facility_id"
cand = gold[gold.is_candidate]
assert cand.evidence_excerpt.fillna("").str.strip().ne("").all(), "candidate without excerpt"
for c in ("service_match_score", "evidence_confidence_score", "readiness_score", "location_quality_score"):
    assert gold[c].between(0, 1).all(), c
safe_write(pandas_to_spark(gold), cfg, "facility_service_gold",
           "Facility x service evidence with exact excerpts, component scores and data-quality flags. Derived by CareProof.")
