# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Build gold + quarantine
# MAGIC * `careproof_facility_service_gold`: one row per (facility, service) with any signal. It holds the exact evidence,
# MAGIC   an evidence category (strong / moderate / weak / conflicting / insufficient), the identity and location
# MAGIC   confidence, and the **hard eligibility gates** (`eligible`, `failed_gates_json`). Only `eligible = true` rows can
# MAGIC   ever be recommended.
# MAGIC * `careproof_facilities_quarantine`: one row per (record, [service], reason code, severity, evidence), used for
# MAGIC   human review and the planner/audit view.

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
for c in ("location_lat", "location_lon", "location_quality_score", "facility_to_pin_km",
          "identity_confidence_score", "location_confidence_score"):
    silver[c] = pd.to_numeric(silver[c], errors="coerce")
print("silver rows:", len(silver))
gold = pipeline.build_facility_service_gold(silver)
print(f"gold rows={len(gold)} built in {time.time()-t0:.0f}s")
display(gold[gold.evidence_level != "none"].groupby(["service", "evidence_category"]).size().unstack(fill_value=0).reset_index())
print("eligible per service:", gold[gold.eligible].groupby("service").size().to_dict())

# COMMAND ----------

# Fanout / integrity checks before writing
assert len(gold.drop_duplicates(["facility_id", "service"])) == len(gold), "duplicate (facility_id, service)"
assert set(gold.facility_id) <= set(silver.facility_id), "gold references unknown facility_id"
elig = gold[gold.eligible]
assert elig.evidence_excerpt.fillna("").str.strip().ne("").all(), "eligible row without excerpt"
assert not elig.evidence_category.isin(["conflicting", "insufficient"]).any(), "conflicting/insufficient row marked eligible"
for c in ("service_relevance_score", "evidence_strength_score", "readiness_evidence_score", "location_confidence_score",
          "identity_confidence_score"):
    assert gold[c].between(0, 1).all(), c
safe_write(pandas_to_spark(gold), cfg, "facility_service_gold",
           "Facility x service evidence with exact excerpts, component scores and data-quality flags. Derived by CareProof.")


# COMMAND ----------

rejected = (spark_to_pandas_str(spark.table(tbl(cfg, "facilities_rejected_rows").replace("`", "")))
            if table_exists(tbl(cfg, "facilities_rejected_rows")) else
            pd.DataFrame(columns=["source_row_id", "quarantine_reason", "unique_id"]))
quarantine = pipeline.build_quarantine(silver, gold, rejected)
display(quarantine.groupby(["reason_code", "severity"]).size().reset_index(name="rows"))
safe_write(pandas_to_spark(quarantine), cfg, "facilities_quarantine",
           "Quarantine reasons (record/service, reason code, severity, evidence, human review). Derived by CareProof.")
