# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Build silver tables
# MAGIC Reads the read-only Marketplace tables (bronze) and writes three project-owned Delta tables:
# MAGIC `careproof_pincode_silver` (PIN grain), `careproof_nfhs_silver` (long format), `careproof_facilities_silver`
# MAGIC (+ `careproof_facilities_rejected_rows`, `careproof_join_report`). Idempotent: every write is a full overwrite
# MAGIC of a project-owned table. Set `rebuild_tables = yes` to rebuild tables that already exist.

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

create_widgets()
cfg = get_config()
print("Target schema:", f"`{cfg.target_catalog}`.`{cfg.target_schema}`", "| rebuild_tables:", cfg.rebuild_tables)
needed = ["pincode_silver", "nfhs_silver", "facilities_silver"]
if not cfg.rebuild_tables and all(table_exists(tbl(cfg, k)) for k in needed):
    dbutils.notebook.exit("Silver tables already exist; set rebuild_tables=yes to rebuild. SKIPPED.")

# COMMAND ----------

t0 = time.time()
pin_raw = spark_to_pandas_str(spark.table(SOURCE_TABLES["pincode"].replace("`", "")))
print("pincode bronze rows:", len(pin_raw))
pin_silver = pipeline.build_pincode_silver(pin_raw)
print(f"pincode silver rows={len(pin_silver)} ambiguous={int(pin_silver.is_ambiguous.sum())} "
      f"defensible_coords={int(pin_silver.coords_defensible.sum())} missing_coords={int(pin_silver.missing_coordinates.sum())} "
      f"({time.time()-t0:.0f}s)")
safe_write(pandas_to_spark(pin_silver), cfg, "pincode_silver",
           "PIN-level summary of India Post directory (post-office grain collapsed). Derived by CareProof.")

# COMMAND ----------

nfhs_raw = spark_to_pandas_str(spark.table(SOURCE_TABLES["nfhs"].replace("`", "")))
nfhs_silver = pipeline.build_nfhs_silver(nfhs_raw)
print(f"nfhs silver rows={len(nfhs_silver)} (districts={nfhs_raw.shape[0]} x indicators={nfhs_raw.shape[1]-2}) "
      f"suppressed={int(nfhs_silver.is_suppressed.sum())} small_sample={int(nfhs_silver.is_small_sample.sum())} "
      f"unrecognised_state_rows={int(nfhs_silver.state_norm.isna().sum())}")
safe_write(pandas_to_spark(nfhs_silver), cfg, "nfhs_silver",
           "NFHS-5 district indicators, long format. District-level community context only, never patient-level or facility quality.")

# COMMAND ----------

fac_raw = spark_to_pandas_str(spark.table(SOURCE_TABLES["facilities"].replace("`", "")))
fac_silver, rejected, rep = pipeline.build_facility_silver(fac_raw, pin_silver)
print(json.dumps(rep, indent=1, default=str))
safe_write(pandas_to_spark(fac_silver), cfg, "facilities_silver",
           "Facilities with original evidence fields preserved plus normalized location, PIN cross-checks and flags.")
if len(rejected):
    safe_write(pandas_to_spark(rejected), cfg, "facilities_rejected_rows", "Structurally corrupted source rows excluded from silver (original content preserved).")
else:
    print("No structurally corrupted rows.")

# COMMAND ----------

# MAGIC %md ## Join report (NFHS ↔ PIN districts is informational; it is never joined onto facility rows)

# COMMAND ----------

pin_d = pin_silver[pin_silver.primary_district.notna()][["primary_district", "primary_state"]].drop_duplicates()
nf_d = nfhs_silver[["district_norm", "state_norm"]].drop_duplicates()
m = nf_d.merge(pin_d, how="left", left_on=["district_norm", "state_norm"], right_on=["primary_district", "primary_state"], indicator=True)
nfhs_rep = {"name": "nfhs districts x pin districts (lookup only)", "left_rows": len(nf_d), "right_rows": len(pin_d),
            "output_rows": len(m), "matched_rows": int((m._merge == "both").sum()), "right_duplicate_keys": 0}
nfhs_rep["match_rate"] = round(nfhs_rep["matched_rows"] / nfhs_rep["left_rows"], 4)
nfhs_rep["duplicate_key_rate"] = 0.0
nfhs_rep["fanout_rows"] = nfhs_rep["output_rows"] - nfhs_rep["left_rows"]
if nfhs_rep["fanout_rows"]:
    raise pipeline.FanoutError(f"Unexpected NFHS district fanout: {nfhs_rep}")
print(nfhs_rep)
print("Unmatched NFHS districts (sample):", m[m._merge == "left_only"][["district_norm", "state_norm"]].head(15).values.tolist())

jr = pd.DataFrame([rep["join_pincode"], nfhs_rep])
jr["built_at"] = pd.Timestamp.now(tz="UTC").isoformat()
safe_write(pandas_to_spark(jr), cfg, "join_report", "Join reconciliation for CareProof builds.")
display(spark.table(tbl(cfg, "join_report").replace("`", "")))
print(f"Silver build complete in {time.time()-t0:.0f}s")
