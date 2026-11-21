# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Data audit (read-only)
# MAGIC Profiles the authoritative Marketplace tables and compares them with the local CSV copies used for
# MAGIC development. Nothing is written. Paste the **SUMMARY** cell output back to the team.

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

create_widgets()
cfg = get_config()
fac = spark.table(SOURCE_TABLES["facilities"].replace("`", ""))
pin = spark.table(SOURCE_TABLES["pincode"].replace("`", ""))
nfhs = spark.table(SOURCE_TABLES["nfhs"].replace("`", ""))

# Columns observed in the local CSV export (data/raw/*.csv) on 2026-11-21.
LOCAL_COLUMNS = {
    "facilities": ["unique_id", "source_types", "source_ids", "source_content_id", "name", "organization_type",
                   "content_table_id", "phone_numbers", "officialPhone", "email", "websites", "officialWebsite",
                   "yearEstablished", "acceptsVolunteers", "facebookLink", "address_line1", "address_line2",
                   "address_line3", "address_city", "address_stateOrRegion", "address_zipOrPostcode",
                   "address_country", "address_countryCode", "countries", "facilityTypeId", "operatorTypeId",
                   "affiliationTypeIds", "description", "area", "numberDoctors", "capacity", "specialties",
                   "procedure", "equipment", "capability", "recency_of_page_update",
                   "distinct_social_media_presence_count", "affiliated_staff_presence", "custom_logo_presence",
                   "number_of_facts_about_the_organization", "post_metrics_most_recent_social_media_post_date",
                   "post_metrics_post_count", "engagement_metrics_n_followers", "engagement_metrics_n_likes",
                   "engagement_metrics_n_engagements", "source", "coordinates", "latitude", "longitude",
                   "cluster_id", "source_urls"],
    "pincode": ["circlename", "regionname", "divisionname", "officename", "pincode", "officetype", "delivery",
                "district", "statename", "latitude", "longitude"],
}
LOCAL_ROWS = {"facilities": 10088, "pincode": 165627, "nfhs": 706}

# COMMAND ----------

# MAGIC %md ## Schema comparison: Databricks (authoritative) vs local CSV

# COMMAND ----------

schema_diffs = {}
for key, df in [("facilities", fac), ("pincode", pin), ("nfhs", nfhs)]:
    cols = df.columns
    print(f"\n=== {key}: rows={df.count()} (local CSV rows={LOCAL_ROWS[key]}) cols={len(cols)}")
    print("types:", {f.name: f.dataType.simpleString() for f in df.schema.fields if f.dataType.simpleString() != "string"}
          or "all string")
    if key in LOCAL_COLUMNS:
        missing = [c for c in LOCAL_COLUMNS[key] if c not in cols]
        extra = [c for c in cols if c not in LOCAL_COLUMNS[key]]
        schema_diffs[key] = {"missing_vs_local": missing, "extra_vs_local": extra}
        print("missing vs local:", missing, "| extra vs local:", extra)

# COMMAND ----------

# MAGIC %md ## Missingness (true NULL, empty, or literal null-like tokens)

# COMMAND ----------

def missingness(df):
    n = df.count()
    exprs = []
    for f in df.schema.fields:
        c = F.col(f"`{f.name}`")
        s = F.trim(F.to_json(c) if f.dataType.simpleString().startswith(("array", "struct", "map")) else c.cast("string"))
        miss = c.isNull() | F.lower(s).isin("", "null", "na", "none", "nan", "[]")
        exprs.append(F.round(F.avg(miss.cast("int")) * 100, 1).alias(f.name))
    row = df.agg(*exprs).first().asDict()
    return spark.createDataFrame([(k, float(v)) for k, v in row.items()], "column string, pct_missing double")

display(missingness(fac).orderBy(F.desc("pct_missing")))

# COMMAND ----------

# MAGIC %md ## Facilities: identifiers, corruption, duplicates, coordinates, PIN format

# COMMAND ----------

fac.createOrReplaceTempView("cp_fac")
display(spark.sql(r"""
SELECT
  COUNT(*)                                                                    AS rows,
  COUNT(DISTINCT unique_id)                                                   AS distinct_unique_id,
  SUM(CASE WHEN NOT CAST(unique_id AS STRING) RLIKE '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$' THEN 1 ELSE 0 END) AS non_uuid_unique_id,
  COUNT(*) - COUNT(DISTINCT to_json(struct(*)))                               AS exact_duplicate_rows,
  SUM(CASE WHEN try_cast(latitude AS DOUBLE) IS NULL THEN 1 ELSE 0 END)       AS lat_null_or_unparseable,
  SUM(CASE WHEN try_cast(latitude AS DOUBLE) NOT BETWEEN 6 AND 37.6
             OR try_cast(longitude AS DOUBLE) NOT BETWEEN 68 AND 97.5 THEN 1 ELSE 0 END) AS coords_outside_india,
  SUM(CASE WHEN regexp_replace(CAST(address_zipOrPostcode AS STRING), '[ -]', '') RLIKE '^[1-9][0-9]{5}$' THEN 0 ELSE 1 END) AS pin_not_6_digit,
  COUNT(DISTINCT cluster_id)                                                  AS distinct_cluster_id,
  COUNT(DISTINCT content_table_id)                                            AS distinct_content_table_id
FROM cp_fac
"""))

# COMMAND ----------

display(spark.sql("""
SELECT facilityTypeId, operatorTypeId, organization_type, COUNT(*) n
FROM cp_fac GROUP BY ALL ORDER BY n DESC LIMIT 30"""))

# COMMAND ----------

# MAGIC %md ## Service-related fields: representative values

# COMMAND ----------

display(fac.select("unique_id", "name", "specialties", "procedure", "equipment", "capability", "description").limit(5))

# COMMAND ----------

terms = {"dialysis": "dialys", "c_section": "c-section|caesarean|cesarean|lscs", "nicu": "nicu|neonatal intensive",
         "mri": "\\\\bmri\\\\b|magnetic resonance", "cardiac_surgery": "cardiac surg|cabg|open heart|cardiothoracic",
         "burn": "\\\\bburns?\\\\b", "trauma": "trauma|casualty|emergency"}
rows = []
for k, p in terms.items():
    r = spark.sql(f"""SELECT
      SUM(CAST(lower(CAST(description AS STRING)) RLIKE '{p}' AS INT)) d,
      SUM(CAST(lower(to_json(struct(procedure))) RLIKE '{p}' AS INT)) pr,
      SUM(CAST(lower(to_json(struct(equipment))) RLIKE '{p}' AS INT)) eq,
      SUM(CAST(lower(to_json(struct(capability))) RLIKE '{p}' AS INT)) ca,
      SUM(CAST(lower(to_json(struct(specialties))) RLIKE '{p}' AS INT)) sp FROM cp_fac""").first()
    rows.append((k, r.d, r.pr, r.eq, r.ca, r.sp))
display(spark.createDataFrame(rows, "service string, description long, procedure long, equipment long, capability long, specialties long"))

# COMMAND ----------

# MAGIC %md ## Suspicious merged records: coordinates far from their own PIN, and the first-row example

# COMMAND ----------

pin.createOrReplaceTempView("cp_pin")
display(spark.sql(r"""
WITH p AS (
  SELECT CAST(pincode AS STRING) pincode, percentile_approx(try_cast(latitude AS DOUBLE), 0.5) lat,
         percentile_approx(try_cast(longitude AS DOUBLE), 0.5) lon, COUNT(DISTINCT statename) n_states
  FROM cp_pin WHERE try_cast(latitude AS DOUBLE) BETWEEN 6 AND 37.6 AND try_cast(longitude AS DOUBLE) BETWEEN 68 AND 97.5
  GROUP BY 1)
SELECT f.name, f.address_city, f.address_stateOrRegion, f.address_zipOrPostcode, f.facilityTypeId,
       ROUND(2*6371*ASIN(SQRT(POW(SIN(RADIANS(p.lat - try_cast(f.latitude AS DOUBLE))/2),2) +
         COS(RADIANS(try_cast(f.latitude AS DOUBLE)))*COS(RADIANS(p.lat))*POW(SIN(RADIANS(p.lon - try_cast(f.longitude AS DOUBLE))/2),2))),1) AS km_from_own_pin
FROM cp_fac f JOIN p ON regexp_replace(CAST(f.address_zipOrPostcode AS STRING), '[ -]', '') = p.pincode
ORDER BY km_from_own_pin DESC LIMIT 20"""))

# COMMAND ----------

# MAGIC %md ## PIN directory (post-office grain) and NFHS

# COMMAND ----------

display(spark.sql(r"""
SELECT COUNT(*) offices, COUNT(DISTINCT pincode) pins,
  SUM(CASE WHEN CAST(pincode AS STRING) RLIKE '^[1-9][0-9]{5}$' THEN 0 ELSE 1 END) bad_pin_format,
  SUM(CASE WHEN try_cast(latitude AS DOUBLE) IS NULL THEN 1 ELSE 0 END) lat_null_or_text,
  SUM(CASE WHEN try_cast(latitude AS DOUBLE) BETWEEN 68 AND 97.5 AND try_cast(longitude AS DOUBLE) BETWEEN 6 AND 37.6 THEN 1 ELSE 0 END) looks_swapped,
  SUM(CASE WHEN try_cast(latitude AS DOUBLE) IS NOT NULL AND NOT (try_cast(latitude AS DOUBLE) BETWEEN 6 AND 37.6 AND try_cast(longitude AS DOUBLE) BETWEEN 68 AND 97.5) THEN 1 ELSE 0 END) outside_india
FROM cp_pin"""))
display(spark.sql("""
SELECT SUM(CASE WHEN nd > 1 THEN 1 ELSE 0 END) pins_multi_district, SUM(CASE WHEN ns > 1 THEN 1 ELSE 0 END) pins_multi_state
FROM (SELECT pincode, COUNT(DISTINCT district) nd, COUNT(DISTINCT statename) ns FROM cp_pin GROUP BY pincode)"""))

# COMMAND ----------

nfhs_stack = None
val_cols = [c for c in nfhs.columns if c not in ("district_name", "state_ut")]
nf = nfhs.select("district_name", "state_ut", F.explode(F.array(*[F.struct(F.lit(c).alias("k"), F.trim(F.col(f"`{c}`").cast("string")).alias("v")) for c in val_cols])).alias("kv"))
display(nf.agg(F.count("*").alias("cells"), F.sum((F.col("kv.v") == "*").cast("int")).alias("asterisk_suppressed"),
               F.sum(F.col("kv.v").rlike(r"^\(.*\)$").cast("int")).alias("parenthesized_small_sample"),
               F.sum(F.col("kv.v").isNull().cast("int")).alias("null")))
display(nfhs.groupBy("district_name").agg(F.collect_set("state_ut").alias("states")).where(F.size("states") > 1))

# COMMAND ----------

# MAGIC %md ## SUMMARY (paste this back)

# COMMAND ----------

print(json.dumps({
    "rows": {"facilities": fac.count(), "pincode": pin.count(), "nfhs": nfhs.count()},
    "local_rows": LOCAL_ROWS,
    "schema_diffs_vs_local": schema_diffs,
    "facility_types": {f.name: f.dataType.simpleString() for f in fac.schema.fields},
    "pincode_types": {f.name: f.dataType.simpleString() for f in pin.schema.fields},
    "nfhs_nonstring_cols": [f.name for f in nfhs.schema.fields if f.dataType.simpleString() != "string"][:20],
}, indent=1))
