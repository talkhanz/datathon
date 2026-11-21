# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Environment check
# MAGIC Verifies: environment info, read access to the three Marketplace tables, the target catalog/schema,
# MAGIC and that the target schema is writable (using one uniquely named temporary `careproof_tmp_*` table,
# MAGIC which is the only object this notebook ever drops). Ends with a PASS/FAIL report.

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

create_widgets()
cfg = get_config()
results = []  # (check, status, detail)

def record(check, ok, detail=""):
    results.append((check, "PASS" if ok else "FAIL", str(detail)[:300]))
    print(f"[{'PASS' if ok else 'FAIL'}] {check}: {detail}")

# COMMAND ----------

# MAGIC %md ## 1. Environment information

# COMMAND ----------

env = spark.sql("SELECT current_user() AS user, current_catalog() AS catalog, current_schema() AS schema").first()
try:
    dbr = spark.sql("SELECT current_version().dbr_version AS v").first().v
except Exception:
    dbr = os.environ.get("DATABRICKS_RUNTIME_VERSION", "unknown")
print("Current user    :", env.user)
print("Current catalog :", env.catalog)
print("Current schema  :", env.schema)
print("Spark version   :", spark.version)
print("DBR version     :", dbr)
print("Python          :", sys.version.split()[0])
print("Target          :", f"`{cfg.target_catalog}`.`{cfg.target_schema}`")
record("environment info", True, f"user={env.user} spark={spark.version}")

# COMMAND ----------

# MAGIC %md ## 2. Marketplace tables are readable

# COMMAND ----------

source_counts = {}
for key, name in SOURCE_TABLES.items():
    try:
        df = spark.table(name.replace("`", ""))
        n = df.count()
        source_counts[key] = n
        print(f"\n=== {name}: {n} rows, {len(df.columns)} columns")
        for f in df.schema.fields:
            print(f"   {f.name:<55} {f.dataType.simpleString()}")
        record(f"read {key}", n > 0, f"{n} rows, {len(df.columns)} cols")
    except Exception as e:
        record(f"read {key}", False, f"{type(e).__name__}: {e}")

# COMMAND ----------

# MAGIC %md ## 3. Target catalog and schema

# COMMAND ----------

try:
    cats = [r[0] for r in spark.sql("SHOW CATALOGS").collect()]
    record("target catalog exists", cfg.target_catalog in cats, f"{cfg.target_catalog} in {cats[:10]}")
except Exception as e:
    record("target catalog exists", False, e)

schema_fq = f"`{cfg.target_catalog}`.`{cfg.target_schema}`"
exists = spark.catalog.databaseExists(f"{cfg.target_catalog}.{cfg.target_schema}")
if not exists and cfg.target_schema.startswith(PROJECT_PREFIX):
    print(f"Schema {schema_fq} not found; creating it (CREATE SCHEMA IF NOT EXISTS - non-destructive).")
    try:
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_fq} COMMENT 'CareProof India project-owned derived tables'")
        exists = True
    except Exception as e:
        print("Could not create schema:", e)
elif not exists:
    print(f"Schema {schema_fq} does not exist and is not careproof_-prefixed; not creating it. "
          "Create it yourself or set target_schema to a careproof_ name.")
record("target schema exists", exists, schema_fq)

# COMMAND ----------

# MAGIC %md ## 4. Safe writability test (temporary, uniquely named, removed afterwards)

# COMMAND ----------

import uuid
tmp_name = f"careproof_tmp_writecheck_{uuid.uuid4().hex[:10]}"
tmp_fq = fq_table(cfg.target_catalog, cfg.target_schema, tmp_name)
created = False
try:
    print(f"WRITE -> {tmp_fq} (temporary)")
    spark.sql(f"CREATE TABLE {tmp_fq} AS SELECT 1 AS ok, current_timestamp() AS checked_at")
    created = True
    v = spark.table(tmp_fq.replace("`", "")).first().ok
    record("target schema writable", v == 1, tmp_fq)
except Exception as e:
    record("target schema writable", False, f"{type(e).__name__}: {e}")
finally:
    if created:
        assert tmp_name.startswith("careproof_tmp_writecheck_")
        spark.sql(f"DROP TABLE IF EXISTS {tmp_fq}")
        print(f"Removed temporary table {tmp_fq}")

# COMMAND ----------

# MAGIC %md ## 5. Project code import check

# COMMAND ----------

try:
    s = service_matching.resolve_service("haemodialysis")
    d = geography.haversine_km(0, 0, 0, 1)
    record("careproof package importable", s.key == "dialysis" and 111 < d < 112, f"src={SRC}")
except Exception as e:
    record("careproof package importable", False, e)

# COMMAND ----------

# MAGIC %md ## PASS / FAIL report

# COMMAND ----------

report = spark.createDataFrame(results, "check string, status string, detail string")
display(report)
n_fail = sum(1 for r in results if r[1] == "FAIL")
print("=" * 60)
print(f"OVERALL: {'PASS' if n_fail == 0 else 'FAIL'}  ({len(results) - n_fail} passed, {n_fail} failed)")
print("Source row counts:", source_counts)
print("=" * 60)
