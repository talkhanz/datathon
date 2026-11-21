# Databricks notebook source
# MAGIC %md
# MAGIC # CareProof India – shared setup (run via `%run ./_common`)
# MAGIC Adds `src/` to the Python path, defines widgets, and provides safe-write helpers that can only
# MAGIC write `careproof_`-prefixed tables inside the configured target catalog/schema.

# COMMAND ----------

import importlib
import json
import os
import sys
import time

def _find_src():
    cands = [os.path.abspath(os.path.join(os.getcwd(), "..", "src"))]
    try:
        nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()  # noqa: F821
        cands.append("/Workspace" + os.path.dirname(os.path.dirname(nb)) + "/src")
    except Exception:
        pass
    for c in cands:
        if os.path.isdir(os.path.join(c, "careproof")):
            return c
    raise RuntimeError(f"Could not find src/careproof. Tried: {cands}. Run notebooks from the Git folder.")

SRC = _find_src()
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import careproof  # noqa: E402
from careproof import config, normalization, geography, service_matching, scoring, explanations, pipeline  # noqa: E402
for _m in (config, normalization, geography, service_matching, scoring, explanations, pipeline):
    importlib.reload(_m)
from careproof.config import SOURCE_TABLES, TABLES, fq_table, validate_identifier, PROJECT_PREFIX  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pyspark.sql import functions as F, types as T  # noqa: E402

print(f"careproof {careproof.__version__} loaded from {SRC}")

# COMMAND ----------

def create_widgets():
    w = dbutils.widgets  # noqa: F821
    w.text("target_catalog", config.DEFAULT_TARGET_CATALOG, "01 target_catalog")
    w.text("target_schema", config.DEFAULT_TARGET_SCHEMA, "02 target_schema")
    w.dropdown("requested_service", "dialysis",
               ["dialysis", "c-section", "nicu", "mri", "cardiac surgery", "burn care", "trauma"], "03 requested_service")
    w.dropdown("location_type", "pincode", ["pincode", "coordinates", "district", "city"], "04 location_type")
    w.text("location_value", "800001", "05 location_value (PIN / district / city)")
    w.text("latitude", "", "06 latitude (if coordinates)")
    w.text("longitude", "", "07 longitude (if coordinates)")
    w.text("max_distance_km", str(int(config.DEFAULT_MAX_DISTANCE_KM)), "08 max_distance_km")
    w.text("top_k", str(config.DEFAULT_TOP_K), "09 top_k")
    w.dropdown("rebuild_tables", "no", ["no", "yes"], "10 rebuild_tables")


def get_config() -> config.RunConfig:
    g = dbutils.widgets.get  # noqa: F821
    return config.RunConfig(
        target_catalog=g("target_catalog").strip(), target_schema=g("target_schema").strip(),
        requested_service=g("requested_service"), location_type=g("location_type"),
        location_value=g("location_value").strip(), latitude=g("latitude").strip(), longitude=g("longitude").strip(),
        max_distance_km=float(g("max_distance_km")), top_k=int(g("top_k")),
        rebuild_tables=g("rebuild_tables") == "yes",
    ).validate()


def tbl(cfg, key: str) -> str:
    return fq_table(cfg.target_catalog, cfg.target_schema, TABLES[key])


def table_exists(name: str) -> bool:
    return spark.catalog.tableExists(name.replace("`", ""))  # noqa: F821

# COMMAND ----------

def spark_to_pandas_str(sdf):
    """Convert to pandas with every column as string/None. Complex types become JSON strings,
    so the same pure-Python logic tested locally on the CSV runs here unchanged."""
    cols = []
    for f in sdf.schema.fields:
        c = F.col(f"`{f.name}`")
        if isinstance(f.dataType, (T.ArrayType, T.MapType, T.StructType)):
            cols.append(F.to_json(c).alias(f.name))
        else:
            cols.append(c.cast("string").alias(f.name))
    pdf = sdf.select(*cols).toPandas()
    return pdf.astype(object).where(pdf.notna(), None)


def pandas_to_spark(pdf: pd.DataFrame):
    """Explicit-schema conversion; missing values become SQL NULL (never NaN). Logic tested locally in
    tests/test_pipeline.py via pipeline.spark_column_values."""
    types = {"boolean": T.BooleanType(), "long": T.LongType(), "double": T.DoubleType(), "string": T.StringType()}
    fields, cols = [], []
    for c in pdf.columns:
        kind, vals = pipeline.spark_column_values(pdf[c])
        fields.append(T.StructField(c, types[kind], True))
        cols.append(vals)
    rows = list(zip(*cols)) if len(pdf) else []
    return spark.createDataFrame(rows, T.StructType(fields))  # noqa: F821


def safe_write(sdf, cfg, key: str, comment: str = ""):
    """Overwrite a project-owned Delta table. Refuses anything outside the target schema / prefix."""
    name = tbl(cfg, key)
    assert TABLES[key].startswith(PROJECT_PREFIX)
    n = sdf.count()
    print(f"WRITE -> {name}  ({n} rows, mode=overwrite)")
    (sdf.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(name.replace("`", "")))
    if comment:
        spark.sql(f"COMMENT ON TABLE {name} IS '{comment.replace(chr(39), '')}'")  # noqa: F821
    return name
