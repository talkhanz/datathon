# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · Service-profile model: feature selection, supervised training, GA inverse design
# MAGIC **Question.** From a facility's *structured* attributes only (type, operator, size, staff, specialty tags, digital
# MAGIC footprint, local facility density), how likely is it that its own text documents service S?
# MAGIC
# MAGIC * **Label (silver):** `evidence_level == "explicit_text"` from the deterministic evidence engine. It is a label
# MAGIC   about the record, not proof of real-world availability. Human labels remain **PENDING_HUMAN_REVIEW**.
# MAGIC * **Leakage control:** the text fields the label is built from (name, description, procedure, equipment,
# MAGIC   capability) and anything derived from them are never features (`modeling.FORBIDDEN_SOURCES`, enforced and tested).
# MAGIC * **Protocol:** whole states are split dev/test. Filter selection → **genetic-algorithm wrapper** feature
# MAGIC   selection → model choice (logistic vs gradient-boosted trees vs specialty-tag rule) all use dev states with
# MAGIC   state-grouped CV. Metrics are reported on untouched test states.
# MAGIC * **GA inverse design:** a second GA searches for the profile that maximises the model score while staying close
# MAGIC   to real documented facilities (nearest-neighbour realism penalty). Associational, **not** causal.
# MAGIC * **Use:** "profile fit" cross-check and "unverified leads" in the app. The model never changes who is recommended.

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

create_widgets()
cfg = get_config()
from careproof import modeling as M

silver = spark_to_pandas_str(spark.table(tbl(cfg, "facilities_silver").replace("`", "")))
gold = pipeline.coerce_gold_types(spark_to_pandas_str(spark.table(tbl(cfg, "facility_service_gold").replace("`", ""))))
print(f"silver={len(silver)} gold={len(gold)}")

# COMMAND ----------

t0 = time.time()
out = M.train_all(silver, gold)
print(f"trained {len(out['model_profile_summary'])} services in {time.time()-t0:.0f}s")

# COMMAND ----------

# MAGIC %md ## Held-out test metrics (average precision; base rate = random guessing)

# COMMAND ----------

m = out["model_metrics"]
display(pandas_to_spark(m[m.split == "test"][["service", "model", "feature_set", "n_features", "average_precision",
                                            "roc_auc", "brier", "ece", "base_rate", "chosen"]].round(4)))
display(pandas_to_spark(out["model_profile_summary"]))

# COMMAND ----------

# MAGIC %md ## Persist (project-owned `careproof_model_*` tables only)

# COMMAND ----------

comments = {
    "model_scores": "Out-of-fold (state-grouped) service-profile model score per facility x service. Not a recommendation.",
    "model_feature_selection": "Filter + GA wrapper feature selection report with permutation importance.",
    "model_metrics": "Dev-CV and held-out-state metrics for logistic, boosted trees and the specialty-tag rule.",
    "model_ga_history": "Per-generation best/mean fitness of both genetic algorithms.",
    "model_optimal_profile": "GA-optimal provider profile vs documented / undocumented facilities. Associational.",
    "model_profile_summary": "One row per service: chosen model, test AP, optimal-profile score, nearest real facility.",
    "model_calibration": "Reliability bins on held-out states.",
    "model_features": "Structured feature matrix used by the model (no text-derived features).",
    "model_facilities": "Slim facility table (location confidence, high-severity flag) for unverified leads.",
}
for key in config.MODEL_TABLES:
    safe_write(pandas_to_spark(out[key]), cfg, key, comments[key])
