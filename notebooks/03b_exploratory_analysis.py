# Databricks notebook source
# MAGIC %md
# MAGIC # 03b · Exploratory and data-quality analysis (on the authoritative Marketplace data)
# MAGIC Run after 02 and 03. Every section answers **one question** that drove a design decision. The code is
# MAGIC `careproof.eda`, the same functions that produced `docs/data_quality_report.md` from the local CSVs, so any
# MAGIC difference between the two is a real data difference (Databricks is authoritative). Read-only: nothing is written.

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

create_widgets()
cfg = get_config()
from careproof import eda
import matplotlib.pyplot as plt

def load(key, numeric=(), boolean=()):
    d = spark_to_pandas_str(spark.table(tbl(cfg, key).replace("`", "")))
    for c in numeric:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in boolean:
        d[c] = d[c] == "true"
    return d

fac_raw = spark_to_pandas_str(spark.table(SOURCE_TABLES["facilities"].replace("`", "")))
pin_raw = spark_to_pandas_str(spark.table(SOURCE_TABLES["pincode"].replace("`", "")))
silver = load("facilities_silver", ["facility_lat", "facility_lon", "facility_to_pin_km", "location_lat", "location_lon",
                                    "n_exact_source_copies"], ["possible_duplicate"])
quarantine = (load("facilities_rejected_rows") if table_exists(tbl(cfg, "facilities_rejected_rows")) else pd.DataFrame())
pin_silver = load("pincode_silver", ["n_districts", "n_states"], ["coords_defensible", "is_ambiguous", "missing_coordinates"])
nfhs_silver = load("nfhs_silver", ["value"], ["is_suppressed", "is_small_sample"])
gold = load("facility_service_gold", [], ["is_candidate"])
R = eda.run_all(fac_raw, pin_raw, silver, quarantine, pin_silver, nfhs_silver, gold)
print({k: len(v) for k, v in R.items()})

# COMMAND ----------

# MAGIC %md ## Q4a · Do facilities that *claim* a service actually *state* it?
# MAGIC **Implication:** specialty tags vastly overstate services (locally only ~10% of C-section signals were explicit
# MAGIC text). CareProof never recommends on a tag alone.

# COMMAND ----------

t = R["Q4a"].set_index("service")
cols = [c for c in ["explicit_text", "specialty_tag_only", "negated_only", "keyword_only (baseline hit, no evidence)"] if c in t]
ax = t[cols].plot(kind="barh", stacked=True, figsize=(9, 4.5), color=["#1b7f3b", "#d4a72c", "#b42318", "#8c959f"][:len(cols)])
ax.set_xlabel("facilities with any signal"); ax.set_title("Claims vs explicit evidence"); plt.tight_layout(); display(plt.gcf()); plt.close()
display(R["Q4a"])

# COMMAND ----------

# MAGIC %md ## Q3b · Does a facility's position agree with its own PIN code?
# MAGIC **Implication:** records far from their own PIN are mislocated or merged. They get the `location_conflict` flag, a
# MAGIC lower location-quality score and a visible warning.

# COMMAND ----------

t = R["Q3b"]
ax = t.plot(kind="bar", x="facility_to_own_PIN_centroid", y="pct", legend=False, figsize=(7, 3.5), color="#0969da")
ax.set_ylabel("% of checkable facilities"); plt.xticks(rotation=0); plt.tight_layout(); display(plt.gcf()); plt.close()
display(R["Q3b"])

# COMMAND ----------

# MAGIC %md ## Q4b · How often do recommendable claims carry a contradiction?

# COMMAND ----------

display(R["Q4b"])

# COMMAND ----------

# MAGIC %md ## Q6a · Are rural districts covered? (proxy: share of branch post offices in the district)
# MAGIC **Implication:** if rural quartiles have few or no facilities, answers there will be distant or absent. CareProof
# MAGIC abstains honestly instead of stretching weak evidence, and urban benchmark results must not be generalised to rural use.
# MAGIC *The branch-office share is a proxy, not a census urban/rural classification.*

# COMMAND ----------

t = R["Q6a"]
ax = t.plot(kind="bar", x="rurality_quartile", y="total_facilities", legend=False, figsize=(7, 3.5), color="#8250df")
ax.set_ylabel("facilities"); plt.xticks(rotation=0); plt.tight_layout(); display(plt.gcf()); plt.close()
display(R["Q6a"])

# COMMAND ----------

# MAGIC %md ## Remaining questions: structure, missingness, coordinates, PINs, descriptions, sources, NFHS, state coverage

# COMMAND ----------

for code, title, _ in eda.ALL_QUESTIONS:
    if code in ("Q4a", "Q3b", "Q4b", "Q6a"):
        continue
    print(f"\n### {code}. {title}")
    if len(R[code]):
        display(pandas_to_spark(R[code].astype(str)))
    else:
        print("No rows (check passed).")

# COMMAND ----------

# MAGIC %md ## SUMMARY (paste back)

# COMMAND ----------

def first(df, col, row_label, label_col):
    m = df[df[label_col] == row_label]
    return None if m.empty else m.iloc[0][col]

summary = {
    "silver_facilities": len(silver), "corrupted_rows_rejected": len(quarantine),
    "pct_far_from_own_pin_gt25km": round(float(R["Q3b"][R["Q3b"].facility_to_own_PIN_centroid.isin(["25-100 km", "100-500 km", ">500 km"])].pct.sum()), 1),
    "pct_candidates_any_contradiction": first(R["Q4b"], "pct_of_candidates", "ANY of location/state/specialty/negation", "flag"),
    "explicit_share_by_service": dict(zip(R["Q4a"].service, R["Q4a"].explicit_share_of_any_signal_pct)),
    "rural_quartiles": R["Q6a"][["rurality_quartile", "total_facilities", "districts_with_zero_facilities"]].astype(str).to_dict("records"),
    "nfhs_join": R["Q5c"].astype(str).to_dict("records"),
}
print(json.dumps(summary, indent=1, default=str))
