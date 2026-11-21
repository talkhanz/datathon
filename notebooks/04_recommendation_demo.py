# Databricks notebook source
# MAGIC %md
# MAGIC # CareProof India · Find credibly supported care
# MAGIC **The nearest claimed provider is not necessarily the nearest credibly supported provider.**
# MAGIC
# MAGIC **How to use (no coding needed):**
# MAGIC 1. In the widget bar at the top, choose a **requested_service**.
# MAGIC 2. Choose **location_type** and enter a **location_value**: a 6-digit PIN code (e.g. `800001`), or a district
# MAGIC    (e.g. `Patna, Bihar`). For `coordinates`, fill **latitude** and **longitude** instead.
# MAGIC 3. Set **max_distance_km** and **top_k** (how many results).
# MAGIC 4. Click **Run all** (or re-run the cell under *Your search*). Results appear as cards, then as a table.
# MAGIC
# MAGIC Every result shows an **exact quote** from the facility's own listing, the score breakdown, a confidence level,
# MAGIC and warnings. **Nothing here confirms that a service is available today. Always call ahead.**

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

create_widgets()
cfg = get_config()
from careproof.explanations import render_cards_html
from careproof.geography import LocationError, resolve_location
from careproof.service_matching import resolve_service, TAXONOMY

_PIN_INDEX = None
_GOLD_CACHE = {}

def place_index():
    global _PIN_INDEX
    if _PIN_INDEX is None:
        _PIN_INDEX = pipeline.place_index_from_pin_silver(
            spark_to_pandas_str(spark.table(tbl(cfg, "pincode_silver").replace("`", ""))).assign(
                centroid_lat=lambda d: pd.to_numeric(d.centroid_lat), centroid_lon=lambda d: pd.to_numeric(d.centroid_lon),
                coords_defensible=lambda d: d.coords_defensible == "true", is_ambiguous=lambda d: d.is_ambiguous == "true",
                n_offices_with_valid_coords=lambda d: pd.to_numeric(d.n_offices_with_valid_coords),
                n_offices_coord_outliers=lambda d: pd.to_numeric(d.n_offices_coord_outliers),
                post_office_count=lambda d: pd.to_numeric(d.post_office_count)))
    return _PIN_INDEX

def gold_for(service_key):
    if service_key not in _GOLD_CACHE:
        g = spark_to_pandas_str(spark.table(tbl(cfg, "facility_service_gold").replace("`", ""))
                                .where(F.col("service") == service_key))
        for c in ("location_lat", "location_lon", "location_quality_score", "service_match_score",
                  "evidence_confidence_score", "readiness_score", "facility_to_pin_km"):
            g[c] = pd.to_numeric(g[c], errors="coerce")
        for c in ("is_candidate", "baseline_keyword_hit"):
            g[c] = g[c] == "true"
        _GOLD_CACHE[service_key] = g
    return _GOLD_CACHE[service_key]

def run_query(service_text, location_type, location_value="", lat="", lon="", max_km=50.0, top_k=5, title=""):
    try:
        sdef = resolve_service(service_text)
    except ValueError as e:
        displayHTML(f"<div style='background:#ffebe9;padding:10px;border-radius:6px'><b>Service not recognised.</b> {e}</div>")
        return None, None, None, None
    try:
        loc = resolve_location(location_type, location_value, lat, lon, place_index())
    except LocationError as e:
        displayHTML(f"<div style='background:#ffebe9;padding:10px;border-radius:6px'><b>Location could not be resolved "
                    f"reliably.</b> {e}</div>")
        return None, None, None, None
    gold = gold_for(sdef.key)
    results, summary = pipeline.recommend(gold, loc, sdef.key, float(max_km), int(top_k), config.DEFAULT_WEIGHTS)
    baseline = pipeline.keyword_baseline(gold, loc, sdef.key, float(max_km), int(top_k))
    displayHTML(render_cards_html(results, summary, title))
    return results, summary, baseline, loc

def results_table(results):
    cols = ["rank", "facility_name", "address", "distance_km", "confidence", "final_score", "service_match_score",
            "evidence_confidence_score", "location_quality_score", "distance_score", "readiness_score",
            "data_quality_penalty", "evidence_field", "evidence_excerpt", "source_url", "facility_id"]
    df = pd.DataFrame([{**{c: r[c] for c in cols}, "matched_terms": ", ".join(r["matched_terms"]),
                        "warnings": " | ".join(r["contradiction_warnings"] + r["missing_data_warnings"])} for r in results])
    return df

def show(pdf):
    if pdf is None or len(pdf) == 0:
        print("(no rows)")
    else:
        display(pandas_to_spark(pdf))

def baseline_compare(results, baseline):
    cp = {r["facility_id"]: r for r in results}
    rows = []
    for b in baseline:
        c = cp.get(b["facility_id"])
        rows.append({"baseline_rank": b["rank"], "facility_name": b["facility_name"], "distance_km": b["distance_km"],
                     "baseline_evidence_level": b["evidence_level"],
                     "careproof_rank": c["rank"] if c else None, "careproof_confidence": c["confidence"] if c else "not recommended",
                     "flags": ", ".join(b["flags"])})
    return pd.DataFrame(rows)

# COMMAND ----------

# MAGIC %md ## Your search

# COMMAND ----------

out = run_query(cfg.requested_service, cfg.location_type, cfg.location_value, cfg.latitude, cfg.longitude,
                cfg.max_distance_km, cfg.top_k, title="Your search")
if out and out[0] is not None:
    results, summary, baseline, loc = out
    if results:
        show(results_table(results))

# COMMAND ----------

# MAGIC %md ### Comparison: “nearest facility with a keyword match” (naive baseline)
# MAGIC The baseline sorts any facility whose text *or specialty tags* contain the service word by distance only,
# MAGIC with no negation handling and no contradiction checks.

# COMMAND ----------

if out and out[0] is not None and baseline:
    show(baseline_compare(results, baseline))

# COMMAND ----------

# MAGIC %md ### Community context for the searched district (NFHS-5)
# MAGIC District-level survey statistics describe the **population**, not any individual patient, and they say nothing
# MAGIC about facility quality. They are shown only as context for planners. Values in parentheses are small-sample
# MAGIC estimates; `*` means suppressed.

# COMMAND ----------

if out and out[0] is not None and loc.district:
    sk = resolve_service(cfg.requested_service).key
    inds = list(TAXONOMY[sk].nfhs_context)
    if inds:
        q = (spark.table(tbl(cfg, "nfhs_silver").replace("`", "")).where(F.col("district_norm") == loc.district)
             .where(F.col("indicator").isin(inds)))
        if loc.state:
            q = q.where(F.col("state_norm") == loc.state)
        display(q.select("district_name_raw", "state_ut_raw", "indicator", "raw_value", "value", "is_suppressed", "is_small_sample"))
    else:
        print("No NFHS indicator is mapped to this service.")
else:
    print("District could not be determined unambiguously for this location; NFHS context not shown.")

# COMMAND ----------

# MAGIC %md ---
# MAGIC # Guided demonstration (fixed scenarios)
# MAGIC ## A. Strong evidence: dialysis near Patna GPO (PIN 800001)

# COMMAND ----------

a = run_query("dialysis", "pincode", "800001", max_km=25, top_k=3, title="A · Strong evidence")

# COMMAND ----------

# MAGIC %md ## B. Weak / contradictory evidence: dialysis at the location of a merged record
# MAGIC The record named *Aravind Eye Hospital* (an eye hospital) lists a **20-bed dialysis unit**, gives a Delhi PIN,
# MAGIC a Hyderabad city, and coordinates in Puducherry. A keyword search puts it first. CareProof keeps the quote
# MAGIC visible but labels it **Low confidence** and explains every contradiction.

# COMMAND ----------

b = run_query("dialysis", "coordinates", "", "11.938235", "79.485443", max_km=5, top_k=5, title="B · Contradictory record")
if b and b[0] is not None:
    show(baseline_compare(b[0], b[2]))

# COMMAND ----------

# MAGIC %md ## C. CareProof vs nearest-keyword baseline: cardiac surgery near Ahmedabad (PIN 380001)
# MAGIC The baseline puts the closest keyword match first. CareProof checks whether the facility's position agrees with
# MAGIC its own PIN code and how many fields corroborate the claim, and shows why its ranking differs.

# COMMAND ----------

c = run_query("cardiac surgery", "pincode", "380001", max_km=25, top_k=5, title="C · Cardiac surgery, Ahmedabad")
if c and c[0] is not None:
    show(baseline_compare(c[0], c[2]))

# COMMAND ----------

# MAGIC %md ## D. Honest “not found”: burn care within 25 km of Mumbai GPO (PIN 400001)
# MAGIC Keyword matches nearby are skin, hair and cosmetic clinics. CareProof does not promote them; it says no credibly
# MAGIC supported provider was found in this radius and suggests widening the search.

# COMMAND ----------

d = run_query("burn care", "pincode", "400001", max_km=25, top_k=5, title="D · Burn care, Mumbai")
if d and d[0] is not None:
    show(baseline_compare(d[0], d[2]))
