# CareProof India

**An evidence-aware healthcare facility recommender for India.**
Enter a service (dialysis, emergency C-section, NICU, MRI, cardiac surgery, burn care, trauma) and a location
(PIN code, district, city, or coordinates). CareProof returns ranked facilities. Each result comes with an **exact quote
from the facility's own listing**, a transparent score breakdown, a confidence label, and warnings about missing or
contradictory data.

> **The nearest claimed provider is not necessarily the nearest credibly supported provider.**

CMU Datathon Fall 2026 · runs on **Databricks Free Edition (serverless)** · Python, PySpark and Spark SQL only.

---

## What it does

1. **Retrieve.** Find facilities whose own free text (procedure / equipment / capability / description / name)
   explicitly states the service. Specialty tags such as `nephrology` or `gynecologyAndObstetrics` count only as
   supporting context. Negated or planned mentions ("MRI not available", "trauma centre under construction")
   count as contradictions.
2. **Rank transparently.**
   `final = Σ weight × component − data_quality_penalty`, clamped to [0, 1]. The components are
   `service_match_score`, `evidence_confidence_score`, `location_quality_score`, `distance_score` and
   `readiness_score`. The weights are **policy choices** listed in [`config.py`](src/careproof/config.py). They are
   not learned. Notebook 05 tests how sensitive the ranking is to them.
3. **Explain.** Each result includes the facility ID, name, address, approximate straight-line distance, all
   scores, matched terms and fields, the exact excerpt, the source URL, a confidence label (High / Medium / Low),
   missing-data warnings, contradiction warnings, a plain-language explanation, and a **verify-before-travel**
   warning.

Checks that catch merged or unreliable records:

* Facility coordinates vs. the centroid of the facility's own PIN code (> 25 km away = `location_conflict`).
* Stated state vs. the PIN code's state (`state_conflict`).
* Facility name or type vs. the claimed service, e.g. an eye hospital claiming a dialysis unit (`specialty_mismatch`).
* Negated or planned mentions (`negated_mention`), login or payment pages among the source URLs
  (`suspicious_source_url`), and near-duplicate records (`possible_duplicate`).

---

## Quick start on Databricks Free Edition

### 1. Get the data
In Databricks, open **Marketplace** and search for *Virtue Foundation* (the DAIS 2026 dataset). Click
**Get instant access**. This creates the read-only catalog `databricks_virtue_foundation_dataset_dais_2026`.

### 2. Add this repository as a Git folder
1. Push this repository to GitHub. (For a private repo, first link GitHub: avatar → **Settings → Linked
   accounts → Git integration → GitHub**, then authorise or paste a personal access token.)
2. In the left sidebar, open **Workspace**, then your user folder, then **Create → Git folder**.
3. Paste the GitHub HTTPS URL. Leave the provider as *GitHub* and click **Create Git folder**.
4. To update the code later: open the folder, click the branch name, then **Pull**.

### 3. Run the notebooks in order (attach each one to **Serverless** compute)

| # | Notebook | What it does | Approx. time |
|---|---|---|---|
| 00 | `notebooks/00_environment_check` | user, catalog, Spark version; reads the 3 Marketplace tables; creates `workspace.careproof_india` if missing; writability test; **PASS/FAIL** report | < 1 min |
| 01 | `notebooks/01_data_audit` | read-only profiling and schema diff vs. the local CSV | 1–2 min |
| 02 | `notebooks/02_build_silver` | writes `careproof_pincode_silver`, `careproof_nfhs_silver`, `careproof_facilities_silver`, `careproof_facilities_quarantine`, `careproof_join_report` | 2–4 min |
| 03 | `notebooks/03_build_gold` | writes `careproof_facility_service_gold` | 1–3 min |
| 04 | `notebooks/04_recommendation_demo` | **the interface**: widgets → result cards + tables + baseline comparison + NFHS context + guided scenarios | < 1 min |
| 05 | `notebooks/05_evaluation` | benchmark, automatic metrics, labeling template | 1–2 min |

Then, optionally, run [`sql/validation_queries.sql`](sql/validation_queries.sql) in the SQL editor.

### Configuration (widgets, shown at the top of every notebook)

| Widget | Default | Meaning |
|---|---|---|
| `target_catalog` | `workspace` | writable catalog for derived tables |
| `target_schema` | `careproof_india` | writable schema (created by 00 only if its name starts with `careproof_`) |
| `requested_service` | `dialysis` | service to search |
| `location_type` | `pincode` | `pincode`, `coordinates`, `district` or `city` |
| `location_value` | `800001` | PIN, or a district such as `Patna, Bihar` |
| `latitude` / `longitude` | empty | decimal degrees, used when `location_type = coordinates` |
| `max_distance_km` | `100` | search radius (straight-line distance) |
| `top_k` | `5` | number of results |
| `rebuild_tables` | `no` | `yes` forces 02/03 to rebuild tables that already exist |

Every write prints its exact destination first (`WRITE -> ...`). Writes are full overwrites of project-owned tables,
so re-running is idempotent.

### Clean-run checklist
- [ ] 00 ends with `OVERALL: PASS`.
- [ ] 02 prints `fanout=0` for the PIN join and writes 5 tables.
- [ ] 03 prints the evidence-level table and `WRITE -> ...careproof_facility_service_gold`.
- [ ] 04 shows result cards for scenarios A–D, and every card has a quoted excerpt.
- [ ] 05 prints `citation_exact_substring_rate` and `geographic_validity_rate`.
- [ ] `sql/validation_queries.sql`: every "expect zero rows" query returns zero rows.

### Reset (touches project-owned objects only)
Run this in a SQL cell. It never drops the schema or catalog:
```sql
DROP TABLE IF EXISTS workspace.careproof_india.careproof_facility_service_gold;
DROP TABLE IF EXISTS workspace.careproof_india.careproof_facilities_silver;
DROP TABLE IF EXISTS workspace.careproof_india.careproof_facilities_quarantine;
DROP TABLE IF EXISTS workspace.careproof_india.careproof_pincode_silver;
DROP TABLE IF EXISTS workspace.careproof_india.careproof_nfhs_silver;
DROP TABLE IF EXISTS workspace.careproof_india.careproof_join_report;
DROP TABLE IF EXISTS workspace.careproof_india.careproof_eval_results;
DROP TABLE IF EXISTS workspace.careproof_india.careproof_eval_results_detail;
DROP TABLE IF EXISTS workspace.careproof_india.careproof_eval_labeling_template;
```
Alternatively, run 02 and 03 with `rebuild_tables = yes`, which overwrites the tables in place.

---

## Local development

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate   # or python3.12 -m venv .venv
pip install -r requirements.txt
python -m pytest -q tests                                  # 83 tests
# put the CSV exports in data/raw/ (see data/README.md), then:
python scripts/run_local.py build                          # ~1.5 min
python scripts/run_local.py query dialysis pincode 800001 25 5
python scripts/run_local.py query "burn care" district "Koraput" 200 5
python scripts/run_local.py query trauma coordinates 26.2183,78.1828 50 5
python scripts/run_local.py eval
```

---

## Status: what is implemented, measured, heuristic, or not guaranteed

**Implemented and tested locally** (unit tests plus a local end-to-end run on the CSV exports):
normalization (PINs kept as strings, state aliases, coordinate validation, NFHS `*` and `(x)` handling);
PIN-grain aggregation with outlier-robust centroids and ambiguity flags; location resolution for PIN, coordinates
and district/city, with actionable errors; deterministic service matching with exact excerpts, negation and
"planned" detection, and false-positive exclusions (gastric bypass, eye trauma, sunburn, post-burn
reconstruction); contradiction and data-quality flags; transparent scoring with deterministic tie-breaking;
keyword baseline; benchmark with automatic metrics; HTML result cards.

**Implemented, Databricks run pending:** notebooks 00–05 and the SQL validation queries. They were written against
the local schema; notebook 01 reports any schema differences on Databricks.

**Measured (local CSV run, 23 benchmark queries, default weights; to be re-measured on Databricks in notebook 05):**

| Metric | Value |
|---|---|
| Queries handled as expected (results, or the expected actionable error) | 22 / 23 |
| Citation correctness (excerpt is an exact substring of the cited field) | 77 / 77 |
| Geographic validity (within radius, inside India, distance recomputes) | 77 / 77 |
| Baseline (nearest keyword match) results that are weak or contradictory | 30 / 80 |
| Queries where the baseline's #1 is weak or contradictory | 9 / 18 |
| Top-1 unchanged under alternative weight scenarios | 14–15 / 18 |
| **Precision@K** | **PENDING HUMAN REVIEW** (labeling template generated) |

The one unmet expectation is Q12: there is no explicit burn-care text within 25 km of Mumbai GPO in this dataset.
CareProof reports "none found" rather than promoting skin or cosmetic clinics. This is a dataset coverage gap,
not a confirmed absence of burn care in Mumbai.

**Heuristic decisions (policy, not learned):** all weights, penalties, confidence thresholds, the 15 km PIN outlier
radius, the 60% PIN inlier share, the 25 km location-conflict radius, and the field-strength values.

**Proposed / not implemented:** a Databricks App front end, embedding- or LLM-based matching (deliberately
deferred until after the deterministic baseline), road-network travel time, a district "evidence desert" planning
view, and human-labelled precision@K.

**What CareProof cannot guarantee:** that a service is currently available, staffed, affordable, or open;
that web-extracted text is accurate or current; that coordinates are correct; or that distances reflect travel
time. NFHS indicators are district-level population statistics. They are shown only as community context and are
never used to score facilities or to describe individual patients. **Always call the facility before travelling.
In an emergency, call 108 or 112.**

See [`docs/architecture.md`](docs/architecture.md), [`docs/demo_script.md`](docs/demo_script.md) and
[`docs/judging_checklist.md`](docs/judging_checklist.md).
