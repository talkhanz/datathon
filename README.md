# CareProof India: Referral Navigator

**An evidence-aware healthcare referral application for India.**
Enter a service (dialysis, emergency C-section, NICU, MRI, cardiac surgery, burn care, trauma) and a location
(PIN code, district, city, or coordinates). CareProof returns the **most credibly documented** nearby facilities.
For each one it shows the **exact quote from the facility's own record**, an evidence category, identity and
location checks, transparent component scores, alternatives, and a call-ahead checklist. When nothing qualifies,
it **abstains**.

> **The nearest claimed provider is not necessarily the nearest credibly supported provider.**

CMU Datathon Fall 2026 · runs on **Databricks Free Edition (serverless)** · Python, PySpark, Spark SQL ·
Streamlit Databricks App.

---

## How it decides (WHY this facility, HOW it was ranked)

Every field in the source data is an LLM-extracted **claim**, not a fact. CareProof:

1. **Retrieves** records that mention the service, and copies the supporting passage **verbatim**.
2. **Checks the claim.** Negated, planned, historical, referral or tie-up language ("referred to…", "under
   construction", "through tie-up with…") counts as a contradiction. A passage about a **different named
   facility** is not evidence for this one.
3. **Checks the record**: identity (does the description describe a different hospital? duplicates? provenance?) and
   location (coordinates vs its own PIN, state vs PIN, placeholder coordinates, ambiguous PINs).
4. **Categorises the evidence**: strong / moderate / weak / conflicting / insufficient
   ([evidence policy](docs/evidence_policy.md)).
5. Applies **hard eligibility gates** *before* ranking. Conflicting or insufficient evidence, low location or identity
   confidence, probable merged records and missing excerpts are never recommended; they go to
   `careproof_facilities_quarantine` with reason codes ([quarantine rules](docs/quarantine_rules.md)).
6. **Ranks eligible candidates only** using separate components (service relevance, evidence strength, identity
   confidence, location confidence, readiness evidence, distance, data-quality penalty). The weights are policy
   choices in [`config.py`](src/careproof/config.py), not learned. Distance never rescues an ineligible record.
7. **Shows alternatives**, and gives a service-specific **verification checklist** and a printable **referral
   summary**. Current availability is **never** inferred: it reads "not verified" until the user confirms it by phone.

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
| 03b | `notebooks/03b_exploratory_analysis` | question-driven EDA and data-quality analysis on the Marketplace data (read-only) | 1–2 min |
| 04 | `notebooks/04_recommendation_demo` | **the interface**: widgets → result cards + tables + baseline comparison + NFHS context + guided scenarios | < 1 min |
| app | **CareProof Referral Navigator** (Databricks App, `app/app.py`) | Find · Compare · Verify · Refer, the main user interface | see below |
| 05 | `notebooks/05_evaluation` | curated benchmark + seeded random sample; baselines B1–B4; latency p50/p95; blind labeling template | 2–4 min |

Then, optionally, run [`sql/validation_queries.sql`](sql/validation_queries.sql) in the SQL editor.

### Deploy the Referral Navigator app (after notebooks 02 and 03)
1. **Compute → Apps → Create app → Custom**. Name it `careproof-navigator`.
2. Add a resource: **SQL warehouse** (the Free Edition *Serverless Starter Warehouse*), permission *Can use*,
   resource key **`sql-warehouse`** (this must match `app.yaml`).
3. Create the app. Then **Deploy** with source code path = this Git folder's root (e.g.
   `/Workspace/Users/<you>/datathon`). `app.yaml` and `requirements.txt` are at the root.
4. Grant the app's service principal read access (shown on the app page as *Service principal*). In a SQL editor:
   ```sql
   GRANT USE CATALOG ON CATALOG workspace TO `<app-service-principal-id>`;
   GRANT USE SCHEMA ON SCHEMA workspace.careproof_india TO `<app-service-principal-id>`;
   GRANT SELECT ON SCHEMA workspace.careproof_india TO `<app-service-principal-id>`;
   ```
5. Open the app URL. Run locally against Parquet with
   `CAREPROOF_LOCAL_DATA=data/derived streamlit run app/app.py`.

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
python -m pytest -q tests                                  # 139 tests (incl. headless Streamlit app tests)
# put the CSV exports in data/raw/ (see data/README.md), then:
python scripts/run_local.py build                          # ~1.5 min, writes Parquet to data/derived/
python scripts/make_eda_report.py                          # regenerates docs/data_quality_report.md + figures
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

**Measured on Databricks** (notebook 05, serverless, 2026-11-21; local build gives identical eligibility counts). Random sample of 56 queries
(seeded, stratified by service × district rurality, split dev/test by state):

| Metric | Value |
|---|---|
| **Unsupported recommendation rate** (shown results failing the evidence gates) | nearest-synonym baseline **31.9%** · exact-word baseline 29.7% · CareProof 0% (by construction) |
| Coverage (queries with ≥1 eligible result) | 46 / 56; of 10 abstentions, 6 had mentions that were all held back |
| Exact-substring citations · valid locations | 169 / 169 · 169 / 169 |
| Results by evidence category | 43 strong · 84 moderate · 42 weak |
| Top-1 unchanged under 4 alternative weight policies | 45–52 / 56 |
| Latency per query (warm, Databricks serverless driver) | p50 7.4 ms · p95 15.0 ms |
| **Precision@K, quarantine precision/recall, retrieval recall** | **PENDING HUMAN REVIEW** (142-pair blind review sheet + query labeling template) |

CareProof's 0% unsupported rate is by construction. The baselines' rates are the informative numbers.

Further documentation: [evidence policy](docs/evidence_policy.md) · [quarantine rules](docs/quarantine_rules.md) ·
[confidence semantics](docs/confidence_semantics.md) · [referral workflow](docs/referral_workflow.md) ·
[evaluation protocol](docs/evaluation_protocol.md) · [safety & limitations](docs/safety_and_limitations.md) ·
[EDA / data-quality report](docs/data_quality_report.md) · [problem spec](docs/problem_spec.md) ·
[literature review](docs/literature_review.md) · [decision log](docs/decision_log.md) · [system card](docs/system_card.md)

**Heuristic decisions (policy, not learned):** all weights, penalties, confidence thresholds, the 15 km distance
half-score, the 15 km PIN outlier
radius, the 60% PIN inlier share, the 25 km location-conflict radius, and the field-strength values.

**Not trained, deliberately:** no supervised model. No independent labels exist, and labels derived from the same
text would be target leakage ([problem spec](docs/problem_spec.md)).

**Proposed / not implemented:** a Databricks App front end; embedding- or LLM-based claim verification; a
travel-time surface (Weiss et al. 2020); a district "evidence desert" planning view; a learned re-ranker after
human labels (state-grouped splits, calibration check).

**What CareProof cannot guarantee:** that a service is currently available, staffed, affordable, or open;
that web-extracted text is accurate or current; that coordinates are correct; or that distances reflect travel
time. NFHS indicators are district-level population statistics. They are shown only as community context and are
never used to score facilities or to describe individual patients. **Always call the facility before travelling.
In an emergency, call 108 or 112.**

See [`docs/architecture.md`](docs/architecture.md), [`docs/demo_script.md`](docs/demo_script.md) and
[`docs/judging_checklist.md`](docs/judging_checklist.md).
