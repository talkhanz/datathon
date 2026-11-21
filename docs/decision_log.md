# Decision log

| # | Date | Decision | Evidence / reason | Alternatives rejected |
|---|---|---|---|---|
| D1 | 11-21 | Quarantine rows with non-UUID `unique_id` | 88 local rows are fragments of multi-line text (CSV quoting broke) | Repairing rows heuristically (an earlier hackathon used an LLM): risk of fabricating fields |
| D2 | 11-21 | Collapse the PIN directory to PIN grain before joining | Post-office grain would multiply facility rows; checked join raises `FanoutError` | Joining on PIN + office: fanout |
| D3 | 11-21 | PIN centroid = median after dropping offices >15 km from the median; defensible only if ≥60% are inliers | Observed: most offices within ~3 km, strays at 20–40 km; first rule (max spread ≤20 km) wrongly rejected Delhi 110001 | Mean centroid (outlier-sensitive); max-spread rule (too strict) |
| D4 | 11-21 | Specialty tags never qualify a facility | C-section: 452 explicit vs 4,162 tag-only | Counting tags as weak positives |
| D5 | 11-21 | ConText-style negation plus a "not yet operational" class | "300-bed trauma hospital under construction" was counted as evidence | Ignoring negation (as the baselines do) |
| D6 | 11-21 | Burn pattern: broad match plus observed exclusions (post-burn, sunburn, scars) | A strict "burn unit/ward" pattern lost real claims ("burns units", "isolated burn beds"); a broad pattern admitted cosmetic clinics | Either extreme |
| D7 | 11-21 | Baseline uses word-boundary matching | The substring baseline matched "mri" inside "Amritaa", a strawman | Substring baseline |
| D8 | 11-21 | Independent-field disagreement detects merged records | e.g. one record: Hyderabad city, Delhi PIN, Puducherry coordinates | Re-running entity resolution without labels |
| D9 | 11-21 (pm) | **Distance score: fixed decay `1/(1+d/15km)`, not `1−d/R`** | Measured: the radius-relative score reordered the same facilities in 6/6 queries when R changed. A test now enforces radius invariance | Linear-in-radius (non-invariant); exponential (too harsh for rural distances) |
| D10 | 11-21 (pm) | **Corroboration = distinct statements, not fields** | Source fields are LLM extractions of the same pages; ~10% of multi-field claims repeat one sentence | Field count (overstates); assuming fields are independent |
| D11 | 11-21 (pm) | Collapse same-name records within 1 km in results; disclose | Duplicate recommendations waste a scarce top-K slot | Deduplicating silver (risk of merging distinct branches) |
| D12 | 11-21 (pm) | Explicit abstention states | Precision-first error costs (problem spec) | Always returning K results |
| D13 | 11-21 (pm) | **No supervised model** | No independent labels; rule-derived labels would leak | Snorkel-style training on our own rules |
| D14 | 11-21 (pm) | Seeded stratified random query sample, state-grouped dev/test split, frozen before inspection | The curated benchmark was chosen after seeing outputs (selection bias) | Reporting curated-benchmark metrics as performance |
| D15 | 11-21 (pm) | Parse each facility once and reuse it across services; regex pre-check | The gold build regressed to 746 s; now 45 s with **identical output** (NaN-aware diff over all 17,582 rows; permanent equivalence test) | Spark UDFs (harder to test locally, no speed need at 10k rows) |
| D16 | 11-21 (pm) | Parquet instead of pickle for local artifacts | Pickle is unsafe to load if provenance is unknown | — |
| D17 | 11-21 (pm) | Never use beds or doctor counts as evidence | Missing for 64–75% of rows, and size ≠ service | Capacity as a trust signal (used by a prior hackathon entry) |
| D18 | 11-21 (eve) | **Hard eligibility gates before ranking** replace penalty-only handling | Penalties let contaminated claims rank where options are sparse (merged "Aravind" record shown as #2) | Subtractive penalties only |
| D19 | 11-21 (eve) | **Claim attribution**: a passage naming another facility (and not this one) is not evidence | Observed: "Offers dialysis services at SPMM Hospital" in *Srp Hospital*; "Mars Hospital provides … dialysis" in *Abs Hospital* | Counting any passage in the record |
| D20 | 11-21 (eve) | Referral, tie-up, formerly and discontinued language counts as a contradiction | "C-section deliveries are referred to Kalpana Chawla GMC" (Kaithal) | Treating it as support |
| D21 | 11-21 (eve) | Description about a different named facility → `MULTIPLE_FACILITY_IDENTITIES` (high) | 240 records, e.g. *Saifee Hospital* → "Prince Aly Khan Hospital is located in Mumbai" | Re-running entity resolution |
| D22 | 11-21 (eve) | Default minimum evidence = weak, with categories dominating the rank score | `moderate` would cut C-section eligibility 353 → 104; weak still means explicit, attributable, uncontradicted | Strict moderate minimum (configurable) |
| D23 | 11-21 (eve) | Streamlit Databricks App reads the gold table via a SQL warehouse; same `careproof` package | One tested code path for notebooks and app | Re-implementing logic in the app |
