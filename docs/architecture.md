# Architecture

```
Marketplace (read-only, "bronze")                 Project-owned Delta tables in <target_catalog>.<target_schema>
──────────────────────────────────                ─────────────────────────────────────────────────────────────
india_post_pincode_directory ──02──► careproof_pincode_silver      (1 row / PIN: counts, districts, states,
  (post-office grain)                                              robust centroid, defensible?, ambiguous?)
                                          │
facilities ───────────────────────02──► careproof_facilities_silver (1 row / facility; ALL original columns kept
  (10k, web-extracted)                    ▲  + normalized PIN/state/coords, PIN cross-checks, flags, location_quality)
                                          │  careproof_facilities_quarantine (corrupted rows)
                                          │  careproof_join_report  (row counts, match rate, fanout)
                                          │
                                   03 ────┴──► careproof_facility_service_gold (1 row / facility × service:
                                                evidence level, exact excerpts, matched fields/terms, scores, flags)
nfhs_5_district_health_indicators ─02─► careproof_nfhs_silver (long: district × indicator, raw + parsed value,
                                        suppressed / small-sample flags)  ── context only, never joined to facilities
                                          │
                                   04 ────┴──► widgets → resolve location → filter gold by service → distance →
                                                score → rank → cards + tables + baseline + NFHS context
                                   05 ───────► careproof_eval_results(_detail), careproof_eval_labeling_template
```

## Code organisation
* `src/careproof/*.py` are pure Python with no Spark dependency. They are unit-tested locally.
  * `normalization.py` handles PINs (always strings), names, states, coordinates, JSON arrays, and NFHS values.
  * `geography.py` has Haversine distance, PIN aggregation, and location resolution with `LocationError`.
  * `service_matching.py` holds the taxonomy, exact-excerpt extraction, negation and planned detection,
    profile conflicts, and the baseline keyword hit.
  * `scoring.py` computes the components, penalties, final score, confidence label, and deterministic `rank_key`.
  * `explanations.py` produces the warnings text, the plain-language explanation, and the HTML cards.
  * `pipeline.py` builds tables (pandas), `checked_left_join` (raises `FanoutError`), `recommend`, and the baseline.
  * `evaluation.py` runs the benchmark and computes the automatic metrics.
* Notebooks only orchestrate. Spark tables go through `spark_to_pandas_str` (every column becomes a string or
  JSON string), then through the same `pipeline` functions, then `pandas_to_spark` (explicit schema), then
  `safe_write`. The data is small (about 10k facilities and 165k post offices), so pandas on the serverless driver
  is fast. It also guarantees that the code tested locally is the code that runs on Databricks.

## Key design decisions
| Decision | Why |
|---|---|
| The PIN directory is collapsed to PIN grain before any join | Post-office grain would multiply facility rows. `checked_left_join` fails loudly on fanout. |
| PIN centroid = median of in-India offices after dropping offices > 15 km from the median; defensible only if ≥ 60% are inliers | Observed: most offices lie within ~3 km, with a minority of mis-geocoded strays 20–40 km out. ~800 rows are lat/lon-swapped. |
| Specialty tags never qualify a facility on their own | 4,162 facilities carry an obstetrics tag but only 452 explicitly mention C-section or caesarean. |
| Negated or planned mentions count as contradictions | e.g. "300-bed trauma care hospital under construction". |
| Merged-record detection uses agreement between independent fields | e.g. the record "Aravind Eye Hospital": Hyderabad city, Delhi PIN, Puducherry coordinates, and a Delhi hospital's dialysis unit. |
| Distance is a straight-line Haversine distance, labelled as such | Road networks and travel times are not in the data. |
| NFHS is shown as context only | District statistics are not patient attributes and not facility quality. |

## Scoring
`final = clamp01(Σ w_i · c_i − penalty)`, where the weights are normalised to sum to 1.
Defaults: service 0.30, evidence 0.25, location 0.10, distance 0.25, readiness 0.10.
* `service_match_score` is the strength of the strongest field with a positive mention (procedure or equipment 1.0,
  capability or description 0.9, name 0.7).
* `evidence_confidence_score` is corroboration: 1 field 0.5, 2 fields 0.75, 3 or more 0.9; plus 0.1 for a related
  specialty tag and 0.05 for specific detail (counts, "unit", "ward", …).
* `location_quality_score` is 1.0 if the coordinates are within 5 km of the facility's own PIN centroid, 0.8 within
  25 km, 0.3 beyond 25 km, 0.6 if this can't be checked, and 0.4 if the position comes from the PIN centroid; minus
  0.2 for a state conflict.
* `distance_score` is `1 − d / max_distance_km`.
* `readiness_score` is 0.5 × (24x7 stated with the service = 1, elsewhere = 0.5) + 0.5 × (dedicated unit or
  quantity stated). Beds and doctor counts are deliberately **not** used.
* Penalties: negated mention 0.20, specialty mismatch 0.25, location conflict 0.15, state conflict 0.10, suspicious
  URL 0.05, possible duplicate 0.03, coordinates from PIN 0.05; capped at 0.60.
* Confidence is **High** if service ≥ 0.9, evidence ≥ 0.7, penalty ≤ 0.05 and location ≥ 0.6. It is **Low** if
  evidence < 0.5, penalty ≥ 0.25 or location < 0.4. Everything else is **Medium**.
* Tie-break: final score descending, then distance ascending, then `facility_id` ascending.
