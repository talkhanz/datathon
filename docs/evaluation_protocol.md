# Evaluation protocol

## Query and pair sets (all frozen with seed 20261121, before results were inspected)
| Set | File | Purpose |
|---|---|---|
| Curated behaviour benchmark | `evaluation/benchmark_queries.csv` (23) | Edge cases: ambiguous PIN, swapped coordinates, merged record, invalid input. **Not** an unbiased estimate |
| Random query sample | `evaluation/labeling_queries_v1.csv` (56) | 7 services × 4 district-rurality quartiles × 2 PINs; split dev/test **by state** |
| Facility-service pair benchmark | `evaluation/pair_benchmark_v1.csv` (142) | Service × stratum (strong/moderate/weak/conflicting, tag-only and keyword-only hard negatives, negation, merged/misattributed, geographic contradiction, missing coordinates, ambiguous PIN) |

Only IDs, PINs and strata are committed. Review sheets with excerpts are generated at runtime
(`careproof_eval_pair_review_sheet`, `careproof_eval_labeling_template`).

## Labels (separate columns, never merged)
`label_service_relevant` · `label_evidence_strength` (strong/moderate/weak/conflicting/insufficient) ·
`label_identity_consistent` · `label_location_consistent` · `label_eligible_for_recommendation` ·
`label_quarantine_reason` (a reason code) · `reviewer_confidence` (high/medium/low) · `reviewer`, `reviewer_2`, `notes`.
Reviewers must not look at the `HIDDEN_*` system columns before labelling. Ideally two reviewers label every
demo-critical and high-risk pair (merged, negation, geographic contradiction), and agreement (Cohen's κ) is reported.

## Metrics
| Metric | Definition | Source | Status |
|---|---|---|---|
| **Unsupported recommendation rate** (primary safety) | share of shown recommendations that fail the evidence gates. Label-free: gold `eligible = false`. Label-based: `label_eligible = no` among system-eligible | `compare_baselines`, `pair_metrics` | label-free measured; label-based pending |
| Candidate-retrieval recall | relevant pairs (human) that retrieval surfaced | `pair_metrics` | pending |
| Precision@K, NDCG@K | `label_service_supported = yes` over ranks ≤ K | `label_metrics` | pending |
| Evidence-excerpt correctness | excerpt is an exact substring of the cited field (automatic); relevance (human) | `run_benchmark` | automatic measured |
| Quarantine precision / recall | system-ineligible vs `label_eligible = no` | `pair_metrics` | pending |
| Location-validity rate | within radius, inside India, distance recomputes | `run_benchmark` | measured |
| Appropriate-abstention rate | abstentions where reviewers agree no credible candidate existed | manual on abstained queries | pending |
| Coverage | queries with ≥1 eligible result | `summarize` | measured |
| Ranking stability | top-1 / Jaccard across weight scenarios | `run_benchmark` | measured |
| Comparison vs nearest keyword match | all of the above for B1–B4 | `compare_baselines` | label-free measured |

## Current measured results (local CSV build, 2026-11-21; re-run in notebook 05 on Databricks)
Random sample (56 queries):
* unsupported-recommendation rate: **B3 nearest synonym 33.0%**, B2 exact word 31.7%, B1 nearest any 92.3%,
  **CareProof 0.0%** (true by construction, since CareProof only shows eligible rows; the informative numbers
  are the baselines');
* coverage 46/56 for CareProof (82.1%) vs 47/56 for B3 (83.9%). Of 10 CareProof abstentions, 6 had mentions that
  were all held back;
* exact-substring citations 169/169; geographically valid results 169/169; results by category: 43 strong,
  84 moderate, 42 weak;
* top-1 differs from B3 in 31/56; top-1 unchanged under alternative weights in 45–52/56;
* latency (warm, local): CareProof p50 16.6 ms, p95 26.2 ms.

Precision@K, quarantine precision/recall, retrieval recall and appropriate-abstention rate: **PENDING HUMAN REVIEW.**
No value will be reported until labels exist.
