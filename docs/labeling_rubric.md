# Labeling rubric (human review)

**Purpose:** create independent labels so that precision@K, NDCG@K and calibration of the confidence labels can be
measured. The system's own rules must not be used as labels: that would be target leakage.

**Unit:** one (query, system, rank, facility) row in `careproof_eval_labeling_template` (Databricks) or
`evaluation/labeling_template_*.csv` (local, git-ignored).

**Queries:** `evaluation/labeling_queries_v1.csv` is a seeded (20261121) stratified random sample:
7 services × 4 district-rurality quartiles × 2 PINs = 56 queries. It was frozen before any result was inspected.
Each query is assigned to `dev` or `test` **by state**, so facilities and nearby locations never straddle splits.
Use `dev` for any tuning of weights or thresholds; touch `test` once, at the end.

**Blinding:** reviewers should see the facility name, the excerpt and the source URL, **not** the system name or
the score. Shuffle rows before review.

Record three separate judgements. Never merge them:

| Column | Question | Values |
|---|---|---|
| `label_service_supported` | Does the facility's own text (the excerpt, plus the source page if reachable) state that **this facility** offers **this service**? | `yes` / `no` / `unclear` |
| `label_evidence_credible` | Is the claim internally consistent (no merged-facility signs, right city, plausible specialty)? | `yes` / `no` / `unclear` |
| `label_currently_available` | Only if verified by phone or an official current source. Otherwise leave blank. | `yes` / `no` / blank |

Rules:
* A specialty tag alone (e.g. "nephrology") is **not** `yes` for a specific service (e.g. dialysis).
* Planned or under-construction services are `no`.
* If the excerpt belongs to a different facility (merged record), `label_service_supported = no` and
  `label_evidence_credible = no`.
* Record the reason in `notes`. Fill in `reviewer` with initials. Ideally two reviewers label the same 20% so
  agreement (Cohen's κ) can be reported.

**Metrics computed from labels** (`careproof.evaluation.label_metrics`): precision@K = mean(`yes`) over labeled
ranks ≤ K, excluding `unclear`; NDCG@K within the labeled set. Report dev and test separately, with the number of
labeled rows. Until labels exist, these metrics are reported as **PENDING_HUMAN_REVIEW**.
