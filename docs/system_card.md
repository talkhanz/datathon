# System card: CareProof India

**Intended use:** produce a shortlist of facilities whose own web-extracted listing explicitly states a requested
service near a location in India, with exact quotes, transparent scores and warnings, so a health worker or planner
knows **whom to call first**.

**Out of scope:** emergency dispatch; medical advice; guaranteeing availability, quality, cost or staffing;
patient-level risk assessment; ranking facilities by clinical quality; use outside India.

**Data sources:** Virtue Foundation facilities (about 10k, web-extracted by an LLM pipeline; claims, not facts);
India Post PIN directory (post-office grain; 7–8% of rows unusable as coordinates); NFHS-5 district indicators
(population statistics, context only). All are read-only Marketplace tables.

**Approach:** deterministic retrieval, then a transparent policy score (weights are policy choices, not learned).
No ML model and no LLM in the decision path. See `architecture.md` and `decision_log.md`.

**Evaluation data:** (1) 23 curated edge-case queries (behaviour); (2) 56 seeded, stratified random queries
(service × district-rurality quartile), split dev/test by state. Human labels are pending.

**Metrics** (local CSV run; Databricks run pending):

| Metric | Random sample (56 queries) |
|---|---|
| Exact-substring citations | 187/187 |
| Geographically valid results | 187/187 |
| Queries with ≥1 result (coverage) | 46/56 (82.1%); abstentions 10, of which 8 in the two most rural quartiles |
| Top-1 differs from nearest-synonym baseline | 29/56 |
| Synonym-baseline results failing the evidence policy (tag-only, negated or contradicted) | 57/188 |
| Top-1 unchanged under 4 alternative weight policies | 47–52 / 56 |
| Latency per query (warm, local) | p50 8.6 ms, p95 20.1 ms |
| Precision@K, NDCG@K, calibration | **PENDING HUMAN REVIEW** |

**Subgroup behaviour:** coverage falls sharply with rurality. The most rural quartile of districts (branch-office
proxy) holds 2% of facilities, and 114 of its 189 districts have none. Abstentions concentrate there. Do not
generalise urban results to rural queries.

**Confidence semantics:** High / Medium / Low are **rules**, not probabilities. High: service match ≥0.9,
evidence ≥0.7, penalty ≤0.05, location ≥0.6. Low: evidence <0.5, penalty ≥0.25, or location <0.4. Medium:
everything else. They have not been calibrated against labels; once labels exist, report the share of `yes`
labels per confidence level.

**Known failure modes:** LLM-extraction errors inherited from the source; merged records that pass all consistency
checks; facilities described only in Indian languages (descriptions are essentially English-only); stale pages;
co-located records snapped to a shared coordinate; straight-line distance under-states travel in hilly or poorly
connected regions; district name changes break NFHS matching (84.7% matched locally).

**Ethics and safety:** the system shows the evidence and its weaknesses rather than hiding them, and abstains
rather than guessing. It never infers individual characteristics from district data. Coverage bias against rural
areas is disclosed.

**Verification requirement:** every result carries "Availability is NOT verified … call before travelling … in an
emergency call 108/112."
