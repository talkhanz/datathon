# Problem specification

| Item | Specification |
|---|---|
| **Intended user** | Community health worker, NGO coordinator or district planner choosing where to send a patient for a specific service (challenge scenarios: dialysis in rural Bihar, a high-risk delivery in Maharashtra, highway trauma). Decision support only. |
| **Decision supported** | "Which nearby facilities *credibly state* that they offer service S, and how far away are they?" A shortlist to **call and verify**, not a dispatch decision. |
| **Unit of analysis** | A (facility, service) pair: `careproof_facility_service_gold`, one row per pair with any signal. |
| **Query** | (service S, location L, radius R, K), issued at decision time. |
| **Candidate generation** | Pairs for S with *explicit, non-negated, not-planned* text evidence and a usable position within R of L. |
| **Ranking** | Order the candidates by a transparent policy score: evidence strength, corroboration, location trust, distance decay, readiness signals, minus contradiction penalties. |
| **Possible learning targets** | (a) `service_supported` (does the facility's own text state the service?); (b) `evidence_credible`; (c) `currently_available`. Only (a) and (b) can be labelled from data we have. (c) needs a phone call or an official source. |
| **Positive label** | A human judges, from the excerpt and the source page, that *this* facility states it offers *this* service (see `labeling_rubric.md`). |
| **Information at inference time** | Facility record fields (LLM-extracted claims), PIN directory, NFHS district statistics, user query. No real-time availability, no road network, no claim-to-URL mapping. |
| **Label noise sources** | LLM extraction errors and hallucinated or negated items; merged records; stale web pages; aggregator and social-media sources; truncated lists (50-item cap). |
| **Error costs** | **False positive** (recommending a facility that lacks the service) costs travel time in an emergency, so it is the most severe error. **False negative** (missing a real provider) means a longer trip, which is mitigated by abstaining and asking the user to widen the search or call. The system is therefore tuned to be precision-first, with honest abstention. |
| **Required evidence** | A non-empty exact excerpt from the facility's own field, plus its provenance (field, facility_id, record-level source URL). |
| **Abstain when** | Service unsupported; location invalid, ambiguous or unresolvable; no explicit evidence within R (`abstain_no_explicit_evidence_in_radius`); only Low-confidence candidates (`answered_low_confidence_only`, shown as leads). |
| **Offline metrics** | Label-free (now): citation exactness, geographic validity, coverage and abstention, policy-compliance rates of baselines, weight-sensitivity stability, latency p50/p95. Label-based (after review): precision@K, NDCG@K, and calibration of the confidence labels (reliability of High/Medium/Low against `service_supported`). |
| **Product success metrics** (proposed) | Share of shortlisted facilities confirmed by phone; time-to-confirmed-facility; rate of wasted trips. These cannot be measured offline. |
| **Safety limits** | Not medical advice, not emergency dispatch, no availability guarantee; straight-line distance only; NFHS is population context, never patient-level. |

## Problem type: hybrid, mainly deterministic information retrieval
* **Retrieval:** deterministic, interpretable matching with negation and planned-service handling.
* **Ranking:** a transparent policy score (not learned).
* **Geography:** validation, outlier-robust PIN centroids, straight-line distance decay.

### Is supervised training justified now? **No.**
1. There are **no independent labels**. Every available signal is derived from the same text the ranker reads, so a
   model trained on rule-derived labels would only reproduce the rules. That is target leakage.
2. The decisive quantity, current availability, is not observable in the data at all.
3. Without labels we cannot show improvement over the deterministic baseline, so added complexity can't be justified.

**Path to learning, once labels exist:** label the 56-query random sample (≈375 rows) with the rubric. Measure
precision@K for B1–B4 on dev. If CareProof's errors are systematic, fit a regularised logistic model on
interpretable features (field strengths, distinct-statement count, flags, profile conflict), using **state-grouped**
splits and a calibration check on a separate fold. Keep the deterministic system unless the model beats it on
dev *and* test.
