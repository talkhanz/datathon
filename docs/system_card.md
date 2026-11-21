# System card: CareProof Referral Navigator

**Intended use:** help community health workers (incl. ASHAs), NGO care coordinators, referring-facility staff and
referral coordinators find the **most credibly documented** facilities for a service near a location in India, see
the exact evidence and its weaknesses, and verify by phone before referral.

**Out of scope:** emergency dispatch; medical advice; clinical triage or choosing a treatment; patient-specific
suitability; verifying clinical quality, beds or staffing; travel-time routing; use outside India.

## What the system shows, and what kind of knowledge it is
| Kind | Examples in the UI | Source |
|---|---|---|
| **Documented facility claims** | exact excerpt, matched field/terms, source URL | Virtue Foundation record (LLM-extracted from public web pages) |
| **CareProof inferences** | evidence category, identity/location confidence, findings, eligibility, distance, rank | deterministic rules in `src/careproof` (policy weights, not learned) |
| **Human verification** | call outcome and checklist | the user, this session only; never written back as fact |
| **Unknown current conditions** | service today, specialist on duty, equipment working, beds/admission | not in the data; must be verified |
| **Out-of-scope clinical decisions** | whether the patient needs this service, urgency, transport mode | clinician / referral protocol |

**Data:** facilities (~10k, web-extracted claims); India Post PIN directory (post-office grain, aggregated to PIN);
NFHS-5 district indicators (context only). All are read-only Marketplace tables. Derived tables are prefixed
`careproof_`.

**Approach:** retrieve → exact evidence → negation/attribution → identity & location checks → **hard eligibility
gates** → rank eligible only → alternatives → verification → abstain when nothing qualifies. There is no ML model
and no LLM in the decision path (no independent labels exist, so training would leak; see `problem_spec.md`).

**Evaluation data:** curated behaviour benchmark (23 queries); seeded random query sample (56, state-grouped
dev/test); stratified facility-service pair benchmark (142). See `evaluation_protocol.md`.

**Metrics (Databricks notebook 05, 2026-11-21):** unsupported-recommendation rate of the nearest-synonym baseline 31.9% vs
CareProof 0% by construction; coverage 46/56; exact citations 169/169; valid locations 169/169; top-1 stable under
alternative weights in 45–52/56; p50 latency 7.4 ms. **Precision@K, quarantine precision/recall, retrieval recall:
pending human review.**

**Subgroups:** coverage collapses in rural districts (most rural quartile: 2% of facilities, and 114/189 districts
have none), so abstentions concentrate there.

**Confidence semantics:** evidence categories describe documentation quality, not the probability that a service
operates today. Scores are never shown as percentages (`confidence_semantics.md`).

**Failure modes:** inherited extraction errors; merged records that pass every rule; look-alike chain branches;
English-only text; stale pages; straight-line distance in difficult terrain; fuzzy-name false positives
(`quarantine_rules.md`).

**Safety and verification:** the safety banner is always visible; abstention is explicit; a checklist and next
alternative are shown when a call fails; no patient data is collected (`safety_and_limitations.md`).
