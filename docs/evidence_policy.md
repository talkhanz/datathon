# Evidence policy

Every facility field is an **LLM-extracted claim from public web pages**, not verified fact
(see `literature_review.md` §1). CareProof never states that a facility offers a service. It reports how well the
available record documents the claim, and it requires human verification before referral.

## Pipeline (`src/careproof/pipeline.py::recommend`)
1. Normalise the service (`service_matching.resolve_service`).
2. Resolve the location (`geography.resolve_location`), with actionable errors and ambiguity warnings.
3. **Retrieve** every record in the radius that mentions the service (broad; evaluated separately for recall).
4. Extract exact excerpts: a sentence/element window copied verbatim from the source field.
5. Check negation and non-current language: *not available, does not provide, planned, proposed, under construction,
   formerly, discontinued, referred to, tie-up with, through a partner, near a hospital offering*.
6. Check **attribution**: a passage that names a different facility (fuzzy-dissimilar name) and not this one is
   *not* evidence for this facility. Example: "Offers dialysis services at SPMM Hospital" in *Srp Hospital*'s record.
7. Assess identity and location consistency (facility-level findings, `quarantine_rules.md`).
8. Detect duplicates and merged records.
9. Apply the **hard eligibility gates** (below).
10. Quarantine anything that fails (`careproof_facilities_quarantine`, planner/audit view).
11. **Rank eligible candidates only.**
12. Return alternatives (`additional_eligible_beyond_top_k`).
13. Require call-ahead verification (`referral.py`).
14. Abstain when no candidate passes.

## Evidence categories (`scoring.evidence_category`)
| Category | Rule (as implemented) |
|---|---|
| **Strong documented** | An attributable explicit statement in the **description** + ≥2 *distinct* supporting statements + corroboration (related specialty tag or a structured field) + **no** medium/high identity or location finding + source provenance present |
| **Moderate documented** | Attributable statement in the description **or** ≥2 distinct statements; no high-severity finding |
| **Weak documented** | One distinct statement in a single structured field |
| **Conflicting** | Negated, planned, historical or third-party language; a passage attributed to another facility; a facility-name/type profile mismatch (e.g. an eye hospital claiming dialysis); or any high-severity facility finding |
| **Insufficient** | No attributable explicit statement (specialty tag only, negated only, other-facility only) |

"Distinct statements" collapses near-duplicates, and a bare term contained in a longer sentence counts once. The
source fields come from the same LLM extraction of the same pages, so repetition is not independent corroboration.

## Hard eligibility gates (`scoring.eligibility`, thresholds in `config.GATES`)
```
eligible = evidence_category >= min_evidence_category (default: weak; conflicting/insufficient never pass)
       AND supporting excerpt present
       AND location_confidence_score >= 0.6
       AND identity_confidence_score >= 0.5
       AND no high-severity identity/location/claim finding
       AND not a probable merged or misattributed record
       AND provenance fields present (facility_id, source_row_id, evidence_field)
```
Gates run **before** ranking, so distance can never rescue an ineligible record (tested in
`tests/test_pipeline.py::test_distance_never_rescues_ineligible`). Within the eligible set, the category dominates
`evidence_strength_score` (strong 0.9 / moderate 0.6 / weak 0.25 base), so a weak claim at 0 km cannot outrank a
moderate one at 10 km (`test_nearby_weak_does_not_beat_slightly_farther_moderate`).

The minimum category is configurable. With `moderate` instead of `weak`, eligible counts fall sharply (e.g.
C-section 353 → 104). We keep `weak` as the default and label it prominently, because weak still means an explicit,
attributable, uncontradicted statement. This is a policy choice; see `evaluation_protocol.md` for how to re-assess
it once labels exist.

## What is never evidence
Bed counts, doctor counts, specialty tags alone, the number of scraped mentions, NFHS district burden, and missing
values (absence of a mention is not treated as negative evidence).

## Measured effect (local CSV build, 2026-11-21)
Explicit mentions → eligible: dialysis 1,052 → 785 · C-section 449 → 353 · NICU 734 → 541 · MRI 944 → 721 ·
cardiac surgery 589 → 443 · burn care 216 → 154 · trauma/emergency 3,201 → 2,181.
