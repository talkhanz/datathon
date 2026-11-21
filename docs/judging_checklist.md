# Judging checklist

## Challenge "musts" (challenge brief)
| Requirement | Where | Status |
|---|---|---|
| Runs on Databricks Free Edition | notebooks 00–05, serverless, Python/PySpark/SQL only | Implemented; Databricks run pending |
| Uses the provided facility dataset | Marketplace `facilities` read directly by 01–03 | Implemented |
| Clear non-technical workflow | notebook 04 widgets + result cards; guided scenarios A–D | Implemented |
| Cites underlying facility text for every claim / score / ranking | exact excerpt + field + matched terms on every card; validation query 7 | Implemented; 187/187 exact on the random sample (local) |
| Communicates uncertainty | confidence label, missing-data and contradiction warnings, verification banner, "none found" | Implemented |
| WHY this facility? | explanation text + quoted evidence + distance | Implemented |
| HOW did you decide? | visible component scores, documented weights (policy), sensitivity analysis | Implemented |

## Judging criteria (event page: Exploration, Modeling, Impact)
| Criterion | Evidence |
|---|---|
| **Exploration** | 03b question-driven EDA + `docs/data_quality_report.md` with figures (claims vs evidence, location vs PIN, rural coverage collapse, contradiction rates); 01 data audit; documented findings: 88 corrupted rows, merged records, coordinate–PIN conflicts, PIN-directory swaps and outliers, specialty-tag vs explicit-text gap (4,162 vs 452 for C-section), NFHS suppression |
| **Modeling** | literature-grounded design (`docs/literature_review.md`); explicit decision not to train without labels (leakage); four baselines B1–B4 with latency; radius-invariant distance decay; distinct-statement corroboration; seeded stratified evaluation sample with state-grouped split; two-stage retrieve-then-rank; deterministic interpretable matcher with negation and planned detection; contradiction detection across independent fields; transparent scoring; baseline comparison; weight sensitivity |
| **Impact** | safer routing for community health workers (challenge scenarios: Bihar dialysis, Maharashtra delivery, highway trauma); honest "none found" results reveal coverage gaps; NFHS context for planners. Proposed: a district evidence-desert view for policy |

## Data-engineering hygiene
- [x] Marketplace tables never written; three-part names; `careproof_` prefix enforced in code
- [x] Configurable target catalog/schema; destination printed before each write; idempotent overwrites
- [x] Join fanout checks that fail loudly; join report table
- [x] PINs kept as strings; PIN directory collapsed to PIN grain before joining
- [x] No credentials in the repo; data git-ignored
- [x] 91 local unit tests incl. radius invariance, determinism, and optimized-vs-reference equivalence

## Honesty
- [x] Measured vs heuristic vs proposed separated in the README
- [x] Precision@K marked PENDING HUMAN REVIEW
- [x] Distances labelled approximate straight-line; availability never claimed
