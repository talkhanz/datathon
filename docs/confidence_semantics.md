# Confidence semantics

Seven concepts are kept separate everywhere: code, tables, UI and the printed summary.

| Concept | Field | Meaning | Is NOT |
|---|---|---|---|
| Service relevance | `service_relevance_score` | How directly the strongest attributable passage names the service (procedure/equipment 1.0 … name 0.7) | proof the service exists |
| Evidence strength | `evidence_category` + `evidence_strength_score` | How explicitly and consistently the record documents the claim (strong / moderate / weak / conflicting / insufficient) | current availability |
| Identity confidence | `identity_confidence_score` | Confidence the record describes one identifiable facility (1 − severity costs of identity findings) | facility quality |
| Location confidence | `location_confidence_score` | Agreement of coordinates, PIN, state; placeholder/ambiguity penalties | travel time |
| Readiness evidence | `readiness_evidence_score` | Text mentions a dedicated unit, quantities, 24x7 near the service | capacity today |
| Current availability | `availability_status` | **Only** what the user confirmed by phone this session; default "Current availability not verified" | ever inferred by CareProof |
| Final ranking score | `final_ranking_score` | Weighted ordering of eligible candidates (policy weights) | a probability or a percentage |

**Display labels:** Strong documented evidence · Moderate documented evidence · Weak documented evidence ·
Conflicting evidence · Insufficient evidence · Current availability confirmed by user (session only) ·
Current availability not verified.

**Wording:** "The available record contains moderate evidence that X may offer dialysis." Never "X offers
dialysis". Scores are shown under an expander labelled "policy weights; not probabilities".

**Calibration:** none of these scores has been calibrated against labels. Once human labels exist, report the
share of `service_supported = yes` per evidence category (see `evaluation_protocol.md`).
