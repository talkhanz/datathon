# Quarantine rules

`careproof_facilities_quarantine` holds one row per (source record, optional service, reason code). Columns:
`source_row_id, facility_id, facility_name, service, reason_code, severity, evidence, human_review_required,
pipeline_version`. Rows with **high** severity make the record ineligible for normal recommendation. **Medium**
rows lower identity or location confidence and block a *strong* rating. **Low** findings stay eligible with a small
rank penalty and are not written to the table. Quarantined records appear only in the planner/audit view.

| Reason code | Level | Severity | Rule (implementation) | Observed example |
|---|---|---|---|---|
| `CORRUPTED_SOURCE_ROW` | record | high | non-UUID `unique_id` / missing name (CSV fragments) | 88 local rows of markdown fragments |
| `MULTIPLE_FACILITY_IDENTITIES` | record | high | the description's subject is a different, fuzzy-dissimilar named facility (`credibility.description_subject_conflict`) | *Saifee Hospital* → "Prince Aly Khan Hospital is located in Mumbai" |
| `MULTIPLE_FACILITY_IDENTITIES` | claim | high | the supporting passage names another facility and not this one (`credibility.attribution`) | *Srp Hospital*: "Offers dialysis services at SPMM Hospital" |
| `COORDINATE_LOCATION_CONFLICT` | record | high >100 km / medium 25–100 km | coordinates vs centroid of the record's own PIN | "Aravind Eye Hospital": 1,871 km from PIN 110002 |
| `MULTIPLE_INCOMPATIBLE_LOCATIONS` | record | high | listed state not among the PIN's states | — |
| `MULTIPLE_INCOMPATIBLE_LOCATIONS` | record | medium | description/capability mentions a PIN whose centroid is >50 km away | — |
| `UNRESOLVED_LOCATION` | record | high (no position) / medium (PIN centroid only) | — | 5 / 31 records |
| `SHARED_PLACEHOLDER_COORDINATE` | record | medium | ≥3 differently named facilities share exact coordinates | 13 records |
| `AMBIGUOUS_PIN` | record | medium (multi-state) / low (multi-district) | from PIN-grain aggregation | — |
| `MISSING_SOURCE_PROVENANCE` | record | medium | no source URL and no website | 2 records |
| `SERVICE_CONTRADICTION` | claim | high | negated / planned / historical / referral / tie-up language about the service | *District Civil Hospital Kaithal*: "C-section deliveries are referred to Kalpana Chawla GMC" |
| `SPECIALTY_PROFILE_MISMATCH` | claim | high | name/type profile (eye, dental, skin, diagnostic…) incompatible with the claimed service | eye hospital claiming dialysis |
| `WEAK_SERVICE_EVIDENCE` | claim | medium (tag only) / low (single field) | — | — |
| `MISSING_EVIDENCE_EXCERPT` | claim | medium | no attributable sentence states the service | — |
| `POSSIBLE_DUPLICATE` | record | low | same normalised name within 1 km (collapsed in results) | two *Rajindra Hospital Patiala* rows |

**Local counts (2026-11-21):** coordinate conflict 1,001 high + 496 medium; specialty mismatch 426; incompatible
locations 364 high + 76 medium; multiple identities 240; service contradiction 43; corrupted rows 88; unresolved
location 5 high + 31 medium; shared coordinate 13; ambiguous PIN 15; missing provenance 2.

**Known limits:** fuzzy name matching misses look-alike branches of the same chain in different cities (e.g.
"Krushna" vs "Krishna" Hospital), and acronyms are matched heuristically. Every high- or medium-severity row is
marked `human_review_required = true`. Quarantine never deletes source data.
