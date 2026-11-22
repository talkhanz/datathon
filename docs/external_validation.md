# External validation against official government empanelment (Haryana, AB-PMJAY)

Code: `src/careproof/external.py`. Fetch: `python scripts/fetch_external.py`. Run locally with `python scripts/run_local.py external`,
or on Databricks with `notebooks/08_external_validation`. Sources are listed in [`evaluation/external_sources_v1.csv`](../evaluation/external_sources_v1.csv)
(44 official PDFs).

## Why this source
Under Ayushman Bharat PM-JAY, the State Health Agency empanels hospitals **for specific specialities** after checking
their infrastructure and staff against national criteria. The Haryana SHA publishes district-wise lists. They are official,
public and produced independently of both the Virtue Foundation data and CareProof's rules. We use:
- **Speciality lists:** 18 of 22 districts, dated 26/06/2025, except Hisar (24/01/2025). Gurugram, Fatehabad, Kaithal and Mewat had no
  speciality-wise list available.
- **Current empanelled lists:** all 22 districts, dated 03/08/2026.
- **Adverse-status lists** (03/08/2026): suspended, de-empanelled, withdrawn and inactive hospitals.

**Privacy.** The PDFs also contain nodal officers' phone numbers and e-mails. These are dropped at parse time and never stored.
Raw PDFs and parsed tables stay out of Git (`data/external/`, git-ignored).

## Speciality → service mapping
| Service | PM-JAY speciality | Strength |
|---|---|---|
| NICU | Neo-natal care packages / Neonatology | direct |
| Burn care | Burns management | direct |
| Trauma & emergency | Polytrauma / Emergency packages | direct |
| Cardiac surgery | Cardio-thoracic & vascular surgery | direct (only 5 hospitals in Haryana) |
| MRI | MRI | direct |
| C-section | Obstetrics & gynaecology | broad |
| Dialysis | Nephrology | partial (haemodialysis is also billed under other specialities) |

**How to read it.** "Empanelled for X" is strong evidence that the hospital is equipped for X. "Not empanelled" is weak
evidence, because hospitals need not empanel every service. So we report **support rates**: the share of hospitals flagged by
each system that are empanelled for the matching speciality. We do not report accuracy.

## Matching (precision first)
- **Rule:** names and districts must match exactly. Each name is first reduced to its *core*, with legal suffixes, "a unit of …",
  generic words and place names removed. The cores must be identical, or have SequenceMatcher similarity ≥ 0.92.
- **Exclusions:** generic names ("Civil Hospital Rohtak") reduce to an empty core and never match. Ambiguous candidates are dropped.
- **Manual review:**
  - The first fuzzy rule, reused from deduplication, made obvious errors, e.g. Meenakshi Jain Hospital ↔ Jain Free Eye Hospital.
  - All matches from the strict rule were reviewed by eye. One false match (Shivam ↔ Shiva, similarity 0.91) led to the 0.92 threshold.
- **Result:** 98 unique matches, 89 of them with speciality lists. That covers 89 of the 462 Haryana records.

## Results (local run; Databricks run in `careproof_external_validation`)
Share of flagged hospitals that are empanelled for the matching speciality (n in brackets):

| Service | All matched | Keyword match (B3) | CareProof eligible | CareProof held back | CareProof moderate | CareProof weak |
|---|---|---|---|---|---|---|
| NICU | 16% (89) | 55% (11) | 56% (9) | 50% (2) | 60% (5) | 40% (5) |
| Trauma & emergency | 27% (89) | 35% (54) | 35% (37) | 35% (17) | **55% (22)** | **11% (19)** |
| Dialysis | 8% (89) | 20% (20) | 21% (14) | 17% (6) | 18% (11) | 25% (4) |
| MRI | 6% (89) | 9% (11) | 10% (10) | 0% (1) | 0% (4) | 17% (6) |
| Burn care | 7% (89) | 100% (2) | 100% (2) | – | – | 100% (2) |

The service-profile model, which never saw these labels, ranks empanelled hospitals above the rest. ROC-AUC by service:
- NICU 0.76
- burns 0.82
- dialysis 0.72
- MRI 0.70
- C-section 0.88
- trauma 0.62
- cardiac surgery 0.97, but only one empanelled hospital, so not meaningful

### What this shows, honestly
1. **The text claims carry real signal.** Hospitals whose records state a service are 2–4× more likely to be
   government-empanelled for it than matched hospitals in general.
2. **Evidence grading is informative for trauma.** Moderate claims are empanelled 55% of the time and weak claims 11%.
   That gap is the clearest sign that the strength categories mean something. We report it but have **not** retuned the
   gate on this data, to avoid fitting to the check.
3. **The eligibility gates do not beat plain keyword matching on this measure.** Support rates for eligible and held-back
   hospitals are similar, with small samples and overlapping intervals. The gates exist mainly to stop *unsupported* claims
   (negations, other-facility text, location conflicts). Empanelment status cannot test that, because a held-back record can belong to
   a hospital that is genuinely empanelled.
4. **Independent support for the model.** It was trained on text-derived labels from structured fields, yet it predicts
   an external government label well above chance.
5. **Limits:**
   - one state;
   - 89 matched hospitals;
   - lists from 2025–26, while the facility records are undated;
   - name matching misses records whose names differ.

## In the app
- **Result card badges:**
  - "🏛️ Independently corroborated: empanelled under PM-JAY for neonatal care (Haryana SHA, 26/06/2025)", or a neutral
    "on the list, but not for this speciality" note;
  - a ⚠️ warning for hospitals currently suspended or withdrawn (8 of the matches).
- **Badges never change ranking or eligibility.**
- **The Model tab** shows the support-rate chart with 95% Wilson intervals and the model's AUC against the government label.
