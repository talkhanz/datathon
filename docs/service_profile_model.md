# Service-profile model, feature selection and GA inverse design

Code: `src/careproof/modeling.py`, `src/careproof/ga.py`. Run: `notebooks/07_service_profile_model` or
`python scripts/run_local.py model`. Tests: `tests/test_modeling.py`.

## Question and label
*From a facility's **structured** attributes alone, how likely is it that the facility's own text documents service S?*

- **Label (silver):** `documented = 1` if the evidence engine found an attributable, non-negated statement of S
  (`evidence_level == "explicit_text"`). The label describes the **record**, not real-world availability. Human
  labels are still `PENDING_HUMAN_REVIEW`.
- **Why this is not circular (cf. D13):** the label comes from name, description, procedure, equipment and capability
  text. None of those fields, and nothing derived from them (item counts included), is a feature. This is enforced in
  `build_features` (`FORBIDDEN_SOURCES`). A test scrambles those fields and asserts that the feature matrix does not change.

## Features (94 candidates)
| Group | Features |
|---|---|
| Size / staff | doctors (log), beds (log), years operating, number of specialty tags, missing-value flags |
| Type / operator | hospital / clinic / dentist; public vs private; government / academic / philanthropy / community / faith affiliation |
| Specialty tags | the top 60 tags, plus one "any related tag" flag per service (the service's own tag patterns) |
| Digital footprint | social channels, followers, posts, facts on profile, official phone / website / e-mail, logo, staff listed |
| Location context | facilities within 10 km (log), an urban-density proxy |

## Protocol
1. **State split.** Whole states go to dev or test (seeded; about 34% of facilities are in test states), so no state is in both.
2. **Filter selection (dev only).** Drop near-constant features, rank the rest by mutual information, and prune pairs with |Spearman| > 0.9.
3. **GA wrapper selection (dev only).** The chromosome is a feature on/off mask. Fitness is the inner 3-fold,
   state-grouped CV average precision of a logistic regression, minus 0.001 per feature. Settings: population 24,
   up to 18 generations, elitism, tournament selection, uniform crossover, bit-flip mutation. The population is
   seeded with the top-k features by mutual information.
4. **Model choice (dev only).** Candidates are logistic regression and gradient-boosted trees, each on the GA, filter and all-feature sets, plus the
   **specialty-tag rule**. The winner is the best by dev CV average precision. If the rule wins, no model is used for that service.
5. **Report on test states**: average precision, ROC-AUC, Brier score, ECE, reliability bins, and permutation importance.
6. **Scores for the app** are out-of-fold over all states, so no facility is scored by a model that saw its state.

## Results (held-out test states, local run)
| Service | Chosen | Features | AP | Tag-rule AP | Base rate | ROC-AUC | Brier |
|---|---|---|---|---|---|---|---|
| Dialysis | logistic · GA | 22 | 0.509 | 0.352 | 0.110 | 0.909 | 0.069 |
| C-section | logistic · GA | 19 | 0.201 | 0.107 | 0.052 | 0.841 | 0.045 |
| NICU | boosted · all | 94 | 0.399 | 0.274 | 0.069 | 0.905 | 0.050 |
| MRI | logistic · GA | 29 | 0.389 | 0.174 | 0.092 | 0.833 | 0.069 |
| Cardiac surgery | logistic · GA | 22 | 0.505 | 0.367 | 0.064 | 0.926 | 0.040 |
| Burn care | logistic · GA | 17 | 0.402 | 0.372 | 0.022 | 0.880 | 0.015 |
| Trauma & emergency | boosted · all | 94 | 0.796 | 0.585 | 0.308 | 0.890 | 0.119 |

- The model beats the tag rule on every service, and the GA subset (10 to 29 of 90 features) is chosen for 5 of 7.
- The burn-care gain is small (+0.03).
- In the first run, without the "any related tag" flags, the burn model was far *worse* than the rule (0.06 vs 0.37).
  That run is what motivated the flags and letting the rule compete.
- The Databricks run writes the authoritative numbers to `careproof_model_metrics`.

**Caveats.**
- Missing-value flags such as "beds not listed" are predictive. They measure how richly a record is documented, not clinical
  capability.
- An average precision of about 0.2 to 0.5 means the structured profile is a weak-to-moderate signal. That is enough for triage, and
  not enough to recommend a facility.

## GA inverse design: the "optimal provider profile"
- A second GA searches the selected features for the profile that maximises the model score.
- The search is penalised when it strays from real documented facilities: the penalty applies to a nearest-neighbour
  distance beyond the typical one.
- It is also constrained:
  - one-hot groups stay one-hot;
  - the tag count is capped at the 90th percentile among documented facilities and kept consistent with the tag-count feature;
  - missing flags are fixed to "listed".
- The population is seeded with real documented facilities. The app shows each feature next to documented and undocumented
  medians and rates, plus the closest real facility.
- **This is associational, not causal.** Adding a nephrology tag does not create a dialysis unit.

## How the app uses it (never as a recommendation)
- **Profile fit** bar on each recommended card. The "Why?" popover adds a chart comparing this facility, the GA-optimal profile and a typical
  documented provider. It is a cross-check only: an eye hospital that "documents" cardiac surgery fits poorly.
- **Unverified leads** (collapsed). These are facilities within the radius that have **no** attributable text evidence but score ≥ 0.5.
  The list excludes:
  - negated or other-facility claims;
  - any facility with a high-severity finding;
  - location confidence below 0.6.
  The leads are labelled "Not recommendations… use only to decide whom to phone". Ranking, gates and abstention are unchanged.
- **Model & genetic optimisation tab**:
  - model comparison against the base rate;
  - permutation importance;
  - convergence of both GAs;
  - optimal profile against real facilities;
  - calibration.
