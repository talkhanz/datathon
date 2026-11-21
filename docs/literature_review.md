# Literature review: what the evidence says and what CareProof does about it

Scope: the six design questions behind CareProof. Each section gives what the literature says, what we observed
in *this* dataset, and the design decision we took. Citations were found and checked by web search on
2026-11-21. Where we read only abstracts or secondary sources, we say so.

## 1. What is this dataset, and how should its fields be trusted?
* The facility data comes from the Virtue Foundation / Databricks for Good "Foundational Data Refresh". It is built
  from web sources with a multi-step LLM pipeline: classify medical relevance, then organisation type, then extract
  specialties, equipment and procedures. Entity resolution uses Splink. Databricks reports millions of pages processed
  with OpenAI GPT models (figures differ between sources and are self-reported).
  [Databricks blog](https://databricks.com/blog/databricks-good-and-virtue-foundation-partnering-connect-medical-volunteers-critical-health),
  [DAIS session](https://www.databricks.com/dataaisummit/session/building-ai-matters-powering-global-healthcare-virtue-foundation).
* LLM extraction is not guaranteed to be grounded in the source document. Recent work scores each extracted item
  against its aligned source context and rejects ungrounded ones
  ([SAFEPASSAGE, arXiv 2510.00276](https://www.arxiv.org/pdf/2510.00276)). It also distinguishes groundedness
  errors from citation errors ([WebCiteS, ACL 2024](https://aclanthology.org/2024.acl-long.806.pdf)).
* **Observed here:** extraction artifacts such as "MRI not provided in the supplied content". The same sentence
  recurs across `procedure` / `capability` / `equipment` in about 10% of multi-field claims. Records merge several
  facilities (e.g. an eye hospital carrying a Delhi hospital's dialysis unit).
* **Decision:** treat every field as an *LLM-extracted claim*, not as independent evidence.
  Corroboration = number of *distinct* supporting statements, with near-duplicates collapsed, rather than number of
  fields. We cannot re-check claims against their source pages: claim-to-URL links are absent and Free Edition has
  restricted internet. So we always show the exact claim text and never assert availability.

## 2. Negation, hypotheticals and "not yet" services
* NegEx (Chapman et al.) detects negation with trigger terms and scope windows. ConText (Harkema, Dowling,
  Thornblade, Chapman, *J Biomed Inform* 2009) extends it to hypothetical, historical and experiencer status without
  syntactic parsing. Performance transfers reasonably across note types but drops on some
  ([PMC2757457](https://pmc.ncbi.nlm.nih.gov/articles/PMC2757457), [W07-1011](https://preview.aclanthology.org/moar-dois/W07-1011.pdf)).
* **Observed here:** "Dialysis not available", "Without dialysis", "300-bed trauma care hospital *under construction*".
* **Decision:** a ConText-style rule layer with negation triggers before or after the term, plus a
  *not-yet-operational* class (under construction, proposed, upcoming, will be started). This is the analogue of
  ConText's "hypothetical". Matched mentions are contradictions, never support.

## 3. Entity resolution and duplicates
* Probabilistic record linkage follows Fellegi–Sunter. Splink implements it at scale, with EM-estimated parameters
  ([Linacre et al., IJPDS 2022](https://doaj.org/article/dc08b00d28a54868968c180e5273a235);
  [Splink developments, PMC10929522](https://pmc.ncbi.nlm.nih.gov/articles/PMC10929522)). The source pipeline
  already used Splink.
* **Observed here:** residual exact duplicates (11), same-name records within 1 km (e.g. two "Rajindra Hospital
  Patiala" rows), and the opposite failure, over-merging (one record spanning Hyderabad, Delhi and Puducherry).
* **Decision:** do *not* re-run probabilistic linkage, since there are no labels to validate it and a high risk of
  new merges. Instead: (a) collapse same-name records within 1 km at recommendation time and disclose it;
  (b) detect over-merging through *independent-field disagreement*: coordinates vs own-PIN centroid, state vs PIN
  state, and facility name or type vs claimed service.

## 4. Geographic access: distance, decay and its limits
* Spatial-accessibility work uses floating catchments (2SFCA: Luo & Wang 2003) and distance-decay weighting within
  catchments (E2SFCA: Luo & Qi 2009) ([PMC3520708](https://pmc.ncbi.nlm.nih.gov/articles/PMC3520708),
  [Wikipedia summary](https://en.wikipedia.org/wiki/Two-step_floating_catchment_area_method)).
* Whether Euclidean distance is a good proxy for travel in LMICs depends on the setting. It can be adequate where
  road networks are dense, and it fails where networks are heterogeneous or terrain is difficult
  ([IJHG review PMC4086697](https://pmc.ncbi.nlm.nih.gov/articles/PMC4086697)). Weiss et al. (*Nature Medicine*
  2020) built global travel-time maps from a friction surface and found large walking-vs-motorised gaps
  ([Google Research page](https://research.google/pubs/global-maps-of-travel-time-to-healthcare-facilities/)).
* **Observed here:** we measured that a radius-relative linear distance score (`1 − d/max_km`) **reordered the
  same facilities when the user changed the radius** (6/6 test queries).
* **Decision:** a fixed distance-decay function, `1/(1 + d/15 km)`. The 15 km is a policy constant in the spirit of
  E2SFCA decay. It is independent of the radius, so the radius only filters. This is enforced by a test. Distances
  are always labelled straight-line. A travel-time surface (Weiss et al.) is the natural next step, but it is not
  in the provided data and is out of scope for Free Edition without external downloads.

## 5. Labels, weak supervision and when to train
* Data programming / Snorkel combines noisy labeling functions into probabilistic labels without ground truth
  ([Ratner et al., VLDB 2017, arXiv 1711.10160](https://arxiv.org/pdf/1711.10160)).
* **Our position:** the only label sources available are heuristics over the *same* text the ranker reads. Training
  a model on labels derived from its own input fields would be target leakage: it could only learn to reproduce the
  rules. **No supervised model is justified until independent human labels exist.** We ship a labeling rubric and a
  stratified, seeded query sample. Snorkel-style label modelling becomes useful later, to combine human labels
  with rules.

## 6. Validation design, calibration and abstention
* Spatially dependent data needs blocked or grouped cross-validation. Random folds underestimate error
  ([Roberts et al., *Ecography* 2017](https://www.biometrie.uni-freiburg.de/mitarbeiter/dormann/roberts-et-al-2017-ecography.pdf)).
  **Decision:** any future learned model is split by state (or district) groups, with duplicates kept in the same
  fold.
* Calibration is a separate property from accuracy. It is assessed with reliability diagrams and ECE
  ([Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html)); ECE estimators have known pitfalls
  ([Roelofs et al. arXiv 2106.07998](https://arxiv.org/pdf/2106.07998)). **Decision:** our scores are *not*
  probabilities and are never shown as percentages. The High/Medium/Low labels are documented rules. Calibration
  will be measured only once human labels exist.
* Selective prediction (reject option, Chow 1957; risk–coverage curves, Geifman & El-Yaniv 2017,
  [arXiv 1705.08500](https://arxiv.org/pdf/1705.08500v1)). **Decision:** explicit abstention states:
  `abstain_no_explicit_evidence_in_radius`, `answered_low_confidence_only`, and location errors. Once labels exist,
  report a risk–coverage curve over the confidence rules.

## 7. Prior work on this exact dataset (to differentiate from)
* *ArogyaTrust* (Devpost): weighted keyword matching across description, specialties, equipment and procedures,
  plus per-capability LLM verdicts and trust tiers ([devpost](https://devpost.com/software/arogyatrust)).
* *CareTrust India* (DAIS 2026 hackathon): a trust score based on the availability of capability, equipment,
  procedure and capacity data ([devpost](https://devpost.com/software/caretrust)).
* *CareDelta* (Databricks App): integrated six external sources, including HMIS, OSM and the India Post directory
  ([devpost](https://devpost.com/software/caredelta)).
* **How CareProof differs:** (1) recommendations are *location-conditioned*, not a global trust tier;
  (2) contradiction detection uses *independent-field disagreement* (coordinates vs PIN vs state vs specialty
  profile), not field presence; (3) negation and not-yet-operational handling; (4) corroboration counts distinct
  statements, because fields come from the same LLM extraction; (5) facility size is never used as evidence,
  whereas CareTrust uses capacity data; (6) explicit abstention and a measured baseline comparison;
  (7) no LLM in the decision path, so every result is reproducible and auditable.

## 8. External ground truth considered
* India's Health Facility Registry (ABDM HFR) is pre-filled from NHRR data ([facility.abdm.gov.in](https://facility.abdm.gov.in),
  [HFR consultation paper](https://abdm.gov.in:8081/uploads/Consultation_Paper_on_Health_Facility_Registry_be118825ff.pdf)).
  We found a public dashboard but no bulk download or API in the sources we checked, so it is not used. If
  obtained, it would provide independent existence labels.
* Facility-list data-quality work in LMICs (MEASURE Evaluation's SQUAD spatial-anomaly checks; geocoded
  sub-Saharan inventories in *Scientific Data* 2019; Senegal triangulation in 2024) supports our approach:
  anomaly-driven spatial QA plus triangulating independent fields
  ([SQUAD](https://www.measureevaluation.org/publications/fs-18-263), [Senegal, PMC10808422](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10808422/)).
