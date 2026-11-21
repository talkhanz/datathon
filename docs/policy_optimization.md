# KPI-driven policy optimisation (Bayesian optimisation + genetic algorithm)

**What is tuned:** the six ranking weights (service relevance, evidence strength, identity confidence, location
confidence, readiness evidence, distance). They apply **only to candidates that already passed the hard eligibility
gates**. Gates, evidence categories and quarantine rules are not tunable, so no optimiser can buy a KPI gain with an
unsafe recommendation.

**Why not optimise accuracy directly?** There are no independent human labels yet (`problem_spec.md`). The true
objective, NDCG@K against labels, is implemented (`optimization.ndcg_objective`) and runs in the same loop once the
blind review sheets are filled.

## KPIs (label-free; frozen random query sample)
| KPI | Definition | Direction |
|---|---|---|
| proximity | mean straight-line km of the top-1 recommendation | minimise |
| evidence | mean `evidence_strength_score` of the top-K | maximise |
| consistency | mean identity × location confidence of the top-K | maximise |

Coverage and the unsupported-recommendation rate are fixed by the gates and do not change with the weights.

## Hard constraints (infeasible trials are rejected by the samplers)
* **C1** a weak claim at 0 km must not outrank an otherwise-equal moderate claim at 10 km;
* **C2** a moderate claim at 0 km must not outrank an otherwise-equal strong claim at 10 km;
* **C3** every weight ≥ 0.02.

## Method
* **Bayesian optimisation:** Optuna `TPESampler` (seed 20261121, 150 trials, 20 random start-up trials), maximising a
  stated scalar utility `evidence + consistency − top1_km / 25` (policy: 1 evidence point ≈ 25 km).
* **Genetic algorithm:** Optuna `NSGAIISampler` (population 30, 150 trials), three objectives, giving a **Pareto front**.
* Both include the hand-set policy as a seeded trial. They tune on the **dev** states only; we report on held-out
  **test** states with a paired bootstrap (5,000 resamples).
* Speed: each trial re-weights the precomputed `RecommendationIndex` without re-parsing (about 0.3 s per trial for
  35 queries).

## Results (local build, identical tables to Databricks; re-run in notebook 06)
Feasible Pareto front: 21 policies. On the held-out **test** states (17 answered queries), the rounded TPE policy vs
the hand-set policy:

| KPI | mean Δ | 95% CI | queries better / worse |
|---|---|---|---|
| top-1 distance (km) | **−2.15** | [−5.07, −0.03] | 4 / 0 |
| evidence (top-K) | +0.007 | [−0.001, +0.017] | 3 / 5 |
| consistency (top-K) | +0.007 | [0.000, +0.019] | 2 / 0 |

The optimised policy is never worse on average and is better on held-out distance, but **the effects are small** and
only about 4–5 of 17 queries change. We adopted it, rounded, as the default
(`0.05 / 0.35 / 0.14 / 0.20 / 0.02 / 0.24`) and keep `hand_set_v1` as a named scenario. The optimiser's main lesson:
**location consistency deserved more weight, and readiness and service relevance less**, than the hand-set guess.

## Caveats
Small query sample; label-free KPIs are proxies; the utility scalarisation is a policy choice. Treat the Pareto
front as a menu of trade-offs for decision-makers, not as evidence that the ranking is more correct.
