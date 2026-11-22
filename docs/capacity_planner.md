# Capacity planner (app tab "🗺️ Capacity planner")

Code: `src/careproof/coverage.py`. Tests: `tests/test_coverage.py`. It runs live in the app (about 0.5 s per state) on the
gold, PIN and model tables, and it is meant for **planners, not referrals**.

## Problem
This is a maximal covering location problem (Church & ReVelle, 1974) with an equity term. The setup is close to
GA-based healthcare location-allocation work ([IJPH 2023](https://ssph-journal.org/journals/international-journal-of-public-health/articles/10.3389/ijph.2023.1605015/pdf);
[Li et al. 2017](https://ideas.repec.org/a/sae/envirb/v44y2017i6p994-1011.html)).

| Element | Definition |
|---|---|
| Demand | PIN areas of one state with a defensible centroid. The weight is post offices per PIN (a settlement proxy) or equal per PIN. **There is no population data.** |
| Supply | Facilities that are **eligible** for the service (credible documented evidence), including those just across the state border |
| Candidates | Existing hospitals that do not document the service. They need location confidence ≥ 0.6, no high-severity finding, and no implausible name profile (eye, dental, IVF…). An optional minimum model profile fit applies. Only candidates that would newly cover some demand are kept |
| Coverage | A PIN is covered when its straight-line distance to a credible provider is ≤ the threshold |
| Objective | (1 − e) × demand-weighted coverage + e × district-equal coverage, where e is the "equity emphasis" |

## Solvers
| Solver | Notes |
|---|---|
| Greedy | The classic heuristic, with a (1 − 1/e) approximation guarantee for this monotone submodular objective |
| GA, random start | K integer genes, repaired so every site is distinct |
| **GA seeded with greedy** (shown) | Never worse than greedy, by construction |
| Random | 200 plans of the same size; the mean and 95th percentile are reported |

Local check over 48 runs (6 states × 4 services × K ∈ {3, 8}, threshold 50 km, e = 0.5):
- The seeded GA beat greedy in 5 of 48 runs (by up to +0.014 objective) and tied in the rest.
- The GA from a random start beat greedy in 5 runs, was worse in 22 (by small margins) and tied in the rest.
- All solvers were well above the best 5% of random plans.
- Takeaway: greedy is already strong here, and the GA mainly adds small improvements and an independent check.
- A test confirms that the GA finds the brute-force optimum on a small instance.

## What the app shows
- **Before/after numbers:** demand within reach, districts with no access, average distance to a credible provider,
  and the gain over random sites.
- **Map:** covered, newly covered and uncovered PINs, existing providers and proposed sites.
- **Proposed-sites table:** new demand reached and model profile fit for each site.
- **Charts:**
  - solver comparison;
  - diminishing-returns curve for 0–10 sites;
  - efficiency-versus-equity frontier (equity 0 → 1);
  - GA convergence for both runs;
  - before/after dumbbell chart for the districts that gain most.

## Limits (also stated in the app)
- Distances are straight-line, not travel time.
- Demand is a proxy, not population.
- "Adding a service" is hypothetical: staff, equipment, cost and capacity are not modelled.
- Existing providers are credible *records*, not verified capacity.
