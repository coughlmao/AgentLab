# error-accumulation

**Question:** How does the benefit of external verification change as error accumulation increases?

## Caveats

- The verifier is the ground-truth check itself (a perfect verifier). A noisy or learned verifier is a different question, deliberately out of scope.
- One model (claude-haiku-5-5), one task family (chained transformations, new in this phase). The comparison that matters is within this experiment -- the verification effect as a function of horizon -- not levels against Phase A.
- Steps execute sequentially within every run; team size (fixed at 4) rotates which agent takes each step. This experiment studies dependency, not parallel speedup.
- Five seeds per cell: per-horizon intervals are wide by construction. The citable numbers are the pooled and across-horizon contrasts.
- Survival probability is P(state correct after step k) against the generator's ground truth; local correctness (the step applied correctly to the input actually received) is recorded separately in steps.csv, so inherited errors are distinguishable from new ones.

## Design

- provider / model: anthropic / claude-haiku-5-5
- family: error_accumulation; difficulty: hard; **blind mode** (agents cannot run checks; the external check is the only check feedback)
- grid (from the manifest): {"family": "error_accumulation", "horizons": [2, 5, 10, 20, 50], "seeds": [0, 1, 2, 3, 4], "strategies": ["orchestrated", "verified"], "team_size": 4}; 50 runs completed
- analysis: declared in the config (series=strategy, group_by=horizon, value=final_state_correct); bootstrap 95% CIs, 5000 resamples over runs within each cell

## Headline: mean final_state_correct [95% CI]

| strategy | horizon=2 | horizon=5 | horizon=10 | horizon=20 | horizon=50 |
|---|---|---|---|---|---|
| orchestrated | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| verified | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |

## Primary contrast: verified - orchestrated

- overall (mean of per-horizon differences): +0.000 [+0.000, +0.000]
- horizon=2: +0.000 [+0.000, +0.000]
- horizon=5: +0.000 [+0.000, +0.000]
- horizon=10: +0.000 [+0.000, +0.000]
- horizon=20: +0.000 [+0.000, +0.000]
- horizon=50: +0.000 [+0.000, +0.000]
- across groups (effect at horizon=50 minus effect at horizon=2): +0.000 [+0.000, +0.000] -- the pre-registered test of whether the effect itself changes across the grouping variable

## Survival and hazard (per-step)

- horizon=2: orchestrated final survival 1.000; verified final survival 1.000; survival divergence step: none
- horizon=5: orchestrated final survival 1.000; verified final survival 1.000; survival divergence step: none
- horizon=10: orchestrated final survival 1.000; verified final survival 1.000; survival divergence step: none
- horizon=20: orchestrated final survival 1.000; verified final survival 1.000; survival divergence step: none
- horizon=50: orchestrated final survival 1.000; verified final survival 1.000; survival divergence step: none

Hazard h(k) = P(first failure at step k | survived through k-1); full curves with CIs are in analysis.json and figures/hazard_*.svg.


## Cost accounting

- total cost: $0.4657; cost per run: $0.0093 (source: summary.json (run ledger))

| strategy | runs | total cost | cost/run | cost per correct subtask |
|---|---|---|---|---|
| orchestrated | 25 | $0.2348 | $0.0094 | $0.0005 |
| verified | 25 | $0.2309 | $0.0092 | $0.0005 |

## Caveats

- The experiment-specific caveats are stated at the top of this summary, where they belong.
- Intervals are percentile bootstrap intervals over runs within each cell; cells hold few seeds, so intervals are wide by construction. Read the contrast intervals, not point estimates alone.
- Figures and every number in this summary are regenerated from the run's declared tabular outputs (metrics.csv and any declared steps table) by `python -m teamscale.analyze <run dir>`; grading and accounting can be verified with `python -m teamscale.replay <run dir>`.
