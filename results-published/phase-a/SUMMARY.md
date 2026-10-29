# verification-question

**Question:** When does external verification improve team performance?

## Design

- provider / model: anthropic / claude-haiku-5-5
- difficulty: hard; **blind mode** (agents cannot run checks; the external check is the only check feedback)
- grid: 3 strategies x 4 team sizes, 60 runs completed
- intervals: bootstrap 95% CIs, 5000 resamples over runs within each cell

## Headline: mean fraction correct [95% CI]

| strategy | n=1 | n=2 | n=4 | n=8 |
|---|---|---|---|---|
| independent | 0.900 [0.875, 0.950] | 0.950 [0.900, 1.000] | 0.900 [0.875, 0.950] | 0.900 [0.875, 0.950] |
| orchestrated | 0.900 [0.875, 0.950] | 0.900 [0.875, 0.950] | 0.900 [0.875, 0.950] | 0.900 [0.875, 0.950] |
| verified | 0.975 [0.925, 1.000] | 0.975 [0.925, 1.000] | 0.975 [0.925, 1.000] | 0.975 [0.925, 1.000] |

## Primary contrast: verified - orchestrated

Same coordination; the only difference is the external check and its one retry. A confidence interval excluding zero is the verification effect.

- overall (mean of per-size differences): +0.075 [+0.044, +0.106]
- n=1: +0.075 [+0.000, +0.125]
- n=2: +0.075 [+0.000, +0.125]
- n=4: +0.075 [+0.000, +0.125]
- n=8: +0.075 [+0.000, +0.125]

## Cost accounting

- total cost: $0.3410; cost per run: $0.0057 (source: summary.json (run ledger))

| strategy | runs | total cost | cost/run | cost per correct subtask |
|---|---|---|---|---|
| independent | 20 | $0.1017 | $0.0051 | $0.0007 |
| orchestrated | 20 | $0.1107 | $0.0055 | $0.0008 |
| verified | 20 | $0.1286 | $0.0064 | $0.0008 |

## Caveats

- Agents execute sequentially within a run; team size varies the organization (assignment, shared board, verification), not parallel wall-clock speedup.
- One model, one task family (codebase_repair), generated tasks; each seed is a fresh task instance, so intervals include task variance as well as agent variance.
- Cells hold few seeds, so intervals are wide by construction. Read the contrast interval, not point estimates alone.
- The verified strategy's verifier is the ground-truth check itself (a perfect verifier). A noisy or learned verifier is a different question.
- Figures: figures/scaling_curve.svg and figures/cost_breakdown.svg, regenerated from metrics.csv by `python -m teamscale.analyze <run dir>`. Grading and accounting can be verified with `python -m teamscale.replay <run dir>`.
