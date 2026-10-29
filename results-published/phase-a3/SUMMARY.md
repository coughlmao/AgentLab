# model-calibrated-generation

**Question:** Can the generator produce task families whose measured first-attempt failure rate on the target model lands inside a requested band, and stay there on held-out seeds?

## Caveats

- One model (claude-haiku-5-5), one task family (standalone transformation steps from the chained family's template set, extended by the Phase A-3 ladder). Calibrated bands are properties of this model and this family; portability to another model is a different question, deliberately out of scope.
- Bands demonstrated are 10% +/- 3% and 5% +/- 2% only. The 2% and 20% bands from the general proposal are not claimed: they may sit outside the ladder's range, and claiming only what the demonstration covers is the point of the exercise.
- The control law is bisection over a fixed ladder with Wilson-CI error signals. Calibration efficiency numbers describe this controller; a different control law could be cheaper, and the efficiency results are reported so that comparison is possible.
- Failure is first-attempt, blind, single-agent failure on standalone steps -- not chain failure, not team failure. This phase calibrates the step distribution; what calibrated steps do inside chains and teams is the successor phase's question.
- Probes whose outcome was an infrastructure error (API/transport) are excluded from every failure rate and counted separately in the bundle; they are not model failures.

## Design

- provider / model: anthropic / claude-haiku-5-5; blind single-attempt probes of standalone ladder steps
- ladder: 10 settings (generator ladder-gen-1); control law: bisection, error = signed Wilson-CI distance from the band; max 6 settings per band, batches of 200
- spend: $2.0840 (ledger $2.0840) of $10.00 cap; 1500 probes, 2971 API calls, wall 3136.094s

## Bands

### Band t10: target 10% ± 3%

- status: **declared_failure** (no_stable_setting)
- search efficiency: 3 iterations, 600 probes, 1170 API calls, 3839643 tokens, $0.9229, 1377.969s
  - iter 1: `plain-s32-k2` n=200 failures=4 rate=0.02 CI=[0.007804, 0.050287] error=-0.019713 -> move up: search [5, 9]
  - iter 2: `inter-s32-k4` n=200 failures=6 rate=0.03 CI=[0.01382, 0.063894] error=-0.006106 -> move up: search [8, 9]
  - iter 3: `inter-s64-k4` n=200 failures=68 rate=0.34 CI=[0.277915, 0.408115] error=0.147915 -> move down: ladder exhausted

### Band t5: target 5% ± 2%

- status: **converged**; frozen setting `inter-s32-k4` (calibration band-t5@ladder-gen-1:inter-s32-k4)
- search efficiency: 2 iterations, 400 probes, 801 API calls, 2075687 tokens, $0.4556, 715.668s
  - iter 1: `plain-s32-k2` n=200 failures=4 rate=0.02 CI=[0.007804, 0.050287] error=0.0 -> move up: search [5, 9]
  - iter 2: `inter-s32-k4` n=200 failures=9 rate=0.045 CI=[0.023853, 0.083297] error=0.0 -> converged: freeze this setting
- held-out validation (n=500): rate 0.05 CI=[0.034094, 0.072768]; point in band: True; CI intersects band: True; **pass: True**

## Ladder monotonicity (pooled observed rates)

Verdict: **monotone non-decreasing**

- `plain-s32-k2`: rate 0.02 [0.010168, 0.038963] (n=400)
- `inter-s32-k4`: rate 0.044444 [0.032807, 0.059954] (n=900)
- `inter-s64-k4`: rate 0.34 [0.277915, 0.408115] (n=200)
