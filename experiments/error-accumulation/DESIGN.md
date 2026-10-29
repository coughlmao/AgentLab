# Phase A-2 — Error Accumulation

**Status:** designed and pre-registered in this document. Build and
live run authorized 2026-10-28 under a fresh $15 cap. Local only;
publication is held until this experiment completes (see
RESEARCH_LOG.md, Entry 3).

## Release-gate pipeline and freeze rule

The calibration gate is a **release gate**, not a pre-flight check:

```
build → engineering checklist green (ENGINEERING.md)
      → calibration gate passes
      → tag release
      → clean checkout (git status clean)
      → run experiment → replay → analysis
```

The experiment should never be the thing that tells us the
infrastructure is broken.

**Freeze rule.** Once the calibration gate passes, the code is frozen
for the run: no bug fixes, no refactors, no prompt tweaks. If an issue
is discovered during the live run, the run finishes, the issue is
documented in the run bundle and the research log, and any fix lands
afterward under a new version — with a rerun under that version if the
result requires it. This preserves the integrity of the version that
produced the published result.

## Question

How does the benefit of external verification change as error
accumulation increases?

**Milestone.** Phase A demonstrated that external verification has a
measurable treatment effect. Phase A-2 aims to demonstrate that error
accumulation changes — **or does not change** — the magnitude of that
treatment effect. Either outcome answers a mechanistic question.

## Mechanism, not variable

Phase A scaled **width**: more independent subtasks. The verification
benefit was constant (+0.075 across team sizes 1–8), which is what a
local mechanism predicts — verification acts on each subtask
independently, so adding subtasks adds work, not mechanism.

This experiment scales **sequential dependency**. A chain of steps in
which each step consumes the previous step's output changes what an
error *does*: it propagates. That gives verification a different job —
not insuring isolated subtasks, but preventing cascades.

Two competing predictions, stated in advance:

- **Local-insurance model (Phase A extrapolation).** Verification
  benefit is a constant per step checked; the final-outcome gap stays
  ≈ +0.075 at every chain length.
- **Cascade-prevention model.** Unrecovered errors compound, so
  unverified chains decay with length while verified chains decay
  slower; the gap **grows** with horizon.

Horizon (chain length) is the manipulation used to induce accumulation.
It is not the concept under study.

## Pre-registered falsification criterion

The cascade-prevention mechanism is **supported** if the verification
effect on final state correctness at horizon 50 exceeds the effect at
horizon 2 with the bootstrap CI on that difference excluding zero, and
the survival curves (below) diverge progressively rather than at a
single step.

It is **falsified** (for this task family and model) if the effect
stays flat at ≈ +0.075 or shrinks as horizon grows. A flat result is a
publishable answer: it would mean verification's benefit is local even
when errors can propagate.

## Task family: chained transformations (new generator required)

The existing repair family is parallel subtasks and cannot express
horizon. A-2 requires a new generator:

- An instance is a chain of N steps. Step k's specification is defined
  relative to step k−1's output artifact, so the correct input to step
  k exists only if step k−1 ended in the correct state.
- Every step is graded by an executable check against **ground-truth
  state** (what the state after step k should be), independent of the
  path taken. This makes per-step state correctness measurable even
  after an upstream error.
- Steps are drawn from the hard-difficulty defect distribution in
  spirit: a competent single attempt succeeds most of the time, so
  per-step failure probability is small but nonzero — the regime where
  accumulation is visible over 50 steps rather than immediate collapse.
- Execution is sequential by construction. Team size is fixed at 4 and
  varies only which agent takes each step (rotation), not parallelism.
  Stated plainly: this experiment studies dependency, not speedup.

## Pre-experiment gate: offline difficulty calibration (no LLM runs)

The strongest alternative explanation for any horizon effect is not
accumulation but difficulty: longer chains might simply contain harder
steps. Per-step difficulty being constant *by construction* is not
enough — it must be shown. **No live runs happen until this gate
passes.**

1. Generate 500 chains × 50 steps offline from the generator (25,000
   steps, zero model calls). Steps are sampled independently of
   position; the calibration verifies that this is true in practice,
   not just in intent.
2. For every step, record difficulty proxies: template/type, operation
   count, state size, spec length, check-probe count, and a
   probe-visibility score (fraction of the step's check probes failed
   by canonical single-defect mutants of that step — the same
   visibility analysis used to validate hard mode).
3. Test position-independence across position deciles (steps 1–5,
   6–10, …, 46–50): chi-square for template mix, Kruskal–Wallis for
   each continuous proxy.
4. **Pre-registered pass criterion:** no test significant after
   Benjamini–Hochberg correction at q = 0.05, **and** every effect
   size negligible (Cramér's V < 0.10, η² < 0.01). A position-shuffle
   control (permuted position labels) must also return null, as a
   sanity check on the testing pipeline itself.
5. If the gate fails, the generator's sampling is fixed to be
   position-independent, chains are regenerated, and calibration
   reruns. The calibration output (`calibration.json`) ships inside
   the experiment bundle, so the claim is checkable by a reader:

> Horizon changes dependency structure, not intrinsic task difficulty.

## Conditions

- Strategies: **orchestrated** vs **verified**, both blind (no internal
  `run_check`), identical prompts and coordination; the only difference
  is the external check after each step submission plus one retry on
  failure — the same contrast as Phase A, moved to a chained family.
- Horizons: {2, 5, 10, 20, 50}.
- Seeds: {0, 1, 2, 3, 4}. Each seed is a fresh chain instance.
- Model: claude-haiku-5-5. Difficulty: hard. Per-run cap 400K tokens.
- Grid: 5 horizons × 2 strategies × 5 seeds = **50 runs**, 870 total
  step-executions.
- Cost estimate: ≈ $1–2 at Phase A token rates (steps are smaller than
  repair subtasks; the horizon-50 cells dominate). Well under a fresh
  $15 cap. Runtime, not cost, is the binding constraint: horizon-50
  runs are long and sequential; expect hours-scale wall time for the
  full grid.

## Metrics

1. **Survival probability by step (primary).** For each horizon and
   strategy, P(state correct after step k) for k = 1…N: the fraction of
   runs whose state after step k matches ground truth. Reported as
   curves with bootstrap CIs, plus the **divergence point** — the first
   k where the verified−orchestrated survival gap's CI excludes zero.
   Two correctness notions are recorded per step: *state correctness*
   (vs ground truth — defines survival) and *local correctness* (step
   correct given the input it actually received — separates "agent
   erred" from "agent inherited a broken state").
2. **Hazard rate by step.** h(k) = P(chain first fails at step k |
   survived through step k−1), computed per strategy from the
   first-failure step distribution, with bootstrap CIs. The cascade
   model makes a sharper prediction here than for cumulative survival:
   orchestrated hazard should **rise** later in the chain (each
   surviving step carries more accumulated state that can be wrong in
   ways that only surface downstream), while verified hazard stays
   lower and flatter because errors are caught at the step that made
   them. If hazard is flat and equal for both strategies while
   survival still separates, the effect is step-local insurance, not
   cascade prevention — hazard discriminates the two mechanisms in a
   way the endpoint cannot.
3. **Final state correctness** by horizon (the Phase A-comparable
   endpoint; secondary here).
4. **Verification effect by horizon**: verified − orchestrated final
   correctness at each horizon, with the horizon-50 minus horizon-2
   difference as the pre-registered primary contrast.
5. **Cost**: tokens and dollars per run and per surviving step.

## Recovery-event logging (the error-cascade dataset)

Every external check that fires is logged as an event, pass or fail:

| field | meaning |
|---|---|
| run_id, seed, horizon, strategy | provenance |
| step_index | position in the chain |
| check_outcome | pass / fail on first submission |
| error_class | from the check output (assertion, type, contract, state) |
| would_cascade | whether downstream steps consume this step's state (true by construction except where a later step re-derives its input; recorded from the generator's dependency map) |
| retry_outcome | recovered / still failing after the single retry |
| chain_restored | whether state after the step matches ground truth |
| recovery_tokens | tokens spent on the retry |

This dataset outlives the experiment: it records where errors occur,
which ones would have propagated, and whether verification actually
restored the chain — the raw material for studying cascades directly.
It is treated as a standalone research asset, not a by-product: it can
later support questions this experiment does not ask — which defect
classes are most recoverable, which errors almost always cascade,
which failures stay invisible until much later, and which retries are
wasted spend.

## Threats to validity (named in advance)

- Perfect verifier again: the external check is the ground-truth check.
  A noisy verifier is a later question, deliberately out of scope.
- One model, one (new) task family. The generator is new, so A-2
  results are not a pure extension of Phase A's family; the comparison
  that matters is *within* A-2 (effect vs horizon), not A vs A-2
  levels.
- Per-step difficulty must be roughly constant across positions in the
  chain; if later steps are intrinsically harder, horizon confounds
  accumulation with difficulty. This is handled by the
  pre-experiment calibration gate above: difficulty proxies must be
  statistically indistinguishable across positions before any live
  run, and the calibration ships with the results.
- Survival is an absorbing-state measure: once a chain's state is
  wrong, later "correct" states require the agent to re-derive the
  right state from a wrong input. chain_restored and local correctness
  are logged precisely so this is visible rather than hidden.
- Five seeds per cell: per-horizon CIs will be wide; the pooled
  horizon-trend contrast is the citable number, as in Phase A.

## Alternatives considered and rejected

- **Frame the study around "task horizon"**: rejected. Horizon is a
  manipulation; error accumulation is the mechanism. Framing around
  the manipulation would invite "longer tasks are harder" as a
  complete explanation and hide the falsifiable claim.
- **Run the retry-budget sweep next instead** (0/1/2/unlimited retries
  under equal compute): deferred, not rejected. It asks how to
  allocate compute between generation and verification — worth asking
  only once we know whether verification's value scales with
  accumulation. Sequenced as its own phase, one question at a time.
- **More width, more seeds on the Phase A family**: rejected. Width
  does not change the mechanism; Phase A's pooled contrast already
  answers the width question at this precision.

## Caveats placement

Following the Phase A review: the caveats (perfect verifier, one
model, one family, sequential execution, wide per-cell intervals) go at
the **top** of the results summary for this experiment, not the bottom.
