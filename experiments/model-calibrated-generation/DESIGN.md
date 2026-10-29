# Phase A-3 — Model-Calibrated Benchmark Generation

**Status:** designed. Not yet built or run. Local only; publication of
A + A-2 + A-3 together as Version 1.0 happens after this phase
completes (see PUBLICATION_PACKAGE.md).

## Question

Can the generator produce task families whose **measured
first-attempt failure rate on the target model** lands inside a
requested band — and stay there on held-out seeds?

## Why this phase exists

Phase A-2's verdict was precise: the intervention failed to induce
the error regime required to identify the proposed mechanism. The
deeper lesson generalizes beyond that experiment:

> Benchmarks should be calibrated to the model before they are used
> to study scaling.

Today, "difficulty" in agent benchmarks is an intuition asserted by
the benchmark author. A-3 makes it a **measured, controlled
variable**: the generator's target is not "hard tasks" but a number —
the target model's first-attempt failure probability — and a family
is only "at 10%" when probing says so, with a confidence interval,
on seeds the calibration never touched.

Scope discipline: **A-3 demonstrates that the calibration process
works. It does not rerun the accumulation experiment** on calibrated
families; that is the phase this one enables, not this one.

## The difficulty ladder

Calibration needs knobs that move failure rate smoothly and
monotonically. Candidate levers (from Research Log Entry 5), combined
into an ordered ladder of generator settings:

1. **Composite steps** — k transformations per step (k = 1…4+).
   More operations per attempt, more places to err.
2. **State size** — 12 → 32 → 64 elements. More state to track
   exactly through each transformation.
3. **Interacting parts** — transformations whose parameters depend on
   the current state (conditional operations, data-dependent
   indices), so a step cannot be executed by pattern alone.

Monotonicity is a claim to be *verified by the probe data*, not
assumed: if observed failure rate is not non-decreasing along the
ladder, the ladder is wrong and that is reported as a finding.

## Probe protocol

- **Unit:** an individual step, standalone — not a chain. A-3
  calibrates the step distribution; chains are downstream.
- **Conditions:** blind, single agent, single attempt, no check
  feedback of any kind. Failure = first-submission state incorrect
  under the same executable ground-truth grading as A-2.
- **Model:** claude-haiku-5-5, unchanged. One model, deliberately —
  the claim is "calibrated *to a* model," and a second model would
  test portability, a different question.
- **Batches:** 200 fresh steps per ladder setting per round.

## Calibration loop — a closed-loop controller, not a search

Calibration is structured as a feedback controller whose setpoint is
the target band:

```
target band → generate candidates → probe first-attempt failure rate
→ estimate error from target → adjust generator parameters → repeat
until convergence or declared failure
```

1. Start at the ladder setting nearest the target band (A-2's base
   setting is known: ≈ 0% failure, the floor).
2. Probe a batch; compute the failure rate with a Wilson 95% CI.
   The signed distance from the band is the controller's error
   signal.
3. If the CI sits below the band, move up the ladder; above it, move
   down, with step size shrinking as the error shrinks (bisection is
   the initial control law). Bounded rounds (max 6 settings probed
   per band).
4. **Convergence:** a setting's CI is consistent with the band at the
   pre-registered tolerance. Freeze that setting as a named,
   versioned calibration (`calibration_version` in the manifest,
   per the engineering invariants). Exhausting the bounded rounds
   without convergence is a *declared failure* of the controller for
   that band, reported as such — not a prompt to widen the tolerance.

For each band, the controller itself is measured and reported:

- iterations to convergence (or failure),
- probes consumed, API calls, tokens, wall-clock time,
- final achieved failure rate with its CI,
- the full control trace (setting, rate, CI, error, adjustment) in
  `calibration.json`.

**Calibration efficiency** — probes / tokens / wall-clock per band
hit — is a first-class result, not bookkeeping: calibration is an
optimization problem, and if two control laws reach the same band,
the cheaper one is the better algorithm. Efficiency numbers are what
let a future phase (or another researcher) compare controllers.

## Demonstration targets (pre-registered)

The framework accepts any band; the *demonstration* is deliberately
narrow — two bands, sized honestly:

| band | tolerance | validation n | Wilson half-width at that n |
|---|---|---|---|
| 10% | ±3% | 500 held-out steps | ≈ ±2.6% |
| 5% | ±2% | 500 held-out steps | ≈ ±1.9% |

**Success criterion, per band:** on held-out seeds (never used during
the search), the point estimate falls inside the band tolerance AND
the Wilson CI intersects the band. Both numbers reported regardless.

The 2% and 20% bands from the general proposal are **not** claimed
by this phase: 2% may sit below the ladder's first rung above the
A-2 floor, and 20% may need ladder extensions. Claiming only what
the demonstration covers is the point of the exercise.

**Falsification:** calibration fails for a band if the ladder
saturates below it (max setting still under the band), overshoots it
at the minimum setting, or is non-monotone so no setting is stable.
Any of these is a reportable result about the generator, not a
license to quietly widen the tolerance.

## Budget estimate

Probes are single steps (≈ 3–4K tokens each at A-2 rates, ≈ $0.0006
per probe). Search ≈ 6 settings × 200 × 2 bands ≈ 2,400 probes;
validation 1,000 probes. Total ≈ 3,400 probes ≈ **$2–3**, plus
margin — a fresh **$10 cap** is ample. As before, the release-gate
pipeline applies unchanged: checklist → offline checks → tag →
clean tree → frozen probe runs → replay/analysis. Probe runs are
experiments under the invariants, not scripts.

## Deliverables

- A reusable calibration harness (probe runner + ladder search) that
  any future phase — and any other researcher — can point at a new
  family or band.
- `calibration.json` per band: ladder setting, generator +
  calibration versions, probe counts, the controller's full trace
  (every setting probed, rate, CI, error, adjustment), efficiency
  totals (probes, API calls, tokens, wall-clock, iterations),
  validation result on held-out seeds.
- Two validated families (5% and 10%) as generated artifacts with
  full provenance.
- Research Log entry with outcome and a "What would have changed my
  mind?" section.

## What would have changed my mind? (pre-registered)

- If the ladder is non-monotone, failure rate is not a controllable
  function of these knobs, and "calibration" over them is illusory —
  the knobs, not the concept, would need rethinking first.
- If a setting calibrates on search seeds but misses on held-out
  seeds, the probe is overfitting seed noise and the batch sizes —
  not the tolerance — are what must change.
- If both bands hit cleanly on the first ladder design, I would
  suspect the bands were too easy to hit and say so, rather than
  generalize from a weak demonstration.

## Explicitly out of scope

- Rerunning the error-accumulation experiment on calibrated families
  (the successor phase; A-3 exists to make it identifiable).
- Multiple models, new task domains, new strategies.
- The retry-budget / compute-allocation question (still sequenced
  after the accumulation result, per the standing doctrine).
- Publication mechanics — prepared in PUBLICATION_PACKAGE.md, held
  until A-3 completes, then A + A-2 + A-3 publish together as
  Version 1.0.
