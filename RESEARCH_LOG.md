# Research Log

Research decisions, not implementation notes. One entry per experiment:
the question, why it matters, the designs considered and rejected, the
threats to validity named in advance, the outcome, and the next question
the outcome actually raises. Every completed experiment also ends with
a "What would have changed my mind?" section — the outcomes that would
have reversed the interpretation, recorded pre-registered wherever the
experiment hasn't run yet. An experiment that couldn't have changed
any belief wasn't testing anything.

---

## Entry 1 — Real pilot: does team organization change real-model outcomes? (2026-10-28)

**Question.** With a real LLM backend (Claude Haiku 5.5) on generated
codebase-repair tasks, do independent, orchestrated, and verified team
strategies differ in correctness across team sizes 1–8?

**Why it matters.** The simulated lab showed clean scaling curves, but
simulated agents encode the conclusion in their competence model. A real
pilot is the first evidence about actual models.

**Alternative designs considered.** Full grid over all task families
(deferred — pilot first, one family); cheaper models (rejected —
tool-calling reliability is part of what is being tested).

**Threats to validity.** Generated single-defect tasks may be too easy;
three seeds give wide intervals; one model, one family.

**Outcome.** Complete ceiling effect: 36/36 runs at fraction correct
1.000 across every strategy and size, 0 failures, total spend $0.17.
The only signal was cost: orchestration cost ~+13% tokens for zero
quality gain. Task difficulty, not team organization, was the binding
variable.

**Next question.** Is the ceiling a property of the tasks or of the
measurement? Harden the tasks and rerun the same grid.

---

## Entry 2 — Hard mode: does defect subtlety break the ceiling? (2026-10-28)

**Question.** If defects are interacting, cross-module, or only visible
on edge/state probes, does correctness separate by strategy?

**Why it matters.** Distinguishes "tasks too easy" from "strategies
genuinely equivalent." Offline analysis confirmed the manipulation is
real: 55% of hard-mode modules are quiet (fail only on edge probes)
vs 0% in standard mode.

**Alternative designs considered.** More modules per run (rejected —
changes workload and difficulty simultaneously); weaker model (rejected —
confounds model capability with task difficulty).

**Threats to validity.** Agents carry a `run_check` tool inside each
attempt and can self-verify and iterate before submitting.

**Outcome.** Live smoke still saturated: fraction 1.000 everywhere. The
threat above was the explanation: internal self-verification makes final
quality saturate regardless of defect subtlety, and strategy-level
verification never sees a failure. The 36-run hard grid was not run —
it would have measured the same artifact at more length.

**Next question.** What happens when the internal check is removed, so
external verification is the only feedback? That became Phase A.

---

## Entry 3 — Phase A: when does external verification improve team performance? (2026-10-28)

**Question.** Under blind execution (no internal `run_check`), does
external verification improve correctness over matched orchestration,
and does the benefit vary with team size?

**Why it matters.** Verification is widely assumed to be how agent teams
scale. If its benefit is local insurance rather than a scaling effect,
that assumption is wrong in a specific, useful way.

**Design.** Blind mode in both real backends; config-driven experiment
(`experiment.yaml` → immutable results bundle with manifest, metrics,
traces, figures); pre-registered primary contrast verified −
orchestrated, both blind; hard tasks; sizes {1,2,4,8}; 5 seeds; 60 runs;
bootstrap CIs; deterministic replay of grading and cost accounting.

**Alternative designs considered.** Sighted control arm (rejected —
Phase A asks about verification under blind execution; the pilot already
showed sighted execution saturates). More seeds per cell (deferred —
pooled contrast is the citable number; per-cell intervals are wide by
construction and reported as such).

**Threats to validity (named in the bundle).** Verifier is the
ground-truth check itself (perfect verifier); one model; one task
family; agents execute sequentially, so team size varies organization,
not parallel speedup; each seed is a fresh task instance, so intervals
include task variance.

**Outcome.** Verified 0.975 vs orchestrated 0.900; primary contrast
+0.075 [95% CI +0.044, +0.106], excluding zero. The effect is flat
across team sizes 1–8. Coordination alone bought no quality, only ~+12%
tokens. Cost per correct subtask nearly unchanged. Interpretation:
verification acts as per-subtask insurance, not a scaling effect.
Total cost $0.341; replay verified exact.

**What would have changed my mind?** *(Recorded retrospectively for
this entry, from the pre-registered contrast; pre-registered in
advance for A-2 and all later entries.)*
- If the primary contrast's CI had crossed zero, I would have concluded
  verification has no measurable treatment effect under blind
  execution.
- If the effect had grown with team size, I would have rejected the
  local-insurance interpretation and treated verification as a scaling
  mechanism.
- If orchestrated had beaten independent on correctness, I would have
  had to credit coordination itself, not verification, as the active
  ingredient.

**Next question.** The flat effect is predicted if verification acts
locally on independent subtasks: width changes the amount of work, not
the mechanism. Under sequential dependency, errors accumulate and can
cascade — there, verification should matter more as chains lengthen.
Test the mechanism, not the variable: Phase A-2.

**Publication decision (2026-10-28).** Phase A is not published alone.
A and A-2 together tell the complete story: saturation → artifact
identified → artifact removed → effect demonstrated → mechanism
hypothesized → mechanism tested. Publishing after the mechanism test
gives readers the result and the explanation.

---

## Entry 4 — Phase A-2 design: error accumulation (2026-10-28, designed, not yet run)

**Question.** How does the benefit of external verification change as
error accumulation increases?

**Framing.** The scaling variable must change the mechanism, not just
the amount of work. Phase A scaled width (independent subtasks, local
errors, local verification, constant benefit — as observed). Phase A-2
scales sequential dependency: each step consumes the previous step's
output, so an unrecovered error propagates. Hypothesis: verification
prevents cascades, so its benefit grows with chain length. If the
benefit stays flat, the mechanism is falsified for this family and
model — a result, not a failure.

**Full design, metrics (survival probability by step, recovery-event
logging), and pre-registered falsification criterion:**
`experiments/error-accumulation/DESIGN.md`.

**Next question (sequenced after, deliberately).** Only once the
accumulation result is known: how should compute be allocated between
generation and verification (retry budgets 0/1/2/unlimited under equal
compute)? Asking it before A-2 would optimize a mechanism that may not
exist.

**What would have changed my mind?** *(Pre-registered, before any
A-2 data exists.)*
- If hazard rises similarly for both strategies, cascade prevention is
  not supported — verification is not intercepting errors before they
  propagate.
- If the verification gap stays flat with horizon, the local-insurance
  model survives and the accumulation mechanism is falsified for this
  family and model.
- If survival diverges but hazard does not, I would revisit whether
  hazard is the right mechanistic metric, or whether another
  explanation fits the data better, before claiming the mechanism.

**Amendment (2026-10-28, pre-implementation review).** Two additions
before any build. (1) An **offline calibration gate**: the strongest
alternative explanation for a horizon effect is "longer = simply
harder," so before any LLM runs, step-difficulty proxies (template
mix, operation count, state size, spec length, probe-visibility) must
be statistically indistinguishable across chain positions 1–50
(BH-corrected tests plus negligible effect sizes, with a
position-shuffle control). Only then can the experiment claim horizon
changes dependency structure, not intrinsic difficulty. (2) **Hazard
rate by step** added alongside survival: if verification prevents
cascades, orchestrated hazard should rise later in the chain while
verified hazard stays lower and flatter; hazard discriminates the two
mechanisms more cleanly than cumulative survival. Also recorded: the
recovery-event dataset is a standalone asset (recoverability by defect
class, errors that almost always cascade, late-surfacing failures,
wasted retries), and scope restraint is a deliberate strength — one
model, one family, no new task domains.

---

## Entry 5 — Phase A-2 run: the manipulation did not induce accumulation (2026-10-28)

**Execution.** Built to the ENGINEERING.md checklist (99 tests green),
calibration gate passed at full size (25,000 steps; every test
non-significant after BH, Cramér's V = 0.015, all η² = 0.0, shuffle
control null), fake-backend end-to-end dry run exact, tagged
`agentlab-a2-v1` (commit efcbbf7), code frozen, live grid run once
from the clean tree (manifest records git.dirty: false). One bug was
found by the fake dry run before the freeze (hazard figure divided by
a zero ymax when no run fails) and fixed pre-tag. No issues during
the live run itself. 50/50 runs completed, 0 failed, 0 skipped;
2,718,137 input / 387,679 output tokens; **total spend $0.4657 of the
$15 cap**. Replay verifies the finished bundle exactly, seal
included (1,029 artifacts + 14 derived files).

**Outcome.** A complete ceiling — the cleanest possible null. Across
870 steps (435 per strategy), there were **zero first-submission
failures**. Every cell at every horizon, both strategies: final state
correctness 1.000 [1.000, 1.000]. The external check fired 435 times
in verified runs and never found an error; zero retries in the whole
grid. Primary contrast 0.000 [0.000, 0.000] overall and at every
horizon; the pre-registered across contrast (effect at horizon 50 −
effect at horizon 2) is 0.000 [0.000, 0.000]; survival is flat at 1.0
for both arms, hazard is 0 at every step, and there is no divergence
point. Per-step failure probability is bounded above at ≈ 0.34%
(95% confidence, rule of three on 0/870).

**Interpretation, against the pre-registered criterion.** The
criterion's supported branch requires a growing effect — not met.
Its falsified branch ("the effect stays flat at ≈ +0.075 or shrinks")
presupposes the Phase A baseline effect and, more importantly, that
errors occur at all. Neither held. The honest verdict is narrower
than falsification: **the manipulation failed to induce the
phenomenon under study.** Stated precisely: the intervention failed
to induce the error regime required to identify the proposed
mechanism — the runs never entered the part of the state space where
error accumulation exists. ("The tasks were too easy" is the
imprecise version of this claim and is not used.) The design assumed a per-step failure rate
"small but nonzero"; for Haiku 5.5 on single transformations of a
12-integer state, the rate is effectively zero, so there was nothing
to accumulate and nothing for verification to prevent. The treatment
contrast is unidentified in this regime, not estimated at zero in a
regime where it could have been nonzero. Unlike the Entry 1 ceiling,
this one is not an evaluation artifact: agents were blind, grading
used first submissions, and replay verifies every number. It is a
task-difficulty finding about the model, measured airtight.

**What would have changed my mind?** (Pre-registered in Entry 4;
reality took none of the three branches, because their shared
antecedent — that errors occur — failed.)
- Had first-submission failures occurred and hazard risen equally in
  both arms, cascade prevention would have been rejected.
- Had failures occurred and the gap stayed flat, local insurance
  would have survived.
- Had survival diverged without hazard diverging, the metric would
  have been in question. With zero failures, all three questions are
  still open; the experiment answered a prior one instead: this step
  distribution is below this model's difficulty threshold.

**Next question.** Before accumulation can be studied, per-step
difficulty must be calibrated *to the model*: steps hard enough that
single-attempt failure is measurably nonzero (on the order of
2–10%), verified by a small live probe before a grid is committed —
composite steps (several transformations per step), larger states,
or transformations with interacting parts are the candidate levers.
That is a design decision for the next phase, not a patch to this
one: A-2's result stands as run, on frozen code, exactly as the
freeze rule intends. The recovery-event dataset from this run is
435 pass events and no failures — a baseline, not yet the cascade
dataset it was designed to become.

**Postscript (2026-10-28, review).** The "next question" above
generalizes into a named direction: **model-calibrated benchmark
generation.** Instead of asking "is this task hard?", the generator
asks "what is this model's first-attempt failure probability?" and
generates tasks to hit target failure bands (2%, 5%, 10%, 20%).
Difficulty stops being an intuition and becomes an experimentally
controlled variable, measured against the model under study before
any scaling hypothesis is tested. That is the candidate Phase A-3,
and it is a contribution in its own right, independent of what the
accumulation experiment eventually finds.

---

## Entry 6 — Phase A-3 design + publication positioning (2026-10-29)

**Decision: publication is held until A-3 completes.** A + A-2 alone
would most likely be summarized as "one positive result, then
another ceiling" — accurate, but not the story. With A-3, the arc
becomes a methodology: artifact-resistant evaluation (Phase A),
mechanism isolation with calibration gates (Phase A-2), and
model-calibrated benchmark generation (Phase A-3). The three publish
together as **Version 1.0**. Relatedly, the repository is renamed
**AgentLab** now rather than at publication (done on GitHub by
Saloni, 2026-10-29; local repo renamed to match), so the old name
accrues no further history.

**Phase A-3 question.** Can the generator produce task families
whose measured first-attempt failure rate on the target model lands
inside a requested band, validated on held-out seeds?

**Why it matters.** Every scaling study on agent systems inherits
its benchmark's difficulty as an unmeasured assumption; A-2 showed
that assumption silently deciding whether the studied phenomenon
exists at all. Calibrating difficulty to the model — as a number,
with intervals — is the precondition for the accumulation question
and, plausibly, for most empirical questions about agent teams.

**Scope, deliberately narrow.** A-3 demonstrates the calibration
process only: a difficulty ladder (composite steps, state size,
interacting parts), a blind single-attempt probe, a bounded search,
and held-out validation at two bands (10% ± 3%, 5% ± 2%). It does
**not** rerun the accumulation experiment on calibrated families —
that is the successor phase this one makes identifiable. Full
design, statistics, falsification conditions, and pre-registered
"What would have changed my mind?":
`experiments/model-calibrated-generation/DESIGN.md`.

**Threats to validity (named in advance).** Ladder non-monotonicity
would make failure rate uncontrollable via these knobs; search-seed
overfitting is controlled by held-out validation at fixed n; bands
2% and 20% are explicitly not claimed by this demonstration.

---

## Entry 7 — Phase A-3 run: one band calibrated, one band in the ladder's gap (2026-10-29)

**Execution.** Built to the ENGINEERING.md invariants (ladder
generator `ladder-gen-1`, a pure function of (seed, setting,
version); probe harness; bisection controller `model-calibration-1`;
120 tests green = 99 pre-existing + 21 new). The fake-backend dry
run of the shipped config exercised both controller paths
(convergence + validation for one synthetic band, declared failure
for the other) and replayed exactly. Tagged `agentlab-a3-v1`
(commit a1ad86e), code frozen, live probing run from the clean tree
(manifest records git.dirty: false). Bundle:
`results/model-calibrated-generation/20261029T071722Z-b4163d9b/`
— 1,500 probes, 2,971 API calls, 6,033,896 input / 2,961,216 output
tokens, **total spend $2.0840 of the $10 cap**, wall 3,136s. Replay
verifies the finished bundle exactly, seal included (4,507
artifacts). Zero probes in the completed run ended as infrastructure
errors.

**Incident (documented, not hidden).** The first live attempt
(run dir `20261029T071344Z-b4163d9b/`) was killed 116 probes into
its first batch when the runtime environment restarted and took the
background process with it; the batch never returned, so nothing was
recorded or sealed. That partial directory stands in place,
unsealed, as an aborted attempt. The experiment was re-run fresh
under a new run id from the same frozen tag — no code changed. The
aborted attempt's completed probes cost $0.0959 of real API spend
(and contained the run's only transport-error probes: 5, in one
brief window), so total A-3 API spend across attempts is $2.1799.
The completed run is otherwise exactly the frozen design.

**Outcome, per band.**

*5% ± 2% — CONVERGED, validated.* Controller trace (batches of 200):
rung 4 `plain-s32-k2` → 4 failures, rate 0.020 [0.008, 0.050], CI
intersects the band but the point estimate is below tolerance →
move up; rung 7 `inter-s32-k4` → 9 failures, rate 0.045 [0.024,
0.083] → converged in **2 iterations** (400 probes, 801 API calls,
2,075,687 tokens, $0.4556, 716s). Frozen as calibration
`band-t5@ladder-gen-1:inter-s32-k4`. Held-out validation on 500
fresh seeds: 25 failures, **rate 0.050 [0.034, 0.073]** — point
estimate inside the band and CI intersecting it: **PASS** under the
pre-registered criterion (validation cost: 500 probes, 1,000 API
calls, $0.7055, 1,042s). The generator now produces a step family
whose measured first-attempt failure rate for this model is 5%,
confirmed on seeds the search never touched.

*10% ± 3% — DECLARED FAILURE (no stable setting).* Trace: rung 4
`plain-s32-k2` → 4/200, rate 0.020 [0.008, 0.050] → up; rung 7
`inter-s32-k4` → 6/200, rate 0.030 [0.014, 0.064] → up; rung 8
`inter-s64-k4` → 68/200, rate 0.340 [0.278, 0.408] → down, ladder
interval exhausted. **3 iterations**, 600 probes, 1,170 API calls,
$0.9229, 1,378s. Per protocol, a non-converged band was not
validated. The tolerance was not widened. The band is not "missed
narrowly": it falls in a cliff gap in this ladder — between rung 7
(3%) and rung 8 (34%) there is no setting, and bisection over a
fixed ladder cannot invent one.

**Monotonicity.** Pooled observed rates over probed rungs are
monotone non-decreasing (rung 4: 0.020 [0.010, 0.039], n=400;
rung 7: 0.044 [0.033, 0.060], n=900; rung 8: 0.340 [0.278, 0.408],
n=200), with no inversions. Only 3 of the 10 rungs were visited —
bisection goes where the bands point — so monotonicity is assessed
on those rungs only. The dominant difficulty jump is the state-size
step 32 → 64 combined with interacting templates (3% → 34% across
one rung boundary); composite count and the interacting pool move
the rate gently by comparison. Any future ladder revision (a
mid-size rung, finer composite steps near the cliff) is a new
generator version and a new calibration, not a patch to this one.

**What would have changed my mind?** (Assessed against the items
pre-registered in the A-3 DESIGN.)
- *Non-monotone ladder* → did not occur on the probed rungs; the
  knobs do control failure rate in the direction claimed, though
  with very uneven granularity.
- *Calibrates on search seeds, misses on held-out seeds* → did not
  occur for the 5% band: search 0.045, validation 0.050. The probe
  is not overfitting seed noise at these batch sizes.
- *Both bands hit cleanly on the first ladder design → suspect a
  weak demonstration* → did not occur, in the strongest way: one
  band calibrated and validated, the other failed outright against
  a pre-registered failure class. The demonstration therefore shows
  both halves of the claim — the controller works when a setting in
  band exists, and it declares failure, cheaply ($0.92, 3 batches),
  when the ladder has none.

**Next question.** The accumulation experiment is now identifiable
in principle: the calibrated 5% family (or a future ladder revision
that can also reach 10%) supplies steps with a known, nonzero
first-attempt failure rate, which is the regime A-2 needed. Running
A-2's question on calibrated families is the successor phase, as
sequenced. Separately, ladder granularity near the 32→64 cliff is
the clearest generator improvement A-3 produced evidence for.

---

## Entry 8 — Version 1.0 packaging decision (2026-10-29)

**Decision: package Version 1.0 now, with Phase A-3 as-is.** No
ladder refinement first. The publication package is assembled
locally (methodology README, published result summaries in
`results-published/`, this entry); the push itself still waits for
Saloni's final go on the exact outward text.

**Why not refine the ladder first.** A-3's question was never "can
we hit every requested failure band?" It was: can the generator
produce families whose measured first-attempt failure rate lands
inside a requested band and validates on held-out seeds? That
question is answered. The 5% band converged and validated; the 10%
band produced a declared failure against a pre-registered class —
the calibration procedure correctly diagnosed that the current
generator family cannot represent that operating point. The failure
is a statement about the generator (its ladder resolution), not
about the calibration algorithm, and it is exactly the failure mode
a calibration methodology should expose: an unsupported regime
detected and reported, not silently approximated. Inserting a
mid-size rung and hitting 10% would teach one thing — the ladder
needed another rung. That is engineering optimization, not a change
to the scientific contribution.

**The v1.0 claims, as packaged.** (1) Artifact-resistant
evaluation: evaluation artifacts can manufacture conclusions, and
removing one (internal self-verification) exposed a real, measured
effect. (2) Calibration gates: a generator was rejected before any
scientific conclusion was drawn from it, for violating a
pre-registered validity criterion. (3) Model-calibrated benchmark
generation: difficulty calibrated to the model when the generator
supports the requested regime, and explicit declaration when it
does not.

**Sequencing recorded.** Version 1.0 establishes calibration
*correctness*. Version 1.1 improves calibration *coverage* (finer
ladder resolution). The scientific questions — error accumulation
on calibrated families, communication bandwidth, compute
allocation — come after v1.0, on calibrated benchmarks.

**The closing claim of v1.0**, which the README ends on and which
this log records as the project's philosophical position:

> AgentLab does not guarantee that every benchmark can be calibrated. It guarantees that unsupported operating regimes are detected and reported rather than silently approximated.
