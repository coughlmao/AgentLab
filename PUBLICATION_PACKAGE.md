# Publication Package — AgentLab Version 1.0 (ASSEMBLED, NOT pushed)

Prepared 2026-10-28; **assembled 2026-10-29**. Nothing here has been
pushed. The push happens only on Saloni's explicit go on the exact
outward text.

**Status: ASSEMBLED.** Saloni decided 2026-10-29 to package v1.0
now, with Phase A-3 as-is (see RESEARCH_LOG.md, Entry 8). The
README was revised the same day per her presentation review
(compressed opening, pipeline figure, reordered sections,
contributions kept conceptual, CLI collapsed) and the `v1.0` tag
moved to the revision commit. The
packaging commit — "agentlab v1.0: publication package — methodology
readme, published result summaries, research log entry 8" — is
tagged **`v1.0`** locally. It contains:

- The rewritten top-level `README.md`: AgentLab identity, the three
  methodological contributions in order (artifact-resistant
  evaluation; calibration gates and mechanistic benchmarking;
  model-calibrated benchmark generation, with the 10% band framed as
  the calibration procedure diagnosing an operating point the
  generator family cannot represent), the four engineering
  invariants, a Results section linking the published summaries, the
  roadmap (v1.0 = calibration correctness; v1.1 = calibration
  coverage; the scientific questions after v1.0, on calibrated
  benchmarks), and the closing guarantee sentence as the final line.
- `results-published/` — summary-form artifacts of the three v1.0
  run bundles, one directory per phase:
  - `phase-a/`: SUMMARY.md, analysis.json, manifest.json,
    figures/{scaling_curve, cost_breakdown}.svg
  - `phase-a2/`: SUMMARY.md, analysis.json, manifest.json,
    calibration.json, figures/{scaling_curve, cost_breakdown,
    survival_{2,5,10,20,50}, hazard_{2,5,10,20,50}}.svg
  - `phase-a3/`: SUMMARY.md, calibration.json, manifest.json,
    summary.json (its bundle has no analysis.json and no figures)
  - `results-published/README.md` explaining what these are and
    that the full bundles are reproducible from the repo's code and
    configs. No traces, sandboxes, raw per-run outputs, or ledgers
    were copied.
- `RESEARCH_LOG.md` Entry 8, recording the packaging decision and
  its reasoning.
- This file, updated.

**The rename is done**: Saloni renamed the GitHub repo to
`saloni-garg/AgentLab` on 2026-10-29 (verified: public, still at the
old commit 8bf0da6c with the pre-Phase-A README, as expected while
publication is held), and the local repo directory is
`~/workspace/AgentLab`.

**Remaining steps, in order:**
(a) Saloni's approval of the README text (the exact outward text).
(b) The push of all local commits through the `v1.0` tag commit to
    `saloni-garg/AgentLab`.
(c) Live verification: read the published repo back and confirm the
    README, results-published/, and tags landed as intended.

The sections below are the pre-assembly plan, kept for the record;
where they differ from the assembled package above (e.g. they
predate Phase A-3 and the rename), the sections above govern.

## The story (methodological contribution, not a benchmark result)

1. We observed a ceiling (pilot: everything saturated).
2. We identified an evaluation artifact (internal self-verification).
3. We removed it and measured a real effect (Phase A: +0.075
   [+0.044, +0.106]).
4. We hypothesized a mechanism (error accumulation under sequential
   dependency).
5. We designed a benchmark to isolate that mechanism (Phase A-2),
   with a calibration gate that rejected the first generator design
   before it could produce a misleading experiment.
6. The mechanism could not be observed: the intervention failed to
   induce the error regime required to identify it (0 failures in
   870 steps, replay-verified).
7. Conclusion: benchmarks for scaling hypotheses must be calibrated
   to the model — which becomes the next contribution
   (model-calibrated benchmark generation).

## What publishing means concretely

Current public repo: `saloni-garg/optimized-agent-team-scaling`,
frozen at the pre-real-backend state (commit 8bf0da6c era). The local
repo is ahead by the full arc: real backends, hard mode, Phase A
platform + AgentLab README, ENGINEERING.md, RESEARCH_LOG.md, the
A-2 build at tag `agentlab-a2-v1`, and the post-run log commit.

A publication push would include:

- All local commits through the A-2 documentation commit (code,
  tests, experiment configs, DESIGN.md, ENGINEERING.md,
  RESEARCH_LOG.md, README).
- The two headline results bundles **in summary form**: each bundle's
  SUMMARY.md, analysis.json, figures, manifest, and calibration.json
  (A-2) copied into a tracked `results-published/` directory. Raw
  traces/sandboxes stay local — they are large, and the summaries plus
  deterministic replay from the published code are the reproducibility
  path. (Changing this is a one-line decision.)
- A README section added *before* the push, in the project's own
  research framing (no job/hiring references, per standing rule):
  Phase A result, Phase A-2 result with the precise framing, the
  calibration-gate failure story, and the model-calibrated generation
  direction. Draft text is shown to Saloni for exact approval before
  anything is pushed — outward text is never published unreviewed.

## The rename decision (bundled with the push, or not)

Options: rename the repo to match the AgentLab identity (GitHub
redirects the old name), or publish under the existing name
`optimized-agent-team-scaling`. The rename is a repo-settings change
the snapshot-push connector cannot perform; it needs a browser step
or Saloni doing it in GitHub settings (30 seconds). Code/docs push
works either way.

## Not included, deliberately

- No resume/website changes (separate decision, previously offered,
  never approved).
- No raw API spend beyond what the bundles already record.
- No new experiments. Publication is of what exists.
