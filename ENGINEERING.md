# Engineering Invariants

The standard for this project is no longer "does the experiment run."
It is: **could another researcher build directly on this without
talking to us?** Every component written for Phase A-2 — and inherited
by A-3 and beyond — is held to the four invariants below. This file
also records an honest audit of the Phase A platform against them.

## Invariant 1 — Generator determinism

A generated task instance (repair task, chain, any future family) is
entirely reproducible from **(seed, generator_version)**. Nothing else.

- No hidden randomness: every random draw flows from a seeded RNG
  created inside the generator. No global/module-level RNG state.
- No timestamps, hostnames, environment values, or dict-ordering
  accidents in generated content.
- No mutable templates: template definitions are versioned constants;
  changing a template increments `generator_version`.
- Test: generating the same (seed, version) twice, in separate
  processes, produces byte-identical artifacts; a regression test pins
  hashes for a fixed seed set per version.

## Invariant 2 — Immutable artifacts

Every experiment run directory is write-once. Canonical layout:

```
run/
    config.yaml
    manifest.json
    calibration.json        (when a calibration gate applies)
    traces.jsonl
    metrics.csv
    figures/
    hashes.json
```

- The directory is created manifest-first; reusing a run id is an
  error (already true since Phase A).
- On completion, `hashes.json` records the SHA-256 of every other file
  in the directory. After that, the directory is treated as read-only:
  any tool that would modify it must refuse, and replay verifies the
  hashes before trusting the contents.
- If anything about a run must change, it becomes a new run id. There
  are no edits in place, ever.

## Invariant 3 — First-class provenance

Every output answers "exactly what generated me?" from inside the run
directory alone. The manifest must carry:

- model id (exact, as served), provider
- prompt hash (SHA-256 of the full system+user prompt templates used)
- generator version, calibration version (when applicable)
- task-family and difficulty parameters
- config hash, git commit **and dirty flag**
- replay version, analysis version, code version
- pricing table used for cost accounting
- host basics (OS, Python) — Phase A already records these

A dirty git tree at run time is recorded, not hidden — but a
publishable run should be made from a clean tree, and the summary
should say so when it wasn't.

## Invariant 4 — Analysis purity

The analysis stage contains **no experiment logic**.

- It consumes only `metrics.csv` (and the manifest for labels, plus
  replay outputs where applicable). It never reads traces, sandboxes,
  or generator internals.
- Grouping variables, contrasts, and figure axes are **declared in the
  config/manifest**, not hardcoded in the analysis code. The analysis
  is generic machinery: given declared columns and contrasts, it
  produces CIs, contrasts, and figures.
- It is deterministic: the bootstrap RNG is seeded from the
  `metrics.csv` bytes, so re-running analysis on the same run
  directory regenerates byte-identical outputs (already true since
  Phase A — keep it).

Rationale: if the analysis "knows" about horizons or verification,
every new experiment forks the analysis, and reproducibility becomes
harder to reason about with each fork.

---

## Audit of the Phase A platform against these invariants

Checked against the code and the Phase A run bundle
(`results/verification-question/20261029T030319Z-849d8f77/`), not from
memory:

| Invariant | Status | Evidence |
|---|---|---|
| 1. Generator determinism | **Partial** | Repair-task generation is seeded (standard-mode tasks verified byte-identical across modes for fixed seeds). But no `generator_version` exists or is recorded, and no cross-process byte-identity test pins it. |
| 2. Immutable artifacts | **Partial** | Run dirs are manifest-first and refuse overwrite (FileExistsError). But there is no `hashes.json`, no post-completion read-only enforcement, and replay does not verify artifact hashes. |
| 3. Provenance | **Mostly holds, with gaps** | Manifest records config + config SHA-256, git commit, code version, model, provider, pricing, difficulty, grid, host. Missing: prompt hash, generator version, calibration version, replay/analysis versions. The Phase A run also honestly records `git.dirty: true` — provenance worked; the discipline of running from a clean tree did not. Publishable A-2 runs must be clean-tree. |
| 4. Analysis purity | **Partial** | `analyze.py` reads only `metrics.csv` (+ manifest/summary) and is deterministic (RNG seeded from metrics bytes). But the primary contrast (verified − orchestrated) and the figure axes (team size) are hardcoded Phase A logic inside the analysis. A-2 must move contrasts and axes into the config and make the analysis generic — otherwise the A-2 analysis becomes a fork, which is exactly what this invariant exists to prevent. |

## What this means for the A-2 build (acceptance checklist)

- [x] Chained generator takes (seed, generator_version); byte-identity
      regression test across processes.
- [x] Manifest gains: generator_version, prompt hash, calibration
      version, replay version, analysis version.
- [x] `hashes.json` written at run completion; replay verifies hashes.
- [x] Contrasts/axes declared in `experiment.yaml`; analysis refactored
      to be generic over them (Phase A's declared contrast must
      reproduce Phase A's published numbers exactly as a regression
      check).
- [x] Calibration runs through the same manifest/hash discipline
      (calibration.json inside the run bundle).
- [x] Live A-2 grid runs from a clean git tree. *(Completed
      2026-10-28: run 20261029T063426Z-1a68400f from tag
      agentlab-a2-v1, manifest git.dirty: false, 50/50 runs, replay
      exact including the seal. The scientific outcome — a complete
      ceiling, zero first-submission failures in 870 steps — is
      recorded in RESEARCH_LOG.md, Entry 5.)*

Status at the A-2 release tag: the first five items are green,
verified by the test suite (99 tests), the calibration gate report
(`experiments/error-accumulation/calibration.json`, gate: pass), a
fake-backend end-to-end run of the shipped config with exact replay,
and the Phase A regression (the generic analysis, run on a copy of
the Phase A bundle under Phase A's declared contrast, reproduces
verified 0.975 / orchestrated 0.900 and the primary contrast +0.075
[+0.044, +0.106] exactly). One development note: the first calibration
probe failed on probe-visibility drift -- contractive transformations
(absolute differences, clamps) concentrate the state distribution as
a chain progresses, making defects less visible at later positions.
The shipped template set is restricted to distribution-preserving
transformations for exactly this reason (see teamscale/chains.py).
