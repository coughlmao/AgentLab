# Published result summaries

These are the published summary artifacts of the three Version 1.0
experiments. Each phase directory holds the summary form of one
sealed run bundle:

- `SUMMARY.md` — the experiment's question, design, headline numbers,
  and caveats, written by the analysis step at run time.
- `analysis.json` — every number in the summary, structured
  (Phases A and A-2; Phase A-3's analysis lives in its calibration
  report).
- `manifest.json` — the run's full provenance: config and its hash,
  git commit, model, pricing, versions (generator, calibration,
  replay, analysis), and prompt hash where recorded.
- `calibration.json` — the calibration evidence: for Phase A-2, the
  offline gate report (step-difficulty indistinguishable across chain
  positions); for Phase A-3, the controller's per-band traces,
  efficiency totals, and held-out validation verdicts.
- `figures/` — the plots, regenerated from the recorded metrics.

| directory | experiment | run id |
|---|---|---|
| `phase-a/` | When does external verification improve team performance? | 20261029T030319Z-849d8f77 |
| `phase-a2/` | How does the benefit of external verification change as error accumulation increases? | 20261029T063426Z-1a68400f |
| `phase-a3/` | Can the generator produce families whose measured first-attempt failure rate lands in a requested band, on held-out seeds? | 20261029T071722Z-b4163d9b |

The full bundles — traces, sandboxes, per-run outputs, spend
ledgers — are not copied here. They are reproducible from the code
and experiment configs in this repository: every run directory is
generated deterministically from its config, sealed with
`hashes.json`, and verifiable by replay (`teamscale.replay`), which
re-grades the recorded artifacts and re-derives the accounting
without re-running the model.

Each `SUMMARY.md` carries its own caveats section, and those caveats
stand: they are part of the result, not fine print.
