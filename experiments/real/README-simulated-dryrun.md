# dry run of the real pilot path (fake backend only)

Everything in this directory from the first dry run was produced with the
**fake backend** (`teamscale/fake_backend.py`) -- no network calls, no API
key, $0 spent. It exercises the exact code path the real pilot uses (real
task generation, real sandboxes, real check execution, the pilot runner,
traces, and the spend ledger) with a scripted stand-in for the model, so
the mechanics could be verified end to end before any real run.

Command:

```bash
.venv/bin/python -m teamscale.real_pilot --fake \
    --sizes 1,2 --strategies orchestrated,verified --seeds 0 \
    --out experiments/real
```

Results (fake competence 0.85, seed 0):

| run | fraction correct | tokens | retries |
|---|---|---|---|
| orchestrated, team of 1 | 0.625 | 19,506 | 0 |
| orchestrated, team of 2 | 0.875 | 19,510 | 0 |
| verified, team of 1 | 0.875 | 26,874 | 3 |
| verified, team of 2 | 1.000 | 21,957 | 1 |

Mean fraction correct 0.844 across the 4 runs. The shape is the point of
the dry run: verification plus one feedback retry visibly buys correctness
at extra token cost, on real executed checks -- the same contrast the real
pilot measures with a live model. Token counts here are the fake backend's
synthetic accounting (same rules as the real backend: per-call usage
events that sum exactly to the run totals, verified by the test suite).

Artifacts: `results/*.json` (per-run), `traces/*.jsonl` (tool calls and
usage only), `pilot_summary.json`, `ledger_fake.json` (the spend ledger,
kept separate from the real pilot's `ledger.json`).

One real bug this dry-run path caught during development: a fix written in
the same second and at the same byte length as the broken file can be
shadowed by stale `__pycache__` bytecode, so checks now clear the sandbox
bytecode cache and run with `-B`. Without that, the runner would grade the
old file instead of the agent's edit.
