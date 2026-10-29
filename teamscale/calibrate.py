"""The Phase A-2 calibration gate: offline difficulty calibration.

    python -m teamscale.calibrate [--chains 500] [--steps 50] \
        [--out experiments/error-accumulation/calibration.json]

Before any LLM run, this gate generates chains entirely offline and
tests whether step difficulty is statistically indistinguishable
across chain positions -- the evidence behind the claim "horizon
changes dependency structure, not intrinsic task difficulty"
(experiments/error-accumulation/DESIGN.md).

Per step, the difficulty proxies are: template (categorical), op_count,
state_size, spec_len (characters of the step spec), probe_count, and
probe_visibility (fraction of the step's check probes failed by the
canonical single-defect mutant of that step, applied to its
ground-truth input). Steps are grouped into position deciles
(1-5, 6-10, ..., 46-50). Template mix gets a chi-square test; each
continuous proxy gets a Kruskal-Wallis test (teamscale.stats, stdlib).

PRE-REGISTERED PASS CRITERION: no test significant after
Benjamini-Hochberg correction at q = 0.05, AND every effect size
negligible (Cramer's V < 0.10 for the chi-square, eta-squared-H < 0.01
for Kruskal-Wallis). A position-shuffle control (decile labels
permuted with a seeded RNG) must also return null.

The report is deterministic: same generator version, same output.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from . import chains as chains_mod
from .stats import benjamini_hochberg, chi_square_test, kruskal_wallis

CALIBRATION_VERSION = "calibration-1"
DEFAULT_CHAINS = 500
DEFAULT_STEPS = 50
SEED_BASE = 100_000          # disjoint from every experiment grid
Q_LEVEL = 0.05
V_NEGLIGIBLE = 0.10
ETA2_NEGLIGIBLE = 0.01
DECILE = 5                   # positions per decile: 1-5, 6-10, ...

CONTINUOUS_PROXIES = ("op_count", "state_size", "spec_len",
                      "probe_count", "probe_visibility")


def step_proxies(step, state_len: int) -> dict:
    return {
        "template": step.template,
        "op_count": step.op_count,
        "state_size": state_len,
        "spec_len": len(step.spec_text),
        "probe_count": chains_mod.PROBE_COUNT,
        "probe_visibility": round(chains_mod.probe_visibility(step), 6),
    }


def collect_proxies(n_chains: int = DEFAULT_CHAINS,
                    n_steps: int = DEFAULT_STEPS,
                    seed_base: int = SEED_BASE) -> list:
    """One proxy dict per step, tagged with its 1-based position."""
    rows = []
    for i in range(n_chains):
        task = chains_mod.make_chain(seed_base + i, n_steps)
        for step in task.steps:
            row = step_proxies(step, chains_mod.STATE_LEN)
            row["position"] = step.index
            rows.append(row)
    return rows


def _decile(position: int) -> int:
    return (position - 1) // DECILE


def run_tests(rows: list, labels: list) -> list:
    """Statistical tests of proxy ~ label. `labels` parallels `rows`
    (decile index per step) -- the shuffle control passes permuted
    labels through the same code path."""
    groups = sorted(set(labels))
    tests = []
    # categorical: template mix by decile
    templates = chains_mod.TEMPLATE_NAMES
    table = [[0] * len(templates) for _ in groups]
    for row, lab in zip(rows, labels):
        table[groups.index(lab)][templates.index(row["template"])] += 1
    res = chi_square_test(table)
    tests.append({"name": "template_mix", "kind": "chi_square",
                  "threshold": V_NEGLIGIBLE, **res})
    # continuous proxies by decile
    for proxy in CONTINUOUS_PROXIES:
        values = [[row[proxy] for row, lab in zip(rows, labels)
                   if lab == g] for g in groups]
        if len({row[proxy] for row in rows}) <= 1:
            tests.append({"name": proxy, "kind": "kruskal_wallis",
                          "statistic": 0.0, "df": len(groups) - 1,
                          "p_value": 1.0, "effect_size": 0.0,
                          "effect_name": "eta_squared_h",
                          "threshold": ETA2_NEGLIGIBLE,
                          "note": "proxy is constant across all steps"})
            continue
        res = kruskal_wallis(values)
        tests.append({"name": proxy, "kind": "kruskal_wallis",
                      "threshold": ETA2_NEGLIGIBLE, **res})
    adjusted = benjamini_hochberg([t["p_value"] for t in tests],
                                  q=Q_LEVEL)
    for test, adj in zip(tests, adjusted):
        test["p_adjusted"] = adj
        test["significant"] = adj <= Q_LEVEL
        test["negligible"] = test["effect_size"] < test["threshold"]
        for key in ("statistic", "p_value", "effect_size"):
            test[key] = round(test[key], 6)
        test["p_adjusted"] = round(test["p_adjusted"], 6)
    return tests


def run_calibration(n_chains: int = DEFAULT_CHAINS,
                    n_steps: int = DEFAULT_STEPS,
                    seed_base: int = SEED_BASE) -> dict:
    rows = collect_proxies(n_chains, n_steps, seed_base)
    labels = [_decile(r["position"]) for r in rows]
    tests = run_tests(rows, labels)
    rng = random.Random(0xCA11B)
    shuffled = list(labels)
    rng.shuffle(shuffled)
    control = run_tests(rows, shuffled)
    gate_pass = (all(not t["significant"] and t["negligible"]
                     for t in tests)
                 and all(not t["significant"] for t in control))
    return {
        "version": CALIBRATION_VERSION,
        "generator_version": chains_mod.GENERATOR_VERSION,
        "n_chains": n_chains, "n_steps_per_chain": n_steps,
        "n_steps_total": len(rows), "seed_base": seed_base,
        "decile_size": DECILE,
        "criteria": {
            "bh_q": Q_LEVEL,
            "cramers_v_negligible_below": V_NEGLIGIBLE,
            "eta_squared_negligible_below": ETA2_NEGLIGIBLE,
            "rule": "no test BH-significant AND every effect size "
                    "negligible AND shuffle control null"},
        "tests": tests,
        "shuffle_control": {
            "tests": [{"name": t["name"], "p_value": t["p_value"],
                       "p_adjusted": t["p_adjusted"],
                       "significant": t["significant"],
                       "effect_size": t["effect_size"]} for t in control],
            "null": all(not t["significant"] for t in control)},
        "gate": "pass" if gate_pass else "fail",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="teamscale.calibrate",
        description="offline difficulty-calibration gate for the "
                    "chained task family (no LLM calls)")
    ap.add_argument("--chains", type=int, default=DEFAULT_CHAINS)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--seed-base", type=int, default=SEED_BASE)
    ap.add_argument("--out", default=None,
                    help="write the calibration report JSON here")
    args = ap.parse_args(argv)
    report = run_calibration(args.chains, args.steps, args.seed_base)
    for t in report["tests"]:
        print(f"{t['name']:<18} {t['kind']:<15} stat={t['statistic']:<12} "
              f"p={t['p_value']:<10} p_adj={t['p_adjusted']:<10} "
              f"{t['effect_name']}={t['effect_size']:<10} "
              f"significant={t['significant']} "
              f"negligible={t['negligible']}")
    print(f"shuffle control null: {report['shuffle_control']['null']}")
    print(f"GATE: {report['gate'].upper()} "
          f"({report['n_steps_total']} steps, "
          f"generator {report['generator_version']}, "
          f"calibration {report['version']})")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
        print(f"wrote {args.out}")
    return 0 if report["gate"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
