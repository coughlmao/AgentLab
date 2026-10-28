"""Experiment grid CLI.

    python -m teamscale.experiment [--seeds 3] [--out experiments] [--no-audit]

Runs team_size x strategy x family at a fixed per-family budget, writes
results.csv / results.json / summary.json, prints the scaling table, the
bend points, the best strategy per family under the budget, and an
evaluation-artifact audit on a chosen configuration.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .audits import run_audits
from .harness import RunConfig, run_experiment
from .metrics import csv_row, CSV_FIELDS
from .strategies import STRATEGIES
from .tasks import FAMILIES, make_task

TEAM_SIZES = [1, 2, 4, 8, 16, 32]
BUDGET_FACTOR = 1.5      # fixed budget = 1.5x the family's expected solo cost
BEND_THRESHOLD = 0.02    # marginal success gain per step in team size, in points


def budget_for(family: str) -> int:
    return int(make_task(family, seed=0).total_expected_tokens() * BUDGET_FACTOR)


def run_grid(team_sizes=TEAM_SIZES, strategies=None, families=None, seeds=(0, 1, 2),
             out_dir="experiments", write_traces=True):
    strategies = strategies or list(STRATEGIES)
    families = families or list(FAMILIES)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trace_dir = str(out / "traces") if write_traces else None
    rows = []
    for family in families:
        budget = budget_for(family)
        for strategy in strategies:
            for team in team_sizes:
                for seed in seeds:
                    cfg = RunConfig(family=family, strategy=strategy, team_size=team,
                                    budget=budget, seed=seed, trace_dir=trace_dir,
                                    sandbox_root=str(out / "sandboxes"))
                    row = csv_row(run_experiment(cfg).metrics)
                    row["team_size"] = team  # requested size (solo effectively runs 1)
                    rows.append(row)
    with open(out / "results.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with open(out / "results.json", "w") as fh:
        json.dump(rows, fh, indent=1)
    summary = summarize(rows)
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    return rows, summary


def _bend(sizes, rates):
    for i in range(1, len(sizes)):
        if rates[i] - rates[i - 1] < BEND_THRESHOLD:
            return sizes[i]
    return sizes[-1]


def summarize(rows):
    # success rate + means per (family, strategy, team_size)
    cells = defaultdict(list)
    for r in rows:
        cells[(r["family"], r["strategy"], r["team_size"])].append(r)
    scaling = {}
    for (family, strategy, team), cell in sorted(cells.items()):
        scaling.setdefault(family, {}).setdefault(strategy, {})[str(team)] = {
            "success_rate": round(statistics.fmean(c["success"] for c in cell), 4),
            "mean_fraction_correct": round(statistics.fmean(c["fraction_correct"] for c in cell), 4),
            "mean_total_tokens": round(statistics.fmean(c["total_tokens"] for c in cell), 1),
            "mean_wall_time": round(statistics.fmean(c["wall_time"] for c in cell), 3),
        }
    bend_points, best = {}, {}
    for family, per_strategy in scaling.items():
        bend_points[family] = {}
        overall_rates = defaultdict(list)
        for strategy, per_team in per_strategy.items():
            sizes = sorted(int(t) for t in per_team)
            # Bend is read off the mean fraction-correct curve: all-or-nothing
            # success over a handful of seeds only takes values k/n and its
            # "bend" is seed noise, not signal. The success table is still
            # reported alongside.
            rates = [per_team[str(s)]["mean_fraction_correct"] for s in sizes]
            bend_points[family][strategy] = _bend(sizes, rates)
            for s, rate in zip(sizes, rates):
                overall_rates[s].append(rate)
        sizes = sorted(overall_rates)
        bend_points[family]["overall"] = _bend(
            sizes, [statistics.fmean(overall_rates[s]) for s in sizes])
        # best strategy under the (fixed) budget: success first, tokens as tiebreak
        candidates = []
        for strategy, per_team in per_strategy.items():
            succ = statistics.fmean(v["success_rate"] for v in per_team.values())
            frac = statistics.fmean(v["mean_fraction_correct"] for v in per_team.values())
            toks = statistics.fmean(v["mean_total_tokens"] for v in per_team.values())
            candidates.append((succ, frac, -toks, strategy))
        candidates.sort(reverse=True)
        succ, frac, neg_toks, strategy = candidates[0]
        best[family] = {"strategy": strategy, "mean_success_rate": round(succ, 4),
                        "mean_fraction_correct": round(frac, 4),
                        "mean_total_tokens": round(-neg_toks, 1)}
    return {"scaling": scaling, "bend_points": bend_points, "best_strategy": best}


def print_report(summary):
    print("== scaling table: success rate by team size ==")
    print("   (bend points are read off the mean fraction-correct curve; see summarize())")
    for family, per_strategy in summary["scaling"].items():
        print(f"\n{family}  (bend, overall: team {summary['bend_points'][family]['overall']})")
        sizes = sorted({int(t) for per in per_strategy.values() for t in per})
        print("  strategy      " + "".join(f"n={s:<8}" for s in sizes) + "bend")
        for strategy, per_team in per_strategy.items():
            row = "".join(f"{per_team[str(s)]['success_rate']:<10.2f}" for s in sizes)
            print(f"  {strategy:<14}{row}{summary['bend_points'][family][strategy]}")
    print("\n== best strategy under fixed budget, per family ==")
    for family, b in summary["best_strategy"].items():
        print(f"  {family:<16} {b['strategy']:<14} success {b['mean_success_rate']:.3f} "
              f" fraction {b['mean_fraction_correct']:.3f} tokens {b['mean_total_tokens']:.0f}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="teamscale.experiment")
    ap.add_argument("--seeds", type=int, default=5, help="seeds per cell (0..n-1)")
    ap.add_argument("--out", default="experiments")
    ap.add_argument("--no-traces", action="store_true")
    ap.add_argument("--no-audit", action="store_true")
    ap.add_argument("--audit-family", default="long_horizon")
    ap.add_argument("--audit-strategy", default="orchestrated")
    ap.add_argument("--audit-team", type=int, default=4)
    args = ap.parse_args(argv)
    seeds = tuple(range(args.seeds))
    rows, summary = run_grid(seeds=seeds, out_dir=args.out, write_traces=not args.no_traces)
    print(f"grid: {len(rows)} runs -> {args.out}/results.csv, results.json, summary.json")
    print_report(summary)
    if not args.no_audit:
        cfg = RunConfig(family=args.audit_family, strategy=args.audit_strategy,
                        team_size=args.audit_team, budget=budget_for(args.audit_family),
                        seed=0)
        print("\n" + run_audits(cfg).summary())


if __name__ == "__main__":
    main()
