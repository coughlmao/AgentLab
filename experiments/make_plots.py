"""Plot scaling curves from experiments/results.csv.

    .venv/bin/python experiments/make_plots.py

Produces experiments/scaling_curves.png: success rate vs team size, one line
per strategy, one panel per task family. Uses matplotlib when available and
falls back to a hand-rolled SVG (scaling_curves.svg) with the stdlib only.
"""
from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.csv"


def load():
    # Main curve = mean fraction of subtasks correct. All-or-nothing success
    # over a handful of seeds only takes values k/n and reads as noise; it is
    # still in results.csv / summary.json for anyone who wants it.
    cells = defaultdict(list)
    with open(RESULTS) as fh:
        for row in csv.DictReader(fh):
            cells[(row["family"], row["strategy"], int(row["team_size"]))].append(
                float(row["fraction_correct"]))
    series = defaultdict(dict)   # family -> strategy -> [(team, mean_success)]
    for (family, strategy, team), vals in sorted(cells.items()):
        series[family].setdefault(strategy, []).append((team, statistics.fmean(vals)))
    for family in series:
        for strategy in series[family]:
            series[family][strategy].sort()
    return series


def plot_matplotlib(series, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    families = sorted(series)
    fig, axes = plt.subplots(1, len(families), figsize=(4.2 * len(families), 3.6),
                             sharey=True)
    if len(families) == 1:
        axes = [axes]
    for ax, family in zip(axes, families):
        for strategy, points in sorted(series[family].items()):
            xs, ys = zip(*points)
            ax.plot(xs, ys, marker="o", label=strategy)
        ax.set_title(family)
        ax.set_xlabel("team size")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted({p[0] for per in series[family].values() for p in per}))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("mean fraction of subtasks correct")
    axes[0].set_ylim(-0.03, 1.03)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.suptitle("Agent team scaling: fraction correct vs team size (simulated agents)")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(out_png, dpi=150)
    return out_png


def plot_svg(series, out_svg):
    """Stdlib fallback: one small-multiple panel per family, axes + polylines."""
    W, H, PAD, PANEL_W, PANEL_H, TOP = 340 * len(series), 300, 40, 300, 200, 50
    colors = {"solo": "#888", "independent": "#d95f02", "orchestrated": "#1b9e77",
              "verified": "#1a56db", "budget_aware": "#7570b3"}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'font-family="sans-serif" font-size="11">']
    import math
    for fi, (family, per) in enumerate(sorted(series.items())):
        ox = fi * 340 + PAD
        oy = TOP
        parts.append(f'<text x="{ox}" y="24" font-weight="bold">{family}</text>')
        parts.append(f'<rect x="{ox}" y="{oy}" width="{PANEL_W}" height="{PANEL_H}" fill="none" stroke="#999"/>')
        for gy in (0.0, 0.5, 1.0):
            y = oy + PANEL_H - gy * PANEL_H
            parts.append(f'<line x1="{ox}" y1="{y:.0f}" x2="{ox + PANEL_W}" y2="{y:.0f}" stroke="#ddd"/>')
            parts.append(f'<text x="{ox - 26}" y="{y + 3:.0f}">{gy:.1f}</text>')
        max_log = math.log2(max(p[0] for per_s in per.values() for p in per_s))
        for strategy, points in sorted(per.items()):
            pts = []
            for team, val in points:
                x = ox + (math.log2(team) / max_log) * PANEL_W if max_log else ox
                y = oy + PANEL_H - val * PANEL_H
                pts.append(f"{x:.1f},{y:.1f}")
            c = colors.get(strategy, "#333")
            parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{c}" stroke-width="2"/>')
            lx, ly = pts[-1].split(",")
            parts.append(f'<text x="{float(lx) - 58}" y="{float(ly) - 4}" fill="{c}">{strategy}</text>')
        parts.append(f'<text x="{ox}" y="{oy + PANEL_H + 18}">team size (log2) -&gt;</text>')
    parts.append("</svg>")
    out_svg.write_text("\n".join(parts))
    return out_svg


def main():
    if not RESULTS.exists():
        print("no results.csv yet -- run: python -m teamscale.experiment")
        return 1
    series = load()
    try:
        out = plot_matplotlib(series, HERE / "scaling_curves.png")
    except ImportError:
        out = plot_svg(series, HERE / "scaling_curves.svg")
        print("matplotlib unavailable; wrote SVG fallback")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
