"""Generic analysis for config-driven experiment runs.

    python -m teamscale.analyze results/<experiment>/<run-id>

ANALYSIS PURITY (ENGINEERING.md, invariant 4): this module contains no
experiment logic. It never learns what a "team size", a "horizon", or
a "verified strategy" is. Everything experiment-specific -- which
column is the series, which column groups the cells, which column is
the outcome, which two series form the primary contrast, whether the
contrast is also compared across two groups, which table carries
per-step data -- is DECLARED in the experiment config and read back
from the run's manifest.json ("analysis" declaration). Given declared
columns, this module produces, for any experiment:

- per-cell means with percentile bootstrap 95% CIs over runs,
- the declared contrast per group and overall (mean of per-group
  differences), each with a bootstrap CI; when the declaration names
  an `across` pair of groups, also the difference of the contrast
  between those groups (e.g. effect at horizon 50 minus horizon 2),
- when a per-step table is declared: survival curves P(state correct
  after step k) and hazard rates h(k) = P(first failure at k |
  survived k-1) per group x series, with bootstrap CIs, and the
  survival divergence step (first k whose contrast-gap CI excludes 0),
- cost accounting, figures, analysis.json and SUMMARY.md.

Inputs are tabular only: metrics.csv and the declared steps table,
plus manifest.json (labels, declaration, caveats) and summary.json
(exact cost accounting when present). It never reads traces, result
JSONs, sandboxes, or generator internals. If the run directory is
sealed (hashes.json), the artifact seal is verified before anything
is read; the files written here are recorded under the seal's
"derived" section.

Determinism: the bootstrap RNG is seeded from the metrics.csv bytes
(and the steps RNG from the steps-table bytes), so re-running the
analysis on the same run directory regenerates byte-identical outputs.

DEFAULT_ANALYSIS is the lab's documented default declaration, used
only when a run's manifest carries no declaration of its own (it is
the declaration the Phase A experiment would have written).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import zlib
from collections import defaultdict
from pathlib import Path

ANALYSIS_VERSION = "2"
RESAMPLES = 5000
STEPS_RESAMPLES = 2000
COLORS = {"solo": "#888", "independent": "#d95f02", "orchestrated": "#1b9e77",
          "verified": "#1a56db", "budget_aware": "#7570b3"}
_PALETTE = ["#1a56db", "#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e"]

DEFAULT_ANALYSIS = {
    "series": "strategy",
    "group_by": "team_size",
    "value": "fraction_correct",
    "contrast": {"positive": "verified", "negative": "orchestrated"},
    "x_scale": "log2",
    "x_label": "team size (log2)",
}

CHAIN_ANALYSIS_DEFAULT = {
    "series": "strategy",
    "group_by": "horizon",
    "value": "final_state_correct",
    "contrast": {"positive": "verified", "negative": "orchestrated",
                 "across": [2, 50]},
    "x_scale": "log2",
    "x_label": "chain length (steps, log2)",
    "steps": {"file": "steps.csv", "index": "step_index",
              "value": "state_correct"},
    "caveats_position": "top",
}

GENERIC_CAVEATS = [
    "Intervals are percentile bootstrap intervals over runs within "
    "each cell; cells hold few seeds, so intervals are wide by "
    "construction. Read the contrast intervals, not point estimates "
    "alone.",
    "Figures and every number in this summary are regenerated from "
    "the run's declared tabular outputs (metrics.csv and any declared "
    "steps table) by `python -m teamscale.analyze <run dir>`; grading "
    "and accounting can be verified with `python -m "
    "teamscale.replay <run dir>`.",
]


# ------------------------------------------------------------------- loading
def _coerce(text):
    try:
        return int(text)
    except (TypeError, ValueError):
        pass
    try:
        return float(text)
    except (TypeError, ValueError):
        pass
    return text


def load_table(path: Path) -> list:
    with open(path, newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _num(row, column) -> float:
    return float(row[column])


def resolve_declaration(manifest: dict) -> dict:
    decl = manifest.get("analysis")
    if not isinstance(decl, dict):
        decl = (manifest.get("config") or {}).get("analysis")
    if not isinstance(decl, dict):
        decl = DEFAULT_ANALYSIS
    merged = dict(DEFAULT_ANALYSIS)
    merged.update(decl)
    return merged


# ----------------------------------------------------------------- bootstrap
def _resampled_mean(values, rng):
    n = len(values)
    return statistics.fmean(values[rng.randrange(n)] for _ in range(n))


def bootstrap_ci(values, rng, resamples=RESAMPLES):
    """Percentile 95% CI for the mean, resampling runs within the cell."""
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])
    means = sorted(_resampled_mean(values, rng) for _ in range(resamples))
    lo = means[int(0.025 * resamples)]
    hi = means[min(resamples - 1, int(0.975 * resamples))]
    return (lo, hi)


def _contrast(cells, groups, positive, negative, rng,
              resamples=RESAMPLES):
    """positive - negative per group and overall (mean of per-group
    differences), with bootstrap CIs from within-cell resampling."""
    per_group = {}
    for group in groups:
        v = cells.get((positive, group), [])
        o = cells.get((negative, group), [])
        if not v or not o:
            continue
        diffs = []
        for _ in range(resamples):
            diffs.append(_resampled_mean(v, rng) - _resampled_mean(o, rng))
        diffs.sort()
        per_group[group] = {
            "difference": round(statistics.fmean(v) - statistics.fmean(o), 4),
            "ci95": [round(diffs[int(0.025 * resamples)], 4),
                     round(diffs[min(resamples - 1,
                                     int(0.975 * resamples))], 4)],
        }
    overall = None
    if per_group:
        diffs = []
        for _ in range(resamples):
            ds = []
            for group in groups:
                v = cells.get((positive, group), [])
                o = cells.get((negative, group), [])
                if v and o:
                    ds.append(_resampled_mean(v, rng)
                              - _resampled_mean(o, rng))
            diffs.append(statistics.fmean(ds))
        diffs.sort()
        overall = {
            "difference": round(statistics.fmean(
                c["difference"] for c in per_group.values()), 4),
            "ci95": [round(diffs[int(0.025 * resamples)], 4),
                     round(diffs[min(resamples - 1,
                                     int(0.975 * resamples))], 4)],
        }
    return {"by_group": {str(k): v for k, v in per_group.items()},
            "by_size": {str(k): v for k, v in per_group.items()},
            "overall": overall}


def _contrast_across(cells, pair, positive, negative, rng,
                     resamples=RESAMPLES):
    """The contrast at pair[1] minus the contrast at pair[0]: does the
    treatment effect itself change across the grouping variable?"""
    g_lo, g_hi = pair
    cells_lo = (cells.get((positive, g_lo), []),
                cells.get((negative, g_lo), []))
    cells_hi = (cells.get((positive, g_hi), []),
                cells.get((negative, g_hi), []))
    if not all(cells_lo) or not all(cells_hi):
        return None

    def effect(cell_pair):
        return (statistics.fmean(cell_pair[0])
                - statistics.fmean(cell_pair[1]))

    diffs = []
    for _ in range(resamples):
        e_lo = (_resampled_mean(cells_lo[0], rng)
                - _resampled_mean(cells_lo[1], rng))
        e_hi = (_resampled_mean(cells_hi[0], rng)
                - _resampled_mean(cells_hi[1], rng))
        diffs.append(e_hi - e_lo)
    diffs.sort()
    return {"groups": [g_lo, g_hi],
            "difference": round(effect(cells_hi) - effect(cells_lo), 4),
            "ci95": [round(diffs[int(0.025 * resamples)], 4),
                     round(diffs[min(resamples - 1,
                                     int(0.975 * resamples))], 4)]}


# ------------------------------------------------------- survival and hazard
def _run_vectors(rows, decl):
    """(series, group) -> {run_id: [per-step values in index order]}."""
    steps_decl = decl["steps"]
    idx_col, val_col = steps_decl["index"], steps_decl["value"]
    series_col, group_col = decl["series"], decl["group_by"]
    runs = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row[series_col], _coerce(row[group_col]))
        runs[key][row["run_id"]].append(
            (int(row[idx_col]), int(float(row[val_col]))))
    vectors = {}
    for key, per_run in runs.items():
        vectors[key] = [ [v for _, v in sorted(per_run[rid])]
                         for rid in sorted(per_run)]
    return vectors


def _survival_curve(vectors, upto):
    """P(all steps <= k correct), k = 1..upto, per run-vector set."""
    return [statistics.fmean(
        1.0 if all(vec[i] for i in range(k)) else 0.0
        for vec in vectors) for k in range(1, upto + 1)]


def _hazard_curve(vectors, upto):
    """h(k) = P(first failure at k | survived through k-1)."""
    firsts = []
    for vec in vectors:
        first = next((i + 1 for i, v in enumerate(vec) if not v), None)
        firsts.append(first)
    hazard = []
    for k in range(1, upto + 1):
        at_risk = sum(1 for f in firsts if f is None or f >= k)
        failed = sum(1 for f in firsts if f == k)
        hazard.append(failed / at_risk if at_risk else None)
    return hazard


def _bootstrap_curve(vectors, upto, curve_fn, rng,
                     resamples=STEPS_RESAMPLES):
    n = len(vectors)
    draws = [[] for _ in range(upto)]
    for _ in range(resamples):
        sample = [vectors[rng.randrange(n)] for _ in range(n)]
        curve = curve_fn(sample, upto)
        for k, val in enumerate(curve):
            if val is not None:
                draws[k].append(val)
    cis = []
    for k in range(upto):
        vals = sorted(draws[k])
        if not vals:
            cis.append([None, None])
        else:
            cis.append([round(vals[int(0.025 * len(vals))], 4),
                        round(vals[min(len(vals) - 1,
                                       int(0.975 * len(vals)))], 4)])
    return cis


def steps_analysis(rows, decl, rng):
    """Survival + hazard per group x series, and the divergence step
    for the declared contrast. Generic over the declared columns."""
    vectors = _run_vectors(rows, decl)
    groups = sorted({g for (_, g) in vectors})
    series_names = sorted({s for (s, _) in vectors})
    survival, hazard = {}, {}
    for group in groups:
        survival[str(group)], hazard[str(group)] = {}, {}
        for series in series_names:
            vecs = vectors.get((series, group), [])
            if not vecs:
                continue
            upto = max(len(v) for v in vecs)
            s_curve = _survival_curve(vecs, upto)
            h_curve = _hazard_curve(vecs, upto)
            s_ci = _bootstrap_curve(vecs, upto, _survival_curve, rng)
            h_ci = _bootstrap_curve(vecs, upto, _hazard_curve, rng)
            survival[str(group)][series] = {
                "steps": list(range(1, upto + 1)),
                "survival": [round(v, 4) for v in s_curve],
                "ci95": s_ci}
            hazard[str(group)][series] = {
                "steps": list(range(1, upto + 1)),
                "hazard": [(round(v, 4) if v is not None else None)
                           for v in h_curve],
                "ci95": h_ci}
    divergence = {}
    contrast = decl.get("contrast") or {}
    pos, neg = contrast.get("positive"), contrast.get("negative")
    for group in groups:
        step_found = None
        if pos in series_names and neg in series_names:
            v_pos = vectors.get((pos, group), [])
            v_neg = vectors.get((neg, group), [])
            if v_pos and v_neg:
                upto = max(max(len(v) for v in v_pos),
                           max(len(v) for v in v_neg))
                gaps = [[] for _ in range(upto)]
                for _ in range(STEPS_RESAMPLES):
                    sp = _survival_curve(
                        [v_pos[rng.randrange(len(v_pos))]
                         for _ in v_pos], upto)
                    sn = _survival_curve(
                        [v_neg[rng.randrange(len(v_neg))]
                         for _ in v_neg], upto)
                    for k in range(upto):
                        gaps[k].append(sp[k] - sn[k])
                for k in range(upto):
                    vals = sorted(gaps[k])
                    lo = vals[int(0.025 * len(vals))]
                    if lo > 0:
                        step_found = k + 1
                        break
        divergence[str(group)] = step_found
    return {"survival": survival, "hazard": hazard,
            "divergence_step": divergence}


# ------------------------------------------------------------------- figures
def _panel(width=680, height=380):
    geo = {"ox": 56, "oy": 40, "w": width - 56 - 132, "h": height - 40 - 46,
           "width": width, "height": height}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
             f'height="{height}" font-family="sans-serif" font-size="11">']
    return parts, geo


def _color(series, all_series):
    if series in COLORS:
        return COLORS[series]
    return _PALETTE[all_series.index(series) % len(_PALETTE)]


def _x_mapper(groups, geo, x_scale):
    ox, w = geo["ox"], geo["w"]
    numeric = all(isinstance(g, (int, float)) for g in groups)
    if numeric and x_scale == "log2" and min(groups) > 0 \
            and len(set(groups)) > 1:
        lo, hi = math.log2(min(groups)), math.log2(max(groups))
        return lambda g: ox + (math.log2(g) - lo) / (hi - lo) * w
    if numeric and len(set(groups)) > 1:
        lo, hi = min(groups), max(groups)
        return lambda g: ox + (g - lo) / (hi - lo) * w
    if numeric:
        return lambda g: ox + w / 2
    step = w / max(len(groups), 1)
    return lambda g: ox + step * (groups.index(g) + 0.5)


def _axes_generic(parts, geo, title, ymax, yfmt, groups, x_for, x_label):
    ox, oy, w, h = geo["ox"], geo["oy"], geo["w"], geo["h"]
    parts.append(f'<text x="{ox}" y="22" font-weight="bold">{title}</text>')
    parts.append(f'<rect x="{ox}" y="{oy}" width="{w}" height="{h}" '
                 f'fill="none" stroke="#999"/>')
    for i in range(5):
        gy = ymax * i / 4
        y = oy + h - (gy / ymax) * h if ymax else oy + h
        parts.append(f'<line x1="{ox}" y1="{y:.0f}" x2="{ox + w}" '
                     f'y2="{y:.0f}" stroke="#ddd"/>')
        parts.append(f'<text x="{ox - 30}" y="{y + 3:.0f}">{yfmt(gy)}</text>')
    for g in groups:
        x = x_for(g)
        parts.append(f'<text x="{x - 4:.0f}" y="{oy + h + 16}">{g}</text>')
    parts.append(f'<text x="{ox}" y="{oy + h + 32}">{x_label} -&gt;</text>')
    return parts


def outcome_svg(cell_stats, groups, series_names, decl) -> str:
    parts, geo = _panel()
    ox, oy, w, h = geo["ox"], geo["oy"], geo["w"], geo["h"]
    x_for = _x_mapper(groups, geo, decl.get("x_scale"))
    _axes_generic(parts, geo,
                  f"mean {decl['value']} vs {decl['group_by']} (95% CI)",
                  1.0, lambda v: f"{v:.2f}", groups, x_for,
                  decl.get("x_label", decl["group_by"]))
    value_key = f"mean_{decl['value']}"
    for series in series_names:
        color = _color(series, series_names)
        pts = []
        for group in groups:
            cell = cell_stats.get((series, group))
            if not cell:
                continue
            x = x_for(group)
            mean = cell[value_key]
            y = oy + h - mean * h
            ylo = oy + h - cell["ci95"][0] * h
            yhi = oy + h - cell["ci95"][1] * h
            parts.append(f'<line x1="{x:.1f}" y1="{ylo:.1f}" x2="{x:.1f}" '
                         f'y2="{yhi:.1f}" stroke="{color}" '
                         f'stroke-width="1.5"/>')
            parts.append(f'<line x1="{x - 4:.1f}" y1="{ylo:.1f}" '
                         f'x2="{x + 4:.1f}" y2="{ylo:.1f}" '
                         f'stroke="{color}"/>')
            parts.append(f'<line x1="{x - 4:.1f}" y1="{yhi:.1f}" '
                         f'x2="{x + 4:.1f}" y2="{yhi:.1f}" '
                         f'stroke="{color}"/>')
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" '
                         f'fill="{color}"/>')
            pts.append(f"{x:.1f},{y:.1f}")
        if pts:
            parts.append(f'<polyline points="{" ".join(pts)}" fill="none" '
                         f'stroke="{color}" stroke-width="2"/>')
            lx, ly = pts[-1].split(",")
            parts.append(f'<text x="{float(lx) + 8:.0f}" '
                         f'y="{float(ly) + 3:.0f}" fill="{color}">'
                         f'{series}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def cost_svg(rows, groups, series_names, decl) -> str:
    if not rows or "cost_usd" not in rows[0]:
        return ""
    group_col = decl["group_by"]
    means, ymax = {}, 0.0
    for series in series_names:
        for group in groups:
            cell = [_num(r, "cost_usd") for r in rows
                    if r[decl["series"]] == series
                    and _coerce(r[group_col]) == group]
            if cell:
                means[(series, group)] = statistics.fmean(cell)
                ymax = max(ymax, means[(series, group)])
    ymax = ymax * 1.15 if ymax > 0 else 1.0
    yfmt = (lambda v: f"{v:.3f}") if ymax < 0.2 else (lambda v: f"{v:.2f}")
    parts, geo = _panel()
    ox, oy, w, h = geo["ox"], geo["oy"], geo["w"], geo["h"]
    _axes_generic(parts, geo,
                  f"mean cost per run (USD), by {group_col} and "
                  f"{decl['series']}", ymax, yfmt, [], lambda g: ox,
                  "")
    group_w = w / len(groups)
    bar_w = group_w / (len(series_names) + 1)
    for gi, group in enumerate(groups):
        gx = ox + gi * group_w
        for si, series in enumerate(series_names):
            val = means.get((series, group))
            if val is None:
                continue
            bh = (val / ymax) * h
            x = gx + bar_w * (si + 0.5)
            color = _color(series, series_names)
            parts.append(f'<rect x="{x:.1f}" y="{oy + h - bh:.1f}" '
                         f'width="{bar_w * 0.9:.1f}" height="{bh:.1f}" '
                         f'fill="{color}"/>')
            parts.append(f'<text x="{x:.1f}" y="{oy + h - bh - 3:.1f}" '
                         f'font-size="9">{val:.4f}</text>')
        parts.append(f'<text x="{gx + group_w / 2 - 8:.0f}" '
                     f'y="{oy + h + 16}">{group_col}={group}</text>')
    for si, series in enumerate(series_names):
        color = _color(series, series_names)
        y = oy + 6 + si * 15
        parts.append(f'<rect x="{ox + w + 12}" y="{y - 9}" width="10" '
                     f'height="10" fill="{color}"/>')
        parts.append(f'<text x="{ox + w + 26}" y="{y}" fill="{color}">'
                     f'{series}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _curve_svg(per_series, kind: str, group, decl) -> str:
    """Survival or hazard curves for one group value."""
    parts, geo = _panel()
    ox, oy, w, h = geo["ox"], geo["oy"], geo["w"], geo["h"]
    steps = sorted({k for s in per_series.values() for k in s["steps"]})
    if not steps:
        return ""
    ymax = 1.0
    if kind == "hazard":
        vals = [v for s in per_series.values()
                for v in s["hazard"] if v is not None]
        ymax = max(vals) * 1.2 if vals and max(vals) > 0 else 1.0
    x_for = _x_mapper(steps, geo, "linear")
    _axes_generic(parts, geo,
                  f"{kind} by step -- {decl['group_by']}={group}",
                  ymax, lambda v: f"{v:.2f}", steps, x_for, "step")
    series_names = sorted(per_series)
    for series in series_names:
        data = per_series[series]
        color = _color(series, series_names)
        pts = []
        for k, val, ci in zip(data["steps"], data[kind], data["ci95"]):
            if val is None:
                continue
            x = x_for(k)
            y = oy + h - (val / ymax) * h
            if ci and ci[0] is not None:
                ylo = oy + h - (ci[0] / ymax) * h
                yhi = oy + h - (ci[1] / ymax) * h
                parts.append(f'<line x1="{x:.1f}" y1="{ylo:.1f}" '
                             f'x2="{x:.1f}" y2="{yhi:.1f}" '
                             f'stroke="{color}" stroke-width="1"/>')
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" '
                         f'fill="{color}"/>')
            pts.append(f"{x:.1f},{y:.1f}")
        if pts:
            parts.append(f'<polyline points="{" ".join(pts)}" fill="none" '
                         f'stroke="{color}" stroke-width="2"/>')
            lx, ly = pts[-1].split(",")
            parts.append(f'<text x="{float(lx) + 8:.0f}" '
                         f'y="{float(ly) + 3:.0f}" fill="{color}">'
                         f'{series}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


# ------------------------------------------------------------------ analysis
def analyze_run(run_dir, resamples=RESAMPLES, analysis_spec=None) -> dict:
    run_dir = Path(run_dir)
    manifest = {}
    if (run_dir / "manifest.json").exists():
        manifest = json.loads((run_dir / "manifest.json").read_text())
    decl = dict(analysis_spec) if analysis_spec else \
        resolve_declaration(manifest)

    if (run_dir / "hashes.json").exists():
        from .sealing import verify_run_dir
        bad = [d for d in verify_run_dir(run_dir) if "(artifacts)" in d]
        if bad:
            raise ValueError(
                "run directory seal does not verify; refusing to "
                f"analyze: {bad}")

    metrics_bytes = (run_dir / "metrics.csv").read_bytes()
    rows = load_table(run_dir / "metrics.csv")
    series_col, group_col, value_col = (decl["series"], decl["group_by"],
                                        decl["value"])
    rng = random.Random(zlib.crc32(metrics_bytes) ^ 0x5EED)

    cells = defaultdict(list)
    for r in rows:
        cells[(r[series_col], _coerce(r[group_col]))].append(
            _num(r, value_col))
    groups = sorted({g for (_, g) in cells})
    series_names = sorted({s for (s, _) in cells})

    cell_stats = {}
    for key in sorted(cells):
        vals = cells[key]
        cell_rows = [r for r in rows
                     if r[series_col] == key[0]
                     and _coerce(r[group_col]) == key[1]]
        lo, hi = bootstrap_ci(vals, rng, resamples)
        cell = {"series": key[0], "group": key[1], "n": len(vals),
                series_col: key[0], group_col: key[1],
                f"mean_{value_col}": round(statistics.fmean(vals), 4),
                "ci95": [round(lo, 4), round(hi, 4)]}
        if rows and "tokens_in" in rows[0] and "tokens_out" in rows[0]:
            cell["mean_tokens"] = round(statistics.fmean(
                _num(r, "tokens_in") + _num(r, "tokens_out")
                for r in cell_rows), 1)
        if rows and "cost_usd" in rows[0]:
            cell["mean_cost_usd"] = round(statistics.fmean(
                _num(r, "cost_usd") for r in cell_rows), 6)
        if rows and "retries" in rows[0]:
            cell["retries_total"] = int(sum(_num(r, "retries")
                                            for r in cell_rows))
        cell_stats[key] = cell

    contrast = None
    decl_contrast = decl.get("contrast") or {}
    pos, neg = decl_contrast.get("positive"), decl_contrast.get("negative")
    if pos in series_names and neg in series_names:
        body = _contrast(cells, groups, pos, neg, rng, resamples)
        contrast = {"strategies": [pos, neg], **body}
        across = decl_contrast.get("across")
        if across:
            pair = [_coerce(str(across[0])), _coerce(str(across[1]))]
            contrast["across"] = _contrast_across(
                cells, pair, pos, neg, rng, resamples)

    steps_out = None
    if decl.get("steps"):
        steps_path = run_dir / decl["steps"]["file"]
        if steps_path.exists():
            steps_bytes = steps_path.read_bytes()
            steps_rows = load_table(steps_path)
            steps_rng = random.Random(zlib.crc32(steps_bytes) ^ 0x5EED)
            steps_out = steps_analysis(steps_rows, decl, steps_rng)

    total_cost = (round(sum(_num(r, "cost_usd") for r in rows), 6)
                  if rows and "cost_usd" in rows[0] else 0.0)
    cost = {"total_cost_usd": total_cost,
            "cost_per_run_usd": (round(total_cost / len(rows), 6)
                                 if rows else 0.0),
            "by_strategy": {}}
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        saved = json.loads(summary_path.read_text())
        if "cost_accounting" in saved:
            cost["by_strategy"] = saved["cost_accounting"]["by_strategy"]
            cost["source"] = "summary.json (run ledger)"
    if not cost["by_strategy"]:
        cost["source"] = "metrics.csv totals"
        for series in series_names:
            cell = [r for r in rows if r[series_col] == series]
            c = sum(_num(r, "cost_usd") for r in cell) if cell else 0.0
            cost["by_strategy"][series] = {
                "runs": len(cell), "total_cost_usd": round(c, 6),
                "cost_per_run_usd": (round(c / len(cell), 6)
                                     if cell else 0.0),
                "subtasks_correct": None,
                "cost_per_correct_subtask_usd": None}

    analysis = {
        "experiment": manifest.get("experiment", run_dir.parent.name),
        "question": manifest.get("question", ""),
        "run_id": manifest.get("run_id", run_dir.name),
        "provider": manifest.get("provider"),
        "model": manifest.get("model"),
        "difficulty": manifest.get("difficulty"),
        "blind": manifest.get("blind"),
        "family": manifest.get("family", "codebase_repair"),
        "analysis_version": ANALYSIS_VERSION,
        "analysis_declaration": decl,
        "n_runs": len(rows),
        "bootstrap": {"resamples": resamples,
                      "method": "percentile bootstrap over runs within "
                                "each declared series x group cell",
                      "seed": "crc32(metrics.csv) ^ 0x5EED "
                              "(deterministic)"},
        "cells": [cell_stats[k] for k in sorted(cell_stats)],
        "primary_contrast": contrast,
        "steps_analysis": steps_out,
        "cost": cost,
    }

    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    written = ["analysis.json", "SUMMARY.md", "figures/scaling_curve.svg"]
    (figures / "scaling_curve.svg").write_text(
        outcome_svg(cell_stats, groups, series_names, decl))
    cost = cost_svg(rows, groups, series_names, decl)
    if cost:
        (figures / "cost_breakdown.svg").write_text(cost)
        written.append("figures/cost_breakdown.svg")
    if steps_out:
        for group in sorted(steps_out["survival"]):
            for kind in ("survival", "hazard"):
                svg = _curve_svg(steps_out[kind][group], kind, group,
                                 decl)
                if svg:
                    name = f"figures/{kind}_{group}.svg"
                    (run_dir / name).write_text(svg)
                    written.append(name)
    (run_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=1, sort_keys=True))
    (run_dir / "SUMMARY.md").write_text(render_summary_md(analysis,
                                                          manifest))
    if (run_dir / "hashes.json").exists():
        from .sealing import record_derived
        record_derived(run_dir, written)
    return analysis


def _fmt_cell(cell, value_col) -> str:
    return (f"{cell[f'mean_{value_col}']:.3f} "
            f"[{cell['ci95'][0]:.3f}, {cell['ci95'][1]:.3f}]")


def render_summary_md(analysis: dict, manifest: dict) -> str:
    decl = analysis["analysis_declaration"]
    series_col, group_col, value_col = (decl["series"], decl["group_by"],
                                        decl["value"])
    cells = {(c["series"], c["group"]): c for c in analysis["cells"]}
    series_names = sorted({c["series"] for c in analysis["cells"]})
    groups = sorted({c["group"] for c in analysis["cells"]})
    declared_caveats = manifest.get("caveats") or []
    caveats_top = decl.get("caveats_position") == "top"

    lines = [f"# {analysis['experiment']}", "",
             f"**Question:** {analysis['question'] or '(not stated)'}",
             ""]
    if caveats_top and declared_caveats:
        lines += ["## Caveats", ""]
        lines += [f"- {c}" for c in declared_caveats]
        lines += [""]
    lines += [
        "## Design", "",
        f"- provider / model: {analysis['provider']} / "
        f"{analysis['model']}",
        f"- family: {analysis['family']}; difficulty: "
        f"{analysis['difficulty']}"
        + ("; **blind mode** (agents cannot run checks; the external "
           "check is the only check feedback)" if analysis["blind"]
           else ""),
        f"- grid (from the manifest): "
        f"{json.dumps(manifest.get('grid', {}), sort_keys=True)}; "
        f"{analysis['n_runs']} runs completed",
        f"- analysis: declared in the config (series={series_col}, "
        f"group_by={group_col}, value={value_col}); bootstrap 95% CIs, "
        f"{analysis['bootstrap']['resamples']} resamples over runs "
        "within each cell",
        "",
        f"## Headline: mean {value_col} [95% CI]", "",
        "| " + series_col + " | "
        + " | ".join(f"{group_col}={g}" for g in groups) + " |",
        "|---|" + "---|" * len(groups),
    ]
    for series in series_names:
        row = " | ".join(
            _fmt_cell(cells[(series, g)], value_col)
            if (series, g) in cells else "-"
            for g in groups)
        lines.append(f"| {series} | {row} |")
    contrast = analysis.get("primary_contrast")
    if contrast and contrast.get("overall"):
        pos, neg = contrast["strategies"]
        o = contrast["overall"]
        lines += [
            "", f"## Primary contrast: {pos} - {neg}", "",
            f"- overall (mean of per-{group_col} differences): "
            f"{o['difference']:+.3f} [{o['ci95'][0]:+.3f}, "
            f"{o['ci95'][1]:+.3f}]",
        ]
        for group, c in sorted(contrast["by_group"].items(),
                               key=lambda kv: _coerce(kv[0])):
            lines.append(f"- {group_col}={group}: "
                         f"{c['difference']:+.3f} "
                         f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]")
        across = contrast.get("across")
        if across:
            lines.append(
                f"- across groups (effect at {group_col}="
                f"{across['groups'][1]} minus effect at {group_col}="
                f"{across['groups'][0]}): {across['difference']:+.3f} "
                f"[{across['ci95'][0]:+.3f}, {across['ci95'][1]:+.3f}]"
                " -- the pre-registered test of whether the effect "
                "itself changes across the grouping variable")
    steps_out = analysis.get("steps_analysis")
    if steps_out:
        lines += ["", "## Survival and hazard (per-step)", ""]
        for group in sorted(steps_out["survival"],
                            key=lambda g: _coerce(g)):
            div = steps_out["divergence_step"].get(group)
            finals = []
            for series in sorted(steps_out["survival"][group]):
                s = steps_out["survival"][group][series]
                finals.append(f"{series} final survival "
                              f"{s['survival'][-1]:.3f}")
            lines.append(
                f"- {group_col}={group}: " + "; ".join(finals)
                + f"; survival divergence step: "
                f"{div if div is not None else 'none'}")
        lines += ["", "Hazard h(k) = P(first failure at step k | "
                      "survived through k-1); full curves with CIs "
                      "are in analysis.json and figures/hazard_*.svg.",
                  ""]
    cost = analysis["cost"]
    lines += [
        "", "## Cost accounting", "",
        f"- total cost: ${cost['total_cost_usd']:.4f}; cost per run: "
        f"${cost['cost_per_run_usd']:.4f} (source: "
        f"{cost.get('source')})", "",
        "| strategy | runs | total cost | cost/run | cost per correct "
        "subtask |", "|---|---|---|---|---|",
    ]
    for strategy, cell in cost["by_strategy"].items():
        cps = cell.get("cost_per_correct_subtask_usd")
        lines.append(
            f"| {strategy} | {cell['runs']} | "
            f"${cell['total_cost_usd']:.4f} | "
            f"${cell['cost_per_run_usd']:.4f} | "
            f"{'$' + format(cps, '.4f') if cps is not None else 'n/a'} |")
    lines += ["", "## Caveats", ""]
    if declared_caveats and not caveats_top:
        lines += [f"- {c}" for c in declared_caveats]
    elif declared_caveats and caveats_top:
        lines.append("- The experiment-specific caveats are stated at "
                     "the top of this summary, where they belong.")
    lines += [f"- {c}" for c in GENERIC_CAVEATS]
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="teamscale.analyze",
        description="generic declared analysis for a config-driven "
                    "run directory (bootstrap CIs, declared contrasts, "
                    "survival/hazard when a steps table is declared)")
    ap.add_argument("run_dir", help="results/<experiment>/<run-id>")
    ap.add_argument("--resamples", type=int, default=RESAMPLES,
                    help=f"bootstrap resamples (default {RESAMPLES}, "
                         "minimum 2000 recommended)")
    args = ap.parse_args(argv)
    analysis = analyze_run(args.run_dir, resamples=args.resamples)
    contrast = analysis.get("primary_contrast") or {}
    overall = contrast.get("overall") or {}
    print(f"analyzed {analysis['n_runs']} runs "
          f"({len(analysis['cells'])} cells) in {args.run_dir}")
    if overall:
        pos, neg = contrast["strategies"]
        print(f"primary contrast {pos} - {neg}: "
              f"{overall['difference']:+.3f} "
              f"[{overall['ci95'][0]:+.3f}, {overall['ci95'][1]:+.3f}]")
    print("wrote analysis.json, SUMMARY.md and figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
