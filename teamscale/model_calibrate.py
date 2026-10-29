"""Phase A-3: model-calibrated benchmark generation -- the
closed-loop calibration controller and the experiment driver.

    (driven through teamscale.config_experiment with family:
    model_calibrated_generation -- see
    experiments/model-calibrated-generation/)

The calibration is a feedback controller, not a search (DESIGN.md):
the setpoint is a target first-attempt failure band; the error
signal is the signed distance of a probed batch's Wilson CI from the
band; the control law is bisection over the difficulty ladder
(teamscale.ladder) with a shrinking interval; convergence freezes a
setting as a named calibration. Exhausting the bounded rounds
(max_settings probed per band) without convergence is a DECLARED
FAILURE of the controller for that band -- the pre-registered
tolerances are never widened, here or anywhere.

Controller properties are first-class results: per band we record
the full control trace and the calibration efficiency (iterations,
probes, API calls, tokens, wall-clock, dollars), because calibration
is an optimization problem and a cheaper controller that reaches the
same band is the better algorithm.

Held-out validation: a converged setting is frozen, then probed on
validation_n fresh steps whose seeds were never used during the
search. Success (pre-registered): the point estimate falls inside
the band tolerance AND the Wilson CI intersects the band.

Spend discipline: one SpendLedger gates the whole run at the config
cap. Before every batch, the gate checks actual spent + the batch's
strict worst case (every probe burning its full token budget plus
one call of overshoot, all at the higher token price) against the
cap, so a batch that could breach the cap never starts.

Summaries are computed from the probe records table (probes.csv)
only -- never from traces or sandboxes (Invariant 4). Replay
(teamscale.replay) independently re-derives them.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from . import ladder as ladder_mod
from .ladder import LADDER
from .probe import band_label, run_probe_batch
from .real_pilot import PRICES, SpendLedger
from .stats import wilson_interval

MODEL_CALIBRATION_VERSION = "model-calibration-1"
FAMILY = "model_calibrated_generation"
SEARCH_SEED_BASE = 700_000        # per band: + band_index * 100_000,
                                  # per setting: + rung * 1_000
VALIDATION_SEED_BASE = 900_000    # per band: + band_index * 10_000
_OVERSHOOT_TOKENS = 8_000         # one model call past the budget

PROBES_FIELDS = ["probe_id", "phase", "band_target", "setting",
                 "seed", "outcome", "correct", "error_class",
                 "tokens_in", "tokens_out", "model_calls", "cost_usd",
                 "latency_s", "error"]
METRICS_FIELDS = ["phase", "band_target", "setting", "n_probes",
                  "n_scored", "n_errors", "failures", "rate",
                  "tokens_in", "tokens_out", "cost_usd"]


# ------------------------------------------------------------ statistics
def batch_stats(records: list) -> dict:
    """Failure-rate statistics over one batch of probe records.
    Error (infrastructure) probes are excluded from the rate and
    counted separately."""
    scored = [r for r in records if r["outcome"] in ("correct",
                                                     "incorrect")]
    n = len(scored)
    failures = sum(1 for r in scored if r["outcome"] == "incorrect")
    ci_lo, ci_hi = wilson_interval(failures, n)
    return {"n": n, "failures": failures,
            "errors": len(records) - n,
            "rate": (failures / n) if n else None,
            "ci_lo": ci_lo, "ci_hi": ci_hi}


def efficiency(records: list, wall_s: float,
               iterations: int | None = None) -> dict:
    out = {"probes": len(records),
           "api_calls": sum(r["model_calls"] for r in records),
           "tokens_in": sum(r["tokens_in"] for r in records),
           "tokens_out": sum(r["tokens_out"] for r in records),
           "cost_usd": round(sum(r["cost_usd"] for r in records), 6),
           "wall_clock_s": round(wall_s, 3)}
    if iterations is not None:
        out["iterations"] = iterations
    return out


# ------------------------------------------------------------- controller
def band_error(ci_lo: float, ci_hi: float,
               band_lo: float, band_hi: float) -> float:
    """The controller's error signal: signed distance of the CI from
    the band (negative = CI entirely below, positive = entirely
    above, 0 = CI intersects the band)."""
    if ci_hi < band_lo:
        return round(ci_hi - band_lo, 6)
    if ci_lo > band_hi:
        return round(ci_lo - band_hi, 6)
    return 0.0


def _converged(stats: dict, band_lo: float, band_hi: float) -> bool:
    """Pre-registered convergence: point estimate inside the band
    tolerance AND Wilson CI intersects the band -- the same
    criterion held-out validation must pass."""
    if stats["rate"] is None:
        return False
    return (band_lo <= stats["rate"] <= band_hi
            and stats["ci_lo"] <= band_hi
            and stats["ci_hi"] >= band_lo)


def run_controller(*, target: float, tolerance: float, probe_setting,
                   max_settings: int = 6, ladder=LADDER) -> dict:
    """Bisect the ladder for one band.

    probe_setting(setting) -> batch stats dict (batch_stats shape);
    it may return {"aborted": True, ...} when an outer gate (spend
    cap) refuses the batch, which aborts the controller cleanly.
    Pure logic: no I/O of its own, so tests drive it with synthetic
    probe functions."""
    band_lo, band_hi = target - tolerance, target + tolerance
    lo, hi = 0, len(ladder) - 1
    trace = []
    probed: dict[int, dict] = {}
    converged_index = None
    aborted = False
    while len(probed) < max_settings and lo <= hi:
        mid = (lo + hi) // 2
        if mid in probed:
            break
        setting = ladder[mid]
        stats = probe_setting(setting)
        if stats.get("aborted"):
            trace.append({
                "iteration": len(probed) + 1, "setting_index": mid,
                "setting": setting.name, "n": 0, "failures": 0,
                "rate": None, "ci_lo": None, "ci_hi": None,
                "error": None,
                "adjustment": "aborted: outer gate refused the batch"})
            aborted = True
            break
        probed[mid] = stats
        err = band_error(stats["ci_lo"], stats["ci_hi"],
                         band_lo, band_hi)
        entry = {"iteration": len(probed), "setting_index": mid,
                 "setting": setting.name, "n": stats["n"],
                 "failures": stats["failures"],
                 "errors": stats.get("errors", 0),
                 "rate": (round(stats["rate"], 6)
                          if stats["rate"] is not None else None),
                 "ci_lo": round(stats["ci_lo"], 6),
                 "ci_hi": round(stats["ci_hi"], 6),
                 "error": err}
        if _converged(stats, band_lo, band_hi):
            converged_index = mid
            entry["adjustment"] = "converged: freeze this setting"
            trace.append(entry)
            break
        if stats["rate"] is None:
            entry["adjustment"] = "no scored probes: cannot adjust"
            trace.append(entry)
            break
        if stats["ci_hi"] < band_lo or stats["rate"] < target:
            lo = mid + 1
            entry["adjustment"] = (f"move up: search [{lo}, {hi}]"
                                   if lo <= hi else
                                   "move up: ladder exhausted")
        else:
            hi = mid - 1
            entry["adjustment"] = (f"move down: search [{lo}, {hi}]"
                                   if lo <= hi else
                                   "move down: ladder exhausted")
        trace.append(entry)

    if aborted:
        status, failure_class = "aborted_spend_cap", None
    elif converged_index is not None:
        status, failure_class = "converged", None
    else:
        status = "declared_failure"
        top = max(probed) if probed else None
        bottom = min(probed) if probed else None
        if (top is not None and top == len(ladder) - 1
                and probed[top]["ci_hi"] < band_lo):
            failure_class = "saturated_below"
        elif (bottom is not None and bottom == 0
                and probed[bottom]["ci_lo"] > band_hi):
            failure_class = "overshoot_at_floor"
        else:
            failure_class = "no_stable_setting"
    return {"status": status, "failure_class": failure_class,
            "setting_index": converged_index,
            "setting": (ladder[converged_index].name
                        if converged_index is not None else None),
            "trace": trace, "iterations": len(probed)}


def monotonicity_report(records: list) -> dict:
    """Pool every scored probe by ladder setting (search + validation,
    both bands: iid samples of the same per-setting quantity) and
    check the ladder's monotonicity claim on the observed rates."""
    by_setting: dict[str, list] = {}
    for r in records:
        if r["outcome"] in ("correct", "incorrect"):
            by_setting.setdefault(r["setting"], []).append(r)
    per_setting = []
    for setting in LADDER:
        recs = by_setting.get(setting.name, [])
        if not recs:
            continue
        stats = batch_stats(recs)
        per_setting.append({
            "setting_index": ladder_mod.setting_index(setting.name),
            "setting": setting.name, "n": stats["n"],
            "failures": stats["failures"],
            "rate": (round(stats["rate"], 6)
                     if stats["rate"] is not None else None),
            "ci_lo": round(stats["ci_lo"], 6),
            "ci_hi": round(stats["ci_hi"], 6)})
    inversions = []
    for a, b in zip(per_setting, per_setting[1:]):
        if a["rate"] is not None and b["rate"] is not None \
                and b["rate"] < a["rate"]:
            inversions.append({
                "from": a["setting"], "to": b["setting"],
                "rate_from": a["rate"], "rate_to": b["rate"],
                "ci_overlap": b["ci_hi"] >= a["ci_lo"]})
    hard = [i for i in inversions if not i["ci_overlap"]]
    if hard:
        verdict = "non-monotone"
    elif inversions:
        verdict = "point-estimate inversions within noise"
    else:
        verdict = "monotone non-decreasing"
    return {"per_setting": per_setting, "inversions": inversions,
            "verdict": verdict}


# ----------------------------------------------------------------- driver
def _strict_batch_worst(n_probes: int, token_budget: int,
                        model: str) -> float:
    pin, pout = PRICES.get(model, (0.0, 0.0))
    per_probe = (token_budget + _OVERSHOOT_TOKENS) \
        * max(pin, pout) / 1e6
    return n_probes * per_probe


def run_model_calibration_bundle(*, spec, run_dir: Path, run_id: str,
                                   model: str, provider: str,
                                   factory) -> dict:
    """Execute the A-3 experiment into an already-created run
    directory (manifest written by the caller): controller search +
    held-out validation per band, then the bundle's tables,
    calibration.json, summary.json, and SUMMARY.md. Returns the
    summary dict."""
    run_dir = Path(run_dir)
    ledger_name = ("ledger_fake.json" if provider == "fake"
                   else "ledger.json")
    ledger = SpendLedger(run_dir / ledger_name, spec.spend_cap_usd,
                         model)
    all_records: list[dict] = []
    issues: list[str] = []
    bands_out = []
    started = time.perf_counter()

    for bi, (target, tolerance) in enumerate(
            zip(spec.band_targets, spec.band_tolerances)):
        label = band_label(target)
        band_lo, band_hi = target - tolerance, target + tolerance
        search_records: list[dict] = []
        band_started = time.perf_counter()

        def probe_setting(setting, _bi=bi, _label=label,
                          _target=target, _search=search_records):
            rung = ladder_mod.setting_index(setting.name)
            worst = _strict_batch_worst(spec.batch_size,
                                        spec.per_probe_token_cap,
                                        model)
            if ledger.spent_usd + worst > spec.spend_cap_usd + 1e-9:
                return {"aborted": True}
            seeds = [SEARCH_SEED_BASE + _bi * 100_000 + rung * 1_000
                     + j for j in range(spec.batch_size)]
            backend = factory({"seed": seeds[0], "team_size": 1,
                               "blind": True})
            records = run_probe_batch(
                backend=backend, setting_name=setting.name,
                seeds=seeds, label=_label, target=_target,
                phase="search", out_dir=run_dir,
                token_budget=spec.per_probe_token_cap, model=model,
                workers=spec.workers)
            for r in records:
                ledger.record({"run_id": r["probe_id"],
                               "cost_usd": r["cost_usd"],
                               "tokens_in": r["tokens_in"],
                               "tokens_out": r["tokens_out"]})
            _search.extend(records)
            all_records.extend(records)
            return batch_stats(records)

        result = run_controller(target=target, tolerance=tolerance,
                                probe_setting=probe_setting,
                                max_settings=spec.max_settings)
        search_wall = time.perf_counter() - band_started
        band_entry = {
            "target": target, "tolerance": tolerance,
            "band": [round(band_lo, 6), round(band_hi, 6)],
            "label": label,
            "status": result["status"],
            "failure_class": result["failure_class"],
            "calibration_version": None,
            "setting": None,
            "trace": result["trace"],
            "efficiency": efficiency(search_records, search_wall,
                                     iterations=result["iterations"]),
            "validation": None,
            "validation_efficiency": None,
        }
        if result["status"] == "aborted_spend_cap":
            issues.append(
                f"band {label}: controller aborted by the spend-cap "
                f"gate after {result['iterations']} settings "
                f"(spent ${ledger.spent_usd:.4f} of "
                f"${spec.spend_cap_usd:.2f})")
        if result["status"] == "converged":
            setting = LADDER[result["setting_index"]]
            band_entry["setting"] = setting.to_dict()
            band_entry["calibration_version"] = (
                f"band-{label}@{ladder_mod.GENERATOR_VERSION}"
                f":{setting.name}")
            val_started = time.perf_counter()
            worst = _strict_batch_worst(spec.validation_n,
                                        spec.per_probe_token_cap,
                                        model)
            if ledger.spent_usd + worst > spec.spend_cap_usd + 1e-9:
                issues.append(
                    f"band {label}: validation batch refused by the "
                    f"spend-cap gate (spent ${ledger.spent_usd:.4f}); "
                    "the converged setting is reported WITHOUT "
                    "held-out validation")
                band_entry["validation"] = {
                    "status": "refused_spend_cap"}
            else:
                seeds = [VALIDATION_SEED_BASE + bi * 10_000 + j
                         for j in range(spec.validation_n)]
                backend = factory({"seed": seeds[0], "team_size": 1,
                                   "blind": True})
                records = run_probe_batch(
                    backend=backend, setting_name=setting.name,
                    seeds=seeds, label=label, target=target,
                    phase="validation", out_dir=run_dir,
                    token_budget=spec.per_probe_token_cap, model=model,
                    workers=spec.workers)
                for r in records:
                    ledger.record({"run_id": r["probe_id"],
                                   "cost_usd": r["cost_usd"],
                                   "tokens_in": r["tokens_in"],
                                   "tokens_out": r["tokens_out"]})
                all_records.extend(records)
                stats = batch_stats(records)
                point_in = (stats["rate"] is not None
                            and band_lo <= stats["rate"] <= band_hi)
                intersects = (stats["ci_lo"] <= band_hi
                              and stats["ci_hi"] >= band_lo)
                band_entry["validation"] = {
                    "status": "completed",
                    "n": stats["n"], "failures": stats["failures"],
                    "errors": stats["errors"],
                    "rate": (round(stats["rate"], 6)
                             if stats["rate"] is not None else None),
                    "ci_lo": round(stats["ci_lo"], 6),
                    "ci_hi": round(stats["ci_hi"], 6),
                    "point_estimate_in_band": bool(point_in),
                    "ci_intersects_band": bool(intersects),
                    "pass": bool(point_in and intersects),
                    "held_out_seed_base": VALIDATION_SEED_BASE
                    + bi * 10_000,
                }
                band_entry["validation_efficiency"] = efficiency(
                    records, time.perf_counter() - val_started)
        bands_out.append(band_entry)

    mono = monotonicity_report(all_records)
    total_wall = time.perf_counter() - started
    n_errors = sum(1 for r in all_records if r["outcome"] == "error")
    if n_errors:
        issues.append(
            f"{n_errors} probes ended as infrastructure errors and "
            "were excluded from failure-rate statistics")

    _write_probes_csv(run_dir, all_records)
    _write_metrics_csv(run_dir, all_records)
    _assemble_traces(run_dir, all_records)
    (run_dir / "figures").mkdir(exist_ok=True)

    calibration = {
        "version": MODEL_CALIBRATION_VERSION,
        "generator_version": ladder_mod.GENERATOR_VERSION,
        "family": FAMILY, "provider": provider, "model": model,
        "control_law": "bisection over the ladder, error = signed "
                       "distance of the batch Wilson CI from the "
                       "band",
        "batch_size": spec.batch_size,
        "max_settings_per_band": spec.max_settings,
        "validation_n": spec.validation_n,
        "bands": bands_out,
        "monotonicity": mono,
        "totals": efficiency(all_records, total_wall),
        "issues": issues,
    }
    (run_dir / "calibration.json").write_text(
        json.dumps(calibration, indent=1))

    totals = calibration["totals"]
    summary = {
        "run_id": run_id, "family": FAMILY, "provider": provider,
        "model": model, "blind": True,
        "generator_version": ladder_mod.GENERATOR_VERSION,
        "calibration_version": MODEL_CALIBRATION_VERSION,
        "bands": [{"label": b["label"], "target": b["target"],
                   "tolerance": b["tolerance"], "status": b["status"],
                   "failure_class": b["failure_class"],
                   "setting": (b["setting"] or {}).get("name"),
                   "validation_pass": (
                       (b["validation"] or {}).get("pass"))}
                  for b in bands_out],
        "monotonicity_verdict": mono["verdict"],
        "total_probes": totals["probes"],
        "total_api_calls": totals["api_calls"],
        "total_tokens_in": totals["tokens_in"],
        "total_tokens_out": totals["tokens_out"],
        "total_cost_usd": totals["cost_usd"],
        "ledger_spent_usd": ledger.spent_usd,
        "spend_cap_usd": spec.spend_cap_usd,
        "wall_clock_s": totals["wall_clock_s"],
        "issues": issues,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary,
                                                     indent=1))
    (run_dir / "SUMMARY.md").write_text(
        _render_summary_md(spec, summary, calibration))
    return summary


def _write_probes_csv(run_dir: Path, records: list) -> None:
    with open(run_dir / "probes.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PROBES_FIELDS)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "probe_id": r["probe_id"], "phase": r["phase"],
                "band_target": r["band_target"],
                "setting": r["setting"], "seed": r["seed"],
                "outcome": r["outcome"],
                "correct": ("" if r["correct"] is None
                            else int(r["correct"])),
                "error_class": r["error_class"] or "",
                "tokens_in": r["tokens_in"],
                "tokens_out": r["tokens_out"],
                "model_calls": r["model_calls"],
                "cost_usd": r["cost_usd"], "latency_s": r["latency_s"],
                "error": r["error"] or ""})


def _write_metrics_csv(run_dir: Path, records: list) -> None:
    """Aggregate table: one row per (phase, band, setting) probed --
    the run's metrics.csv, companion to the per-probe probes.csv."""
    groups: dict[tuple, list] = {}
    for r in records:
        groups.setdefault((r["phase"], r["band_target"],
                           r["setting"]), []).append(r)
    with open(run_dir / "metrics.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRICS_FIELDS)
        writer.writeheader()
        for (phase, target, setting), recs in sorted(groups.items()):
            stats = batch_stats(recs)
            writer.writerow({
                "phase": phase, "band_target": target,
                "setting": setting, "n_probes": len(recs),
                "n_scored": stats["n"], "n_errors": stats["errors"],
                "failures": stats["failures"],
                "rate": ("" if stats["rate"] is None
                         else round(stats["rate"], 6)),
                "tokens_in": sum(r["tokens_in"] for r in recs),
                "tokens_out": sum(r["tokens_out"] for r in recs),
                "cost_usd": round(sum(r["cost_usd"] for r in recs),
                                  6)})


def _assemble_traces(run_dir: Path, records: list) -> None:
    with open(run_dir / "traces.jsonl", "w") as out:
        for r in records:
            trace_file = run_dir / "traces" / f"{r['probe_id']}.jsonl"
            if trace_file.exists():
                text = trace_file.read_text()
                out.write(text)
                if not text.endswith("\n"):
                    out.write("\n")


def _render_summary_md(spec, summary: dict, calibration: dict) -> str:
    lines = [f"# {spec.name}", "",
             f"**Question:** {spec.question or '(not stated)'}", "",
             "## Caveats", ""]
    for caveat in spec.caveats:
        lines.append(f"- {caveat}")
    lines += ["", "## Design", "",
              f"- provider / model: {summary['provider']} / "
              f"{summary['model']}; blind single-attempt probes of "
              "standalone ladder steps",
              f"- ladder: {len(LADDER)} settings "
              f"(generator {calibration['generator_version']}); "
              f"control law: bisection, error = signed Wilson-CI "
              f"distance from the band; max {spec.max_settings} "
              f"settings per band, batches of {spec.batch_size}",
              f"- spend: ${summary['total_cost_usd']:.4f} "
              f"(ledger ${summary['ledger_spent_usd']:.4f}) of "
              f"${summary['spend_cap_usd']:.2f} cap; "
              f"{summary['total_probes']} probes, "
              f"{summary['total_api_calls']} API calls, wall "
              f"{summary['wall_clock_s']}s",
              "", "## Bands", ""]
    for band in calibration["bands"]:
        lines.append(f"### Band {band['label']}: target "
                     f"{band['target']:.0%} ± {band['tolerance']:.0%}")
        lines.append("")
        lines.append(f"- status: **{band['status']}**"
                     + (f" ({band['failure_class']})"
                        if band["failure_class"] else "")
                     + (f"; frozen setting "
                        f"`{band['setting']['name']}` "
                        f"(calibration {band['calibration_version']})"
                        if band["setting"] else ""))
        eff = band["efficiency"]
        lines.append(f"- search efficiency: {eff['iterations']} "
                     f"iterations, {eff['probes']} probes, "
                     f"{eff['api_calls']} API calls, "
                     f"{eff['tokens_in'] + eff['tokens_out']} tokens, "
                     f"${eff['cost_usd']:.4f}, {eff['wall_clock_s']}s")
        for entry in band["trace"]:
            lines.append(
                f"  - iter {entry['iteration']}: "
                f"`{entry['setting']}` n={entry['n']} "
                f"failures={entry['failures']} "
                f"rate={entry['rate']} CI=[{entry['ci_lo']}, "
                f"{entry['ci_hi']}] error={entry['error']} -> "
                f"{entry['adjustment']}")
        val = band["validation"]
        if val and val.get("status") == "completed":
            lines.append(
                f"- held-out validation (n={val['n']}): rate "
                f"{val['rate']} CI=[{val['ci_lo']}, {val['ci_hi']}]; "
                f"point in band: {val['point_estimate_in_band']}; "
                f"CI intersects band: {val['ci_intersects_band']}; "
                f"**pass: {val['pass']}**")
        elif val:
            lines.append(f"- held-out validation: {val['status']}")
        lines.append("")
    mono = calibration["monotonicity"]
    lines += ["## Ladder monotonicity (pooled observed rates)", "",
              f"Verdict: **{mono['verdict']}**", ""]
    for row in mono["per_setting"]:
        lines.append(f"- `{row['setting']}`: rate {row['rate']} "
                     f"[{row['ci_lo']}, {row['ci_hi']}] "
                     f"(n={row['n']})")
    if summary["issues"]:
        lines += ["", "## Issues", ""]
        for issue in summary["issues"]:
            lines.append(f"- {issue}")
    lines.append("")
    return "\n".join(lines)
