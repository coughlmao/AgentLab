"""Deterministic replay for config-driven experiment runs.

    python -m teamscale.replay results/<experiment>/<run-id>

Replay verifies that a published run directory is internally honest --
that its recorded metrics are exactly what its recorded artifacts imply:

- grading: every run's task is regenerated from its seed (task
  generation is deterministic), the run's saved final sandbox is taken
  as-is, and every check is RE-EXECUTED. The replayed fraction correct
  and success flag must match metrics.csv exactly.
- accounting: token totals are re-summed from the model_call events in
  traces.jsonl and must match metrics.csv exactly; each run's cost is
  recomputed from those tokens at the manifest's recorded pricing and
  must match to rounding.

What replay does NOT do: it never re-runs the model. Live model calls
are not replayable -- the model's sampling is not recorded and would not
reproduce -- so replay verifies the deterministic half of the pipeline
(grading and accounting) from the artifacts the run saved. If a metric
in the CSV was edited, a sandbox file was changed after the run, or a
trace was dropped, replay exits non-zero and prints the diff.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .real_tasks import check_all, make_real_task

REPLAY_VERSION = "3"  # 3: + model_calibrated_generation (Phase A-3)
COST_TOLERANCE = 1e-6  # recorded costs are rounded to 6 decimals


def _trace_segments(run_dir: Path) -> dict:
    """run_id -> events, split at each run_start in traces.jsonl."""
    segments: dict[str, list] = {}
    current = None
    path = run_dir / "traces.jsonl"
    if not path.exists():
        return segments
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "run_start":
            current = event["run_id"]
            segments[current] = []
        if current is not None:
            segments[current].append(event)
    return segments


def replay_run(run_dir) -> list:
    """Return a list of mismatch strings; empty means the run replays
    clean."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    metrics_path = run_dir / "metrics.csv"
    if not manifest_path.exists() or not metrics_path.exists():
        return [f"{run_dir}: manifest.json and metrics.csv are required"]
    manifest = json.loads(manifest_path.read_text())
    backend = manifest["provider"]
    difficulty = manifest.get("difficulty", "standard")
    pricing = manifest.get("pricing") or {}
    pin = float(pricing.get("input_per_1m_usd", 0.0))
    pout = float(pricing.get("output_per_1m_usd", 0.0))

    with open(metrics_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    segments = _trace_segments(run_dir)

    diffs = []
    if (run_dir / "hashes.json").exists():
        from .sealing import verify_run_dir
        diffs.extend(verify_run_dir(run_dir))
    family = (manifest.get("family")
              or (manifest.get("config") or {}).get("family")
              or "codebase_repair")
    if family == "error_accumulation":
        return diffs + _replay_chains(run_dir, manifest, rows, segments)
    if family == "model_calibrated_generation":
        return diffs + _replay_calibration(run_dir, manifest,
                                           segments)
    seen = set()
    total_metric_tokens = 0
    total_trace_tokens = 0
    for row in rows:
        run_id = (f"{backend}-codebase_repair-{row['strategy']}"
                  f"-n{row['team_size']}-s{row['seed']}")
        seen.add(run_id)
        tin_csv, tout_csv = int(row["tokens_in"]), int(row["tokens_out"])
        total_metric_tokens += tin_csv + tout_csv

        events = segments.get(run_id)
        if events is None:
            diffs.append(f"{run_id}: no trace segment in traces.jsonl")
            continue
        tin = sum(e.get("tokens_in", 0) for e in events
                  if e.get("event") == "model_call")
        tout = sum(e.get("tokens_out", 0) for e in events
                   if e.get("event") == "model_call")
        total_trace_tokens += tin + tout
        if (tin, tout) != (tin_csv, tout_csv):
            diffs.append(f"{run_id}: trace tokens ({tin} in / {tout} out) "
                         f"!= metrics.csv ({tin_csv} in / {tout_csv} out)")

        cost = tin / 1e6 * pin + tout / 1e6 * pout
        if abs(cost - float(row["cost_usd"])) > COST_TOLERANCE:
            diffs.append(f"{run_id}: recomputed cost ${cost:.6f} != "
                         f"metrics.csv ${float(row['cost_usd']):.6f}")

        sandbox = run_dir / "sandboxes" / run_id
        if not sandbox.exists():
            diffs.append(f"{run_id}: saved sandbox missing ({sandbox})")
            continue
        task = make_real_task(seed=int(row["seed"]), difficulty=difficulty)
        final = check_all(task, sandbox)
        fraction = round(sum(1 for v in final.values() if v)
                         / len(final), 4)
        if fraction != float(row["fraction_correct"]):
            diffs.append(f"{run_id}: replayed fraction correct {fraction} "
                         f"!= metrics.csv {row['fraction_correct']}")
        success = all(final.values())
        recorded_success = row["success"] in ("1", "true", "True")
        if success != recorded_success:
            diffs.append(f"{run_id}: replayed success {success} != "
                         f"metrics.csv {row['success']}")

    for run_id in sorted(set(segments) - seen):
        diffs.append(f"{run_id}: trace segment has no metrics.csv row")
    if total_metric_tokens != total_trace_tokens:
        diffs.append(f"total tokens: metrics.csv {total_metric_tokens} != "
                     f"traces {total_trace_tokens}")
    return diffs


def _replay_chains(run_dir: Path, manifest: dict, rows: list,
                   segments: dict) -> list:
    """Replay for the chained family: re-derive every step's grading
    from the recorded input/submitted states and the regenerated
    chain, re-sum tokens from traces, and cross-check steps.csv,
    recovery_events.jsonl, metrics.csv, and the final sandbox state."""
    from .chains import (GENERATOR_VERSION, grade_step, make_chain,
                         parse_state_file)
    diffs = []
    backend = manifest["provider"]
    pricing = manifest.get("pricing") or {}
    pin = float(pricing.get("input_per_1m_usd", 0.0))
    pout = float(pricing.get("output_per_1m_usd", 0.0))
    if manifest.get("generator_version") != GENERATOR_VERSION:
        diffs.append(
            f"manifest generator_version "
            f"{manifest.get('generator_version')} != this code's "
            f"{GENERATOR_VERSION}: grading cannot be replayed")
        return diffs

    steps_rows: dict[str, list] = {}
    if (run_dir / "steps.csv").exists():
        with open(run_dir / "steps.csv", newline="") as fh:
            for srow in csv.DictReader(fh):
                steps_rows.setdefault(srow["run_id"], []).append(srow)
    else:
        diffs.append("steps.csv missing from a chained-family run")
    recovery_counts: dict[str, int] = {}
    if (run_dir / "recovery_events.jsonl").exists():
        for line in (run_dir / "recovery_events.jsonl") \
                .read_text().splitlines():
            if line.strip():
                ev = json.loads(line)
                recovery_counts[ev["run_id"]] = \
                    recovery_counts.get(ev["run_id"], 0) + 1

    seen = set()
    for row in rows:
        run_id = (f"{backend}-error_accumulation-{row['strategy']}"
                  f"-h{row['horizon']}-s{row['seed']}")
        seen.add(run_id)
        events = segments.get(run_id)
        if events is None:
            diffs.append(f"{run_id}: no trace segment in traces.jsonl")
            continue
        tin = sum(e.get("tokens_in", 0) for e in events
                  if e.get("event") == "model_call")
        tout = sum(e.get("tokens_out", 0) for e in events
                   if e.get("event") == "model_call")
        if (tin, tout) != (int(row["tokens_in"]), int(row["tokens_out"])):
            diffs.append(f"{run_id}: trace tokens ({tin} in / {tout} "
                         f"out) != metrics.csv ({row['tokens_in']} in "
                         f"/ {row['tokens_out']} out)")
        cost = tin / 1e6 * pin + tout / 1e6 * pout
        if abs(cost - float(row["cost_usd"])) > COST_TOLERANCE:
            diffs.append(f"{run_id}: recomputed cost ${cost:.6f} != "
                         f"metrics.csv ${float(row['cost_usd']):.6f}")

        result_path = run_dir / "results" / f"{run_id}.json"
        if not result_path.exists():
            diffs.append(f"{run_id}: per-run result JSON missing")
            continue
        result = json.loads(result_path.read_text())
        records = sorted(result["per_step"],
                         key=lambda r: r["step_index"])
        horizon = int(row["horizon"])
        if len(records) != horizon:
            diffs.append(f"{run_id}: {len(records)} step records != "
                         f"horizon {horizon}")
            continue
        task = make_chain(seed=int(row["seed"]), n_steps=horizon)
        srows = sorted(steps_rows.get(run_id, []),
                       key=lambda r: int(r["step_index"]))
        if len(srows) != horizon:
            diffs.append(f"{run_id}: steps.csv has {len(srows)} rows "
                         f"!= horizon {horizon}")
        n_state_ok = 0
        for rec, srow in zip(records, srows):
            step = task.steps[rec["step_index"] - 1]
            state_ok, local_ok, _e, _m = grade_step(
                step, rec["input_state"], rec["submitted_state"])
            n_state_ok += 1 if state_ok else 0
            if state_ok != rec["state_correct"] or \
                    local_ok != rec["local_correct"]:
                diffs.append(
                    f"{run_id} step {rec['step_index']}: replayed "
                    f"grading (state={state_ok}, local={local_ok}) != "
                    f"recorded (state={rec['state_correct']}, "
                    f"local={rec['local_correct']})")
            if int(srow["state_correct"]) != int(state_ok) or \
                    int(srow["local_correct"]) != int(local_ok):
                diffs.append(
                    f"{run_id} step {rec['step_index']}: steps.csv "
                    f"disagrees with replayed grading")
        fraction = round(n_state_ok / horizon, 4)
        if fraction != float(row["fraction_correct"]):
            diffs.append(f"{run_id}: replayed fraction correct "
                         f"{fraction} != metrics.csv "
                         f"{row['fraction_correct']}")
        final_ok = records[-1]["state_correct"]
        if int(row["final_state_correct"]) != int(final_ok):
            diffs.append(f"{run_id}: final_state_correct mismatch")
        if (row["success"] in ("1", "true", "True")) != bool(final_ok):
            diffs.append(f"{run_id}: success flag mismatch")
        sandbox_state = run_dir / "sandboxes" / run_id / "state.py"
        if not sandbox_state.exists():
            diffs.append(f"{run_id}: saved sandbox state.py missing")
        elif parse_state_file(sandbox_state.read_text()) != \
                records[-1]["submitted_state"]:
            diffs.append(f"{run_id}: sandbox final state != last "
                         f"recorded submission")
        verify_events = sum(1 for e in events
                            if e.get("event") == "verify")
        if verify_events != recovery_counts.get(run_id, 0):
            diffs.append(f"{run_id}: {verify_events} verify events in "
                         f"traces != {recovery_counts.get(run_id, 0)} "
                         f"recovery events logged")
    for run_id in sorted(set(segments) - seen):
        diffs.append(f"{run_id}: trace segment has no metrics.csv row")
    return diffs


def _replay_calibration(run_dir: Path, manifest: dict,
                        segments: dict) -> list:
    """Replay for Phase A-3 (model-calibrated generation): regenerate
    every probe step from its (seed, setting), re-grade the saved
    sandbox submission, re-sum tokens from traces, recompute costs at
    the manifest pricing, and cross-check probes.csv against
    metrics.csv and every number in calibration.json (control trace,
    validation verdicts, monotonicity table)."""
    from .chains import parse_state_file
    from .ladder import (GENERATOR_VERSION, grade_probe_step,
                         make_probe_step)
    from .stats import wilson_interval
    diffs = []
    pricing = manifest.get("pricing") or {}
    pin = float(pricing.get("input_per_1m_usd", 0.0))
    pout = float(pricing.get("output_per_1m_usd", 0.0))
    if manifest.get("generator_version") != GENERATOR_VERSION:
        diffs.append(
            f"manifest generator_version "
            f"{manifest.get('generator_version')} != this code's "
            f"{GENERATOR_VERSION}: grading cannot be replayed")
        return diffs
    probes_path = run_dir / "probes.csv"
    if not probes_path.exists():
        return diffs + ["probes.csv missing from a "
                        "model_calibrated_generation run"]
    with open(probes_path, newline="") as fh:
        probes = list(csv.DictReader(fh))

    def agg(rows):
        scored = [r for r in rows
                  if r["outcome"] in ("correct", "incorrect")]
        n = len(scored)
        failures = sum(1 for r in scored
                       if r["outcome"] == "incorrect")
        return n, failures, len(rows) - n

    seen = set()
    for row in probes:
        probe_id = row["probe_id"]
        seen.add(probe_id)
        events = segments.get(probe_id)
        if events is None:
            diffs.append(f"{probe_id}: no trace segment in "
                         "traces.jsonl")
            continue
        tin = sum(e.get("tokens_in", 0) for e in events
                  if e.get("event") == "model_call")
        tout = sum(e.get("tokens_out", 0) for e in events
                   if e.get("event") == "model_call")
        if (tin, tout) != (int(row["tokens_in"]),
                           int(row["tokens_out"])):
            diffs.append(f"{probe_id}: trace tokens ({tin} in / "
                         f"{tout} out) != probes.csv "
                         f"({row['tokens_in']} in / "
                         f"{row['tokens_out']} out)")
        cost = tin / 1e6 * pin + tout / 1e6 * pout
        if abs(cost - float(row["cost_usd"])) > COST_TOLERANCE:
            diffs.append(f"{probe_id}: recomputed cost ${cost:.6f} "
                         f"!= probes.csv ${float(row['cost_usd']):.6f}")
        if row["outcome"] == "error":
            continue  # infrastructure outcome: nothing to re-grade
        step = make_probe_step(int(row["seed"]), row["setting"])
        sandbox_state = run_dir / "sandboxes" / probe_id / "state.py"
        if not sandbox_state.exists():
            diffs.append(f"{probe_id}: saved sandbox state.py "
                         "missing")
            continue
        submitted = parse_state_file(sandbox_state.read_text())
        correct, error_class, _m = grade_probe_step(step, submitted)
        want = "correct" if correct else "incorrect"
        if want != row["outcome"]:
            diffs.append(f"{probe_id}: replayed outcome {want} != "
                         f"probes.csv {row['outcome']}")
        if (row["error_class"] or None) != error_class:
            diffs.append(f"{probe_id}: replayed error_class "
                         f"{error_class} != probes.csv "
                         f"{row['error_class']}")
    for probe_id in sorted(set(segments) - seen):
        diffs.append(f"{probe_id}: trace segment has no probes.csv "
                     "row")

    # metrics.csv aggregates vs probes.csv
    with open(run_dir / "metrics.csv", newline="") as fh:
        for mrow in csv.DictReader(fh):
            cell = [r for r in probes
                    if r["phase"] == mrow["phase"]
                    and float(r["band_target"])
                    == float(mrow["band_target"])
                    and r["setting"] == mrow["setting"]]
            n, failures, errors = agg(cell)
            if (n, failures, errors) != (int(mrow["n_scored"]),
                                         int(mrow["failures"]),
                                         int(mrow["n_errors"])):
                diffs.append(
                    f"metrics.csv {mrow['phase']}/"
                    f"{mrow['band_target']}/{mrow['setting']}: "
                    f"recomputed (n={n}, failures={failures}, "
                    f"errors={errors}) != recorded "
                    f"({mrow['n_scored']}, {mrow['failures']}, "
                    f"{mrow['n_errors']})")

    # calibration.json cross-checks
    cal_path = run_dir / "calibration.json"
    if not cal_path.exists():
        return diffs + ["calibration.json missing"]
    cal = json.loads(cal_path.read_text())
    for band in cal.get("bands", []):
        target = band["target"]
        for entry in band.get("trace", []):
            if entry.get("n") is None:
                continue
            cell = [r for r in probes
                    if r["phase"] == "search"
                    and float(r["band_target"]) == target
                    and r["setting"] == entry["setting"]]
            n, failures, _e = agg(cell)
            if (n, failures) != (entry["n"], entry["failures"]):
                diffs.append(
                    f"band {band['label']} trace "
                    f"{entry['setting']}: recomputed (n={n}, "
                    f"failures={failures}) != calibration.json "
                    f"({entry['n']}, {entry['failures']})")
        val = band.get("validation") or {}
        if val.get("status") == "completed":
            setting = band["setting"]["name"]
            cell = [r for r in probes
                    if r["phase"] == "validation"
                    and float(r["band_target"]) == target
                    and r["setting"] == setting]
            n, failures, errors = agg(cell)
            rate = failures / n if n else None
            ci_lo, ci_hi = wilson_interval(failures, n)
            band_lo = target - band["tolerance"]
            band_hi = target + band["tolerance"]
            point_in = rate is not None and band_lo <= rate <= band_hi
            intersects = ci_lo <= band_hi and ci_hi >= band_lo
            recomputed = {"n": n, "failures": failures,
                          "errors": errors,
                          "rate": (round(rate, 6)
                                   if rate is not None else None),
                          "ci_lo": round(ci_lo, 6),
                          "ci_hi": round(ci_hi, 6),
                          "point_estimate_in_band": bool(point_in),
                          "ci_intersects_band": bool(intersects),
                          "pass": bool(point_in and intersects)}
            for key, want in recomputed.items():
                if val.get(key) != want:
                    diffs.append(
                        f"band {band['label']} validation {key}: "
                        f"recomputed {want} != calibration.json "
                        f"{val.get(key)}")
    mono = cal.get("monotonicity") or {}
    for row in mono.get("per_setting", []):
        cell = [r for r in probes if r["setting"] == row["setting"]]
        n, failures, _e = agg(cell)
        if (n, failures) != (row["n"], row["failures"]):
            diffs.append(
                f"monotonicity {row['setting']}: recomputed (n={n}, "
                f"failures={failures}) != calibration.json "
                f"({row['n']}, {row['failures']})")
    return diffs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="teamscale.replay",
        description="deterministically replay a config-driven run "
                    "directory (grading + accounting, no model calls)")
    ap.add_argument("run_dir", help="results/<experiment>/<run-id>")
    args = ap.parse_args(argv)
    diffs = replay_run(args.run_dir)
    if diffs:
        print(f"REPLAY FAILED: {len(diffs)} mismatch(es) in {args.run_dir}")
        for d in diffs:
            print(f"  - {d}")
        return 1
    print(f"replay ok: {args.run_dir} -- grading and token accounting "
          "match the recorded artifacts exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
