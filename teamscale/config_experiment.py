"""Config-driven experiments: one YAML file in, one immutable run
directory out.

    python -m teamscale.experiment experiments/verification-question/experiment.yaml
    python -m teamscale.config_experiment <experiment.yaml> [--fake]

An experiment config states the whole design up front -- name, research
question, provider, model, difficulty, blind mode, grid axes, caps -- so
an experiment is reproducible from the file alone. Running it creates a
FRESH directory

    <output_root>/<experiment-name>/<run-id>/

(run-id = UTC timestamp + short config hash) and refuses to write into a
run directory that already exists: published results are never silently
overwritten, and a re-run is a new directory next to the old one.

The run directory is the publishable unit:

    manifest.json   the config verbatim + its sha256, the git commit and
                    code version that produced the run, provider, model,
                    pricing, grid, timestamps, minimal host info. Written
                    BEFORE the grid runs and never modified afterwards.
    metrics.csv     one row per run (strategy, team_size, seed, fraction
                    correct, success, tokens in/out, cost, retries, wall
                    seconds) -- the single source the analysis reads.
    traces.jsonl    every run's trace events (same shapes as the pilot's
                    per-run trace files), concatenated per run.
    traces/         the per-run trace files, as written by the runner.
    results/        the per-run result JSONs, as written by the runner.
    sandboxes/      the final sandbox of every run (what replay grades).
    ledger.json     the spend ledger the cap gate enforced.
    summary.json    the pilot summary + a cost-accounting block.
    SUMMARY.md      question, design, and cost accounting in prose.
    figures/        filled by teamscale.analyze from metrics.csv alone.

Execution itself is the existing pilot machinery (teamscale.real_pilot):
same strategies, same spend-cap gate (a run starts only if the ledger
plus its worst-case estimate fits under the cap), same traces. This
module is the reproducibility shell around it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .analyze import (ANALYSIS_VERSION, CHAIN_ANALYSIS_DEFAULT,
                      DEFAULT_ANALYSIS)
from .anthropic_backend import AnthropicLLMBackend
from .calibrate import CALIBRATION_VERSION
from .chains import GENERATOR_VERSION as CHAIN_GENERATOR_VERSION
from .fake_backend import FakeLLMBackend
from .ladder import GENERATOR_VERSION as LADDER_GENERATOR_VERSION
from .llm_backend import OpenAILLMBackend, prompt_hash as repair_prompt_hash
from .model_calibrate import MODEL_CALIBRATION_VERSION
from .probe import prompt_hash as probe_prompt_hash
from .real_pilot import PRICES, run_pilot
from .real_tasks import GENERATOR_VERSION as REPAIR_GENERATOR_VERSION
from .replay import REPLAY_VERSION
from .sealing import seal_run_dir
from .yaml_subset import parse_yaml

METRICS_FIELDS = ["strategy", "team_size", "seed", "fraction_correct",
                  "success", "tokens_in", "tokens_out", "cost_usd",
                  "retries", "wall_seconds"]

CHAIN_METRICS_FIELDS = ["strategy", "team_size", "seed", "horizon",
                        "fraction_correct", "final_state_correct",
                        "success", "tokens_in", "tokens_out", "cost_usd",
                        "retries", "wall_seconds"]

STEPS_FIELDS = ["run_id", "strategy", "horizon", "seed", "step_index",
                "agent", "state_correct", "local_correct",
                "first_state_correct", "tokens_in", "tokens_out",
                "recovery_tokens", "retries"]

FAMILIES = ("codebase_repair", "error_accumulation",
            "model_calibrated_generation")
CHAIN_STRATEGIES = ("orchestrated", "verified")

PROVIDERS = ("anthropic", "openai", "fake")
_DEFAULT_MODELS = {"anthropic": "claude-haiku-4-5", "openai": "gpt-5-mini",
                   "fake": "fake"}


@dataclass
class ExperimentSpec:
    """A validated experiment config (the parsed YAML, normalized)."""
    name: str
    question: str = ""
    provider: str = "anthropic"
    model: str | None = None
    difficulty: str = "standard"
    blind: bool = False
    sizes: tuple = (1, 2, 4, 8)
    strategies: tuple = ("independent", "orchestrated", "verified")
    seeds: tuple = (0, 1, 2)
    per_run_token_cap: int = 400_000
    spend_cap_usd: float = 15.0
    workers: int = 4
    output_root: str = "results"
    fake_competence: float = 0.85
    family: str = "codebase_repair"
    horizons: tuple = ()
    team_size: int = 4
    analysis: dict | None = None
    caveats: tuple = ()
    raw: dict = field(default_factory=dict)
    # Phase A-3 (model_calibrated_generation) fields
    band_targets: tuple = ()
    band_tolerances: tuple = ()
    batch_size: int = 200
    max_settings: int = 6
    validation_n: int = 500
    per_probe_token_cap: int = 12_000

    @property
    def n_runs(self) -> int:
        if self.family == "error_accumulation":
            return (len(self.horizons) * len(self.strategies)
                    * len(self.seeds))
        if self.family == "model_calibrated_generation":
            # an upper bound: every band probes max_settings batches
            # and validates once
            return len(self.band_targets) * (
                self.max_settings * self.batch_size + self.validation_n)
        return len(self.sizes) * len(self.strategies) * len(self.seeds)

    def grid(self) -> dict:
        if self.family == "error_accumulation":
            return {"family": self.family,
                    "horizons": list(self.horizons),
                    "strategies": list(self.strategies),
                    "seeds": list(self.seeds),
                    "team_size": self.team_size}
        if self.family == "model_calibrated_generation":
            return {"family": self.family,
                    "bands": [{"target": t, "tolerance": tol}
                              for t, tol in zip(self.band_targets,
                                                self.band_tolerances)],
                    "batch_size": self.batch_size,
                    "max_settings": self.max_settings,
                    "validation_n": self.validation_n,
                    "per_probe_token_cap": self.per_probe_token_cap}
        return {"sizes": list(self.sizes), "strategies": list(self.strategies),
                "seeds": list(self.seeds)}

    @property
    def analysis_declaration(self) -> dict:
        """The resolved analysis declaration (Invariant 4): what the
        config declares, or the family's documented default. Stored in
        the manifest; the analysis itself is generic machinery."""
        if self.analysis is not None:
            return self.analysis
        if self.family == "error_accumulation":
            return dict(CHAIN_ANALYSIS_DEFAULT)
        if self.family == "model_calibrated_generation":
            # no generic-analysis declaration: the phase's analysis
            # is the calibration itself, computed from probes.csv by
            # teamscale.model_calibrate (versioned there)
            return {"kind": "model_calibration",
                    "calibration_version": MODEL_CALIBRATION_VERSION,
                    "input": "probes.csv"}
        return dict(DEFAULT_ANALYSIS)


def spec_from_dict(data: dict) -> ExperimentSpec:
    if not isinstance(data, dict):
        raise ValueError("experiment config must be a YAML mapping")
    name = data.get("name")
    if not name or not isinstance(name, str):
        raise ValueError("experiment config requires a string 'name'")
    provider = str(data.get("provider", "anthropic")).strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}: "
                         f"use one of {PROVIDERS}")
    difficulty = str(data.get("difficulty", "standard")).strip().lower()
    if difficulty not in ("standard", "hard"):
        raise ValueError(f"unknown difficulty {difficulty!r}: "
                         "use 'standard' or 'hard'")

    def int_list(key, default):
        vals = data.get(key, default)
        if not isinstance(vals, list) or not vals:
            raise ValueError(f"'{key}' must be a non-empty list")
        return tuple(int(v) for v in vals)

    def str_list(key, default):
        vals = data.get(key, default)
        if not isinstance(vals, list) or not vals:
            raise ValueError(f"'{key}' must be a non-empty list")
        return tuple(str(v) for v in vals)

    def float_list(key, default):
        vals = data.get(key, default)
        if not isinstance(vals, list) or not vals:
            raise ValueError(f"'{key}' must be a non-empty list")
        return tuple(float(v) for v in vals)

    family = str(data.get("family", "codebase_repair")).strip().lower()
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}: use one of {FAMILIES}")
    strategies = str_list("strategies",
                          ["independent", "orchestrated", "verified"])
    from .real_pilot import STRATEGIES as _KNOWN
    known = CHAIN_STRATEGIES if family == "error_accumulation" else _KNOWN
    unknown = [s for s in strategies if s not in known]
    if unknown:
        raise ValueError(f"unknown strategies {unknown}: use {known}")
    analysis = data.get("analysis")
    if analysis is not None and not isinstance(analysis, dict):
        raise ValueError("'analysis' must be a mapping when present")
    caveats = data.get("caveats", [])
    if not isinstance(caveats, list):
        raise ValueError("'caveats' must be a list of strings")
    band_targets: tuple = ()
    band_tolerances: tuple = ()
    if family == "model_calibrated_generation":
        band_targets = float_list("band_targets", [])
        band_tolerances = float_list("band_tolerances", [])
        if len(band_targets) != len(band_tolerances):
            raise ValueError("'band_targets' and 'band_tolerances' "
                             "must have the same length")
        for t, tol in zip(band_targets, band_tolerances):
            if not 0.0 < t < 1.0 or tol <= 0.0 or t - tol <= 0.0 \
                    or t + tol >= 1.0:
                raise ValueError(
                    f"invalid band target {t} ± {tol}: bands must lie "
                    "strictly inside (0, 1)")
    return ExperimentSpec(
        name=name,
        question=str(data.get("question", "")),
        provider=provider,
        model=(str(data["model"]) if data.get("model") else None),
        difficulty=difficulty,
        blind=bool(data.get("blind", False)),
        sizes=int_list("sizes", [1, 2, 4, 8]),
        strategies=strategies,
        seeds=int_list("seeds", [0, 1, 2]),
        per_run_token_cap=int(data.get("per_run_token_cap", 400_000)),
        spend_cap_usd=float(data.get("spend_cap_usd", 15.0)),
        workers=int(data.get("workers", 4)),
        output_root=str(data.get("output_root", "results")),
        fake_competence=float(data.get("fake_competence", 0.85)),
        family=family,
        horizons=(int_list("horizons", [2, 5, 10, 20, 50])
                  if family == "error_accumulation" else ()),
        team_size=int(data.get("team_size", 4)),
        analysis=analysis,
        caveats=tuple(str(c) for c in caveats),
        raw=dict(data),
        band_targets=band_targets,
        band_tolerances=band_tolerances,
        batch_size=int(data.get("batch_size", 200)),
        max_settings=int(data.get("max_settings", 6)),
        validation_n=int(data.get("validation_n", 500)),
        per_probe_token_cap=int(data.get("per_probe_token_cap",
                                         12_000)),
    )


def load_config(path):
    """Return (spec, config_text, config_sha256) for a YAML config file."""
    text = Path(path).read_text()
    data = parse_yaml(text)
    spec = spec_from_dict(data)
    return spec, text, hashlib.sha256(text.encode()).hexdigest()


# ------------------------------------------------------------------ helpers
def _git_info() -> dict:
    repo = Path(__file__).resolve().parent.parent
    info = {"commit": "unknown", "dirty": None}
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            info["commit"] = out.stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                                capture_output=True, text=True, timeout=10)
        if status.returncode == 0:
            info["dirty"] = bool(status.stdout.strip())
    except Exception:
        pass
    return info


def _backend_factory(provider: str, model: str, spec: ExperimentSpec):
    if provider == "fake":
        def make_fake(cfg):
            if "horizon" in cfg:
                seed = (cfg["seed"] * 10000 + cfg["horizon"] * 10
                        + cfg["team_size"])
            else:
                seed = cfg["seed"] * 100 + cfg["team_size"]
            return FakeLLMBackend(competence=spec.fake_competence,
                                  seed=seed, blind=cfg.get("blind", False))
        return make_fake
    if provider == "anthropic":
        AnthropicLLMBackend(model=model)  # fail fast if no credential
        return lambda cfg: AnthropicLLMBackend(
            model=model, blind=cfg.get("blind", False))
    OpenAILLMBackend(model=model)  # fail fast if no credential
    return lambda cfg: OpenAILLMBackend(
        model=model, blind=cfg.get("blind", False))


def _read_run_results(run_dir: Path) -> list:
    results = []
    for path in sorted((run_dir / "results").glob("*.json")):
        results.append(json.loads(path.read_text()))
    order = {name: i for i, name in enumerate(
        ["independent", "orchestrated", "verified"])}
    results.sort(key=lambda r: (order.get(r["strategy"], 99),
                                r["team_size"], r["seed"]))
    return results


def _write_metrics_csv(run_dir: Path, results: list) -> None:
    with open(run_dir / "metrics.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRICS_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "strategy": r["strategy"], "team_size": r["team_size"],
                "seed": r["seed"], "fraction_correct": r["fraction_correct"],
                "success": 1 if r["success"] else 0,
                "tokens_in": r["tokens_in"], "tokens_out": r["tokens_out"],
                "cost_usd": r["cost_usd"], "retries": r["retries"],
                "wall_seconds": r["wall_time_s"]})


def _read_chain_results(run_dir: Path) -> list:
    results = []
    for path in sorted((run_dir / "results").glob("*.json")):
        results.append(json.loads(path.read_text()))
    order = {name: i for i, name in enumerate(CHAIN_STRATEGIES)}
    results.sort(key=lambda r: (order.get(r["strategy"], 99),
                                r["horizon"], r["seed"]))
    return results


def _write_chain_metrics_csv(run_dir: Path, results: list) -> None:
    with open(run_dir / "metrics.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CHAIN_METRICS_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "strategy": r["strategy"], "team_size": r["team_size"],
                "seed": r["seed"], "horizon": r["horizon"],
                "fraction_correct": r["fraction_correct"],
                "final_state_correct": r["final_state_correct"],
                "success": 1 if r["success"] else 0,
                "tokens_in": r["tokens_in"], "tokens_out": r["tokens_out"],
                "cost_usd": r["cost_usd"], "retries": r["retries"],
                "wall_seconds": r["wall_time_s"]})


def _write_steps_csv(run_dir: Path, results: list) -> None:
    """Per-step table: the analysis's declared steps input (survival,
    hazard, local-vs-state correctness) -- tabular data only, never
    traces or sandboxes."""
    with open(run_dir / "steps.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=STEPS_FIELDS)
        writer.writeheader()
        for r in results:
            for s in r["per_step"]:
                writer.writerow({
                    "run_id": r["run_id"], "strategy": r["strategy"],
                    "horizon": r["horizon"], "seed": r["seed"],
                    "step_index": s["step_index"], "agent": s["agent"],
                    "state_correct": 1 if s["state_correct"] else 0,
                    "local_correct": 1 if s["local_correct"] else 0,
                    "first_state_correct": (
                        1 if s["first_state_correct"] else 0),
                    "tokens_in": s["tokens_in"],
                    "tokens_out": s["tokens_out"],
                    "recovery_tokens": s["recovery_tokens"],
                    "retries": s["retries"]})


def _write_recovery_events(run_dir: Path, results: list) -> None:
    """The error-cascade dataset: every external check that fired,
    one JSON object per line, in run order."""
    with open(run_dir / "recovery_events.jsonl", "w") as fh:
        for r in results:
            for event in r["recovery_events"]:
                fh.write(json.dumps(event, sort_keys=True) + "\n")


def _assemble_traces(run_dir: Path, results: list) -> None:
    with open(run_dir / "traces.jsonl", "w") as out:
        for r in results:
            trace_file = run_dir / "traces" / f"{r['run_id']}.jsonl"
            if trace_file.exists():
                text = trace_file.read_text()
                out.write(text)
                if not text.endswith("\n"):
                    out.write("\n")


def cost_accounting(results: list) -> dict:
    """Totals from the run results; reconciled against the ledger by the
    caller (the ledger total is in summary.json's ledger_spent_usd)."""
    total = round(sum(r["cost_usd"] for r in results), 6)
    by_strategy = {}
    for strategy in sorted({r["strategy"] for r in results}):
        cell = [r for r in results if r["strategy"] == strategy]
        cell_cost = sum(r["cost_usd"] for r in cell)
        correct = sum(r["n_correct"] for r in cell)
        by_strategy[strategy] = {
            "runs": len(cell),
            "total_cost_usd": round(cell_cost, 6),
            "cost_per_run_usd": round(cell_cost / len(cell), 6),
            "subtasks_correct": correct,
            "cost_per_correct_subtask_usd": (
                round(cell_cost / correct, 6) if correct else None),
        }
    return {
        "total_cost_usd": total,
        "runs_completed": len(results),
        "cost_per_run_usd": (round(total / len(results), 6)
                             if results else 0.0),
        "by_strategy": by_strategy,
    }


def render_summary_md(spec: ExperimentSpec, manifest: dict,
                      accounting: dict, pilot_summary: dict) -> str:
    lines = [
        f"# {spec.name}",
        "",
        f"**Question:** {spec.question or '(not stated)'}",
        "",
        "## Design",
        "",
        f"- provider / model: {manifest['provider']} / {manifest['model']}",
        f"- difficulty: {spec.difficulty}"
        + ("; **blind mode** (agents cannot run checks; verification is "
           "external only)" if spec.blind else ""),
        (f"- grid: strategies {list(spec.strategies)} x horizons "
         f"{list(spec.horizons)} x seeds {list(spec.seeds)} at fixed "
         f"team size {spec.team_size} "
         if spec.family == "error_accumulation" else
         f"- grid: strategies {list(spec.strategies)} x team sizes "
         f"{list(spec.sizes)} x seeds {list(spec.seeds)} ")
        + f"= {pilot_summary['runs_completed']} runs completed, "
        f"{len(pilot_summary['runs_skipped_spend_cap'])} skipped by the "
        "spend-cap gate, "
        f"{len(pilot_summary['runs_failed'])} failed",
        f"- caps: {spec.per_run_token_cap} tokens per run; "
        f"${spec.spend_cap_usd:.2f} spend cap",
        "",
        "## Cost accounting",
        "",
        f"- total cost: ${accounting['total_cost_usd']:.4f} "
        f"(ledger: ${pilot_summary['ledger_spent_usd']:.4f} of "
        f"${pilot_summary['spend_cap_usd']:.2f} cap)",
        f"- cost per run: ${accounting['cost_per_run_usd']:.4f}",
        "",
        "| strategy | runs | total cost | cost/run | correct subtasks "
        "| cost per correct subtask |",
        "|---|---|---|---|---|---|",
    ]
    for strategy, cell in accounting["by_strategy"].items():
        cps = cell["cost_per_correct_subtask_usd"]
        lines.append(
            f"| {strategy} | {cell['runs']} | ${cell['total_cost_usd']:.4f} "
            f"| ${cell['cost_per_run_usd']:.4f} | {cell['subtasks_correct']} "
            f"| {'$' + format(cps, '.4f') if cps is not None else 'n/a'} |")
    lines += [
        "",
        "Mean fraction correct across completed runs: "
        f"{pilot_summary['mean_fraction_correct']}.",
        "",
        "For confidence intervals and figures, run "
        "`python -m teamscale.analyze <this directory>` -- it regenerates "
        "this file with the full analysis, from metrics.csv alone.",
        "",
    ]
    return "\n".join(lines)


# -------------------------------------------------------------------- runner
def run_config_experiment(spec: ExperimentSpec, *, config_text: str = "",
                          config_sha256: str = "", config_source: str = "",
                          fake: bool = False, run_id: str | None = None,
                          base_dir: Path | None = None,
                          calibration_path: str | None = None) -> Path:
    """Execute one config-driven experiment; return the run directory.

    Refuses (FileExistsError) if the target run directory already exists.
    For the error_accumulation family, a LIVE run additionally requires
    a passing calibration gate report (--calibration): the release gate
    in experiments/error-accumulation/DESIGN.md, enforced here so the
    experiment cannot silently run on an uncalibrated generator.
    """
    provider = "fake" if fake or spec.provider == "fake" else spec.provider
    model = "fake" if provider == "fake" else (
        spec.model or _DEFAULT_MODELS[provider])
    if spec.family == "error_accumulation" and not spec.blind:
        raise ValueError("the error_accumulation family is blind-only: "
                         "set blind: true (chains carry no check tool)")

    calibration_info = None
    calibration_bytes = None
    if calibration_path:
        calibration_bytes = Path(calibration_path).read_bytes()
        report = json.loads(calibration_bytes)
        calibration_info = {
            "version": report.get("version"),
            "gate": report.get("gate"),
            "generator_version": report.get("generator_version"),
            "sha256": hashlib.sha256(calibration_bytes).hexdigest(),
            "source": str(calibration_path)}
    if spec.family == "error_accumulation" and provider != "fake":
        if calibration_info is None:
            raise ValueError(
                "live error_accumulation runs require a calibration "
                "report: run teamscale.calibrate and pass --calibration")
        if calibration_info["gate"] != "pass":
            raise ValueError("calibration gate did not pass; refusing "
                             "to run (see DESIGN.md release gate)")
        if (calibration_info["generator_version"]
                != CHAIN_GENERATOR_VERSION):
            raise ValueError(
                "calibration was run against generator "
                f"{calibration_info['generator_version']}, this code "
                f"is {CHAIN_GENERATOR_VERSION}; recalibrate")

    if not config_sha256:
        canonical = json.dumps(spec.raw, sort_keys=True)
        config_sha256 = hashlib.sha256(canonical.encode()).hexdigest()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = run_id or f"{stamp}-{config_sha256[:8]}"
    root = Path(base_dir) if base_dir is not None else Path(".")
    run_dir = root / spec.output_root / spec.name / run_id
    if run_dir.exists():
        raise FileExistsError(
            f"refusing to write into existing run directory: {run_dir} "
            "(run directories are immutable; a re-run gets a new run-id)")
    run_dir.mkdir(parents=True)

    from .chain_experiment import prompt_hash as chain_prompt_hash
    is_chain = spec.family == "error_accumulation"
    is_mcg = spec.family == "model_calibrated_generation"
    pin, pout = PRICES.get(model, (0.0, 0.0))
    manifest = {
        "experiment": spec.name,
        "question": spec.question,
        "run_id": run_id,
        "config": spec.raw,
        "config_sha256": config_sha256,
        "config_source": config_source,
        "git": _git_info(),
        "code_version": __version__,
        "provider": provider,
        "requested_provider": spec.provider,
        "model": model,
        "pricing": {"model": model, "input_per_1m_usd": pin,
                    "output_per_1m_usd": pout},
        "difficulty": spec.difficulty,
        "blind": spec.blind,
        "family": spec.family,
        "generator_version": (
            CHAIN_GENERATOR_VERSION if is_chain
            else LADDER_GENERATOR_VERSION if is_mcg
            else REPAIR_GENERATOR_VERSION),
        "prompt_hash": (
            chain_prompt_hash() if is_chain
            else probe_prompt_hash() if is_mcg
            else repair_prompt_hash()),
        "calibration_version": (
            CALIBRATION_VERSION if is_chain
            else MODEL_CALIBRATION_VERSION if is_mcg
            else None),
        "calibration": calibration_info,
        "replay_version": REPLAY_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "analysis": spec.analysis_declaration,
        "caveats": list(spec.caveats),
        "grid": spec.grid(),
        "n_runs_planned": spec.n_runs,
        "per_run_token_cap": spec.per_run_token_cap,
        "spend_cap_usd": spec.spend_cap_usd,
        "workers": spec.workers,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "host": {"system": platform.system(), "machine": platform.machine(),
                 "python": platform.python_version()},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if calibration_bytes is not None:
        (run_dir / "calibration.json").write_bytes(calibration_bytes)

    factory = _backend_factory(provider, model, spec)
    ledger_name = ("ledger_fake.json" if provider == "fake"
                   else "ledger.json")
    if is_mcg:
        # Phase A-3: the driver runs the calibration controller and
        # writes the whole bundle body (probes.csv, metrics.csv,
        # traces.jsonl, calibration.json, summary.json, SUMMARY.md);
        # this shell provides the manifest, the immutability
        # discipline, and the seal.
        from .model_calibrate import run_model_calibration_bundle
        run_model_calibration_bundle(
            spec=spec, run_dir=run_dir, run_id=run_id, model=model,
            provider=provider, factory=factory)
        seal_run_dir(run_dir)
        return run_dir
    if is_chain:
        from .chain_experiment import run_chain_grid
        pilot_summary = run_chain_grid(
            out_dir=run_dir, backend_factory=factory, model=model,
            horizons=spec.horizons, strategies=spec.strategies,
            seeds=spec.seeds, team_size=spec.team_size,
            spend_cap_usd=spec.spend_cap_usd, workers=spec.workers,
            per_run_token_cap=spec.per_run_token_cap,
            ledger_name=ledger_name)
        results = _read_chain_results(run_dir)
        _write_chain_metrics_csv(run_dir, results)
        _write_steps_csv(run_dir, results)
        _write_recovery_events(run_dir, results)
    else:
        pilot_summary = run_pilot(
            out_dir=run_dir, backend_factory=factory, model=model,
            sizes=spec.sizes, strategies=spec.strategies, seeds=spec.seeds,
            spend_cap_usd=spec.spend_cap_usd, workers=spec.workers,
            per_run_token_cap=spec.per_run_token_cap,
            ledger_name=ledger_name,
            difficulty=spec.difficulty, blind=spec.blind)
        results = _read_run_results(run_dir)
        _write_metrics_csv(run_dir, results)
    _assemble_traces(run_dir, results)
    (run_dir / "figures").mkdir(exist_ok=True)

    accounting = cost_accounting(results)
    summary = dict(pilot_summary)
    summary["cost_accounting"] = accounting
    summary["run_id"] = run_id
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (run_dir / "SUMMARY.md").write_text(
        render_summary_md(spec, manifest, accounting, pilot_summary))
    seal_run_dir(run_dir)
    return run_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="teamscale.config_experiment",
        description="run one config-driven experiment from a YAML file")
    ap.add_argument("config", help="path to the experiment YAML config")
    ap.add_argument("--fake", action="store_true",
                    help="offline dry run on the fake backend ($0, no "
                         "network), whatever provider the config names")
    ap.add_argument("--fake-competence", type=float, default=None,
                    help="fake-backend competence (default: config value "
                         "or 0.85); only meaningful with --fake/provider "
                         "fake")
    ap.add_argument("--run-id", default=None,
                    help="override the generated run-id (the run directory "
                         "must not already exist)")
    ap.add_argument("--calibration", default=None,
                    help="path to a calibration gate report JSON "
                         "(teamscale.calibrate --out); copied into the "
                         "run bundle as calibration.json. Required for "
                         "live error_accumulation runs.")
    args = ap.parse_args(argv)

    spec, text, sha = load_config(args.config)
    if args.fake_competence is not None:
        spec.fake_competence = args.fake_competence
    run_dir = run_config_experiment(
        spec, config_text=text, config_sha256=sha,
        config_source=str(args.config), fake=args.fake, run_id=args.run_id,
        calibration_path=args.calibration)
    summary = json.loads((run_dir / "summary.json").read_text())
    if "runs_completed" in summary:
        print(f"experiment: {spec.name}  runs completed: "
              f"{summary['runs_completed']}  skipped (spend cap): "
              f"{len(summary['runs_skipped_spend_cap'])}  failed: "
              f"{len(summary['runs_failed'])}")
        print(f"cost: ${summary['total_cost_usd']}  "
              f"mean fraction correct: {summary['mean_fraction_correct']}")
    else:
        bands = "; ".join(
            f"{b['label']}: {b['status']}"
            + (f" @ {b['setting']}" if b.get("setting") else "")
            + (f", validation pass={b['validation_pass']}"
               if b.get("validation_pass") is not None else "")
            for b in summary.get("bands", []))
        print(f"experiment: {spec.name}  bands: {bands}")
        print(f"probes: {summary['total_probes']}  "
              f"cost: ${summary['total_cost_usd']} of "
              f"${summary['spend_cap_usd']:.2f} cap")
    print(f"run directory: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
