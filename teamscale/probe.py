"""Probe harness: blind, single-attempt measurements of one
standalone ladder step (Phase A-3).

A probe is the calibration instrument of
experiments/model-calibrated-generation/DESIGN.md: one agent, one
step, one attempt, no check feedback of any kind. The agent reads
state.py in its sandbox, applies the step's transformation(s), and
writes the full new state back. Failure = the first-submission state
is not the generator's ground-truth state (teamscale.ladder
grading). Every probe is recorded with its seed, setting, outcome,
tokens, model calls, latency, and cost, and its sandbox and trace are
kept in the run bundle so replay can re-derive every number.

Outcome taxonomy: a probe is "correct", "incorrect", or "error".
"error" is reserved for infrastructure failures (transport/API
errors, missing credentials) where no model behavior was measured;
error probes are excluded from failure-rate statistics and counted
separately, so an API outage cannot masquerade as model failures.
Model-behavior outcomes -- including a blown token budget, a refusal,
or a max-tokens stop -- are graded on whatever the agent left in
state.py, exactly as in the chained family.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import chains as chains_mod
from . import ladder as ladder_mod
from .chains import parse_state_file, render_state_file
from .ladder import grade_probe_step, make_probe_step
from .real_pilot import cost_usd

FAMILY = "model_calibrated_generation"
PER_PROBE_TOKEN_BUDGET = 12_000

PROBE_SYSTEM_PROMPT = (
    "You are a software engineer carrying out one transformation "
    "task over a state file. You work only inside a sandbox "
    "directory, using the provided tools. Read the state file, apply "
    "exactly the transformation specified, and write the full new "
    "state back. You cannot run any check yourself: your submission "
    "is final and will be graded externally. Be economical: few, "
    "decisive tool calls. When you believe the new state is correct, "
    "stop and reply with a one-line summary."
)

PROBE_USER_TEMPLATE = (
    "The chain state is a list of {L} integers stored in the file "
    "state.py in your sandbox, in the form STATE = [...].\n\n"
    "Current state (the content of state.py):\n{state}\n\n"
    "Your transformation:\n{spec}\n\n"
    "Write the FULL new state to state.py in exactly the same form "
    "(STATE = [ ... ]), using the write_file tool, then stop with a "
    "one-line summary. The list must keep exactly {L} integers, each "
    "in 0..999. Do not edit any other file."
)

PROMPT_TEMPLATES = [PROBE_SYSTEM_PROMPT, PROBE_USER_TEMPLATE]


def prompt_hash() -> str:
    """SHA-256 of the probe family's prompt templates, for manifest
    provenance."""
    return hashlib.sha256(
        "\n".join(PROMPT_TEMPLATES).encode()).hexdigest()


def band_label(target: float) -> str:
    return f"t{int(round(target * 100))}"


def probe_id_for(backend_name: str, label: str, phase: str,
                 setting_name: str, seed: int) -> str:
    return f"{backend_name}-probe-{label}-{phase}-{setting_name}-s{seed}"


def _is_transport_error(error: str | None) -> bool:
    if not error:
        return False
    return ("API returned HTTP" in error
            or "API request failed" in error
            or "no API key configured" in error
            or "produced no key" in error)


def _call_backend(backend, *, step, prompt, sandbox, token_budget):
    """Dispatch one probe attempt: the fake backend takes scripted
    apply/corrupt callables; real backends take the prompt (the same
    split as teamscale.chain_experiment)."""
    if hasattr(backend, "scripted_steps") and backend.scripted_steps:
        return backend.fix_step(
            sandbox=sandbox, state_file=chains_mod.STATE_FILE,
            apply_fn=lambda st: step.apply_to(st),
            corrupt_fn=ladder_mod.corrupt_state,
            subtask=step.subtask_id, token_budget=token_budget,
            agent_id=0, feedback=None)
    return backend.fix_step(
        prompt=prompt, system_prompt=PROBE_SYSTEM_PROMPT,
        sandbox=sandbox, token_budget=token_budget, agent_id=0,
        subtask=step.subtask_id)


def run_single_probe(*, backend, step, label: str, target: float,
                     phase: str, out_dir: Path,
                     token_budget: int = PER_PROBE_TOKEN_BUDGET,
                     model: str = "claude-haiku-5-5") -> dict:
    """Execute one probe; write its sandbox + trace; return the
    probe record."""
    out_dir = Path(out_dir)
    probe_id = probe_id_for(backend.name, label, phase, step.setting,
                            step.seed)
    sandbox = out_dir / "sandboxes" / probe_id
    if sandbox.exists():
        shutil.rmtree(sandbox)
    sandbox.mkdir(parents=True)
    state_path = sandbox / chains_mod.STATE_FILE
    state_path.write_text(render_state_file(step.input_state))
    specs_dir = sandbox / "specs"
    specs_dir.mkdir()
    (specs_dir / "step.txt").write_text(
        f"Setting {step.setting}, seed {step.seed}\n{step.spec_text}\n")

    trace_path = out_dir / "traces" / f"{probe_id}.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    events = [{"event": "run_start", "run_id": probe_id,
               "family": FAMILY, "phase": phase, "band": label,
               "band_target": target, "setting": step.setting,
               "seed": step.seed, "backend": backend.name,
               "blind": True,
               "generator_version": step.generator_version}]

    prompt = PROBE_USER_TEMPLATE.format(
        L=step.state_len, state=repr(step.input_state),
        spec=step.spec_text)
    started = time.perf_counter()
    fix = _call_backend(backend, step=step, prompt=prompt,
                        sandbox=sandbox, token_budget=token_budget)
    latency = round(time.perf_counter() - started, 3)
    events.extend(fix.events)

    submitted = parse_state_file(state_path.read_text())
    correct, error_class, _msg = grade_probe_step(step, submitted)
    if _is_transport_error(fix.error):
        outcome, correct = "error", None
    else:
        outcome = "correct" if correct else "incorrect"
    record = {
        "probe_id": probe_id, "phase": phase, "band": label,
        "band_target": target, "setting": step.setting,
        "seed": step.seed, "outcome": outcome, "correct": correct,
        "error_class": error_class if outcome == "incorrect" else None,
        "tokens_in": fix.tokens_in, "tokens_out": fix.tokens_out,
        "model_calls": fix.model_calls,
        "cost_usd": round(cost_usd(model, fix.tokens_in,
                                   fix.tokens_out), 6),
        "latency_s": latency,
        "error": fix.error,
    }
    events.append({"event": "probe_end", "run_id": probe_id,
                   "outcome": outcome,
                   "error_class": record["error_class"],
                   "tokens_in": fix.tokens_in,
                   "tokens_out": fix.tokens_out,
                   "model_calls": fix.model_calls,
                   "cost_usd": record["cost_usd"],
                   "latency_s": latency})
    with open(trace_path, "w") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")
    return record


def run_probe_batch(*, backend, setting_name: str, seeds, label: str,
                    target: float, phase: str, out_dir: Path,
                    token_budget: int = PER_PROBE_TOKEN_BUDGET,
                    model: str = "claude-haiku-5-5",
                    workers: int = 4) -> list:
    """Probe one batch: len(seeds) standalone steps at one ladder
    setting, concurrently (each probe has its own sandbox; steps are
    independent by construction). Records are returned sorted by
    seed, so batch statistics never depend on completion order."""
    steps = [make_probe_step(seed, setting_name) for seed in seeds]

    def one(step):
        return run_single_probe(
            backend=backend, step=step, label=label, target=target,
            phase=phase, out_dir=out_dir, token_budget=token_budget,
            model=model)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(one, steps))
    records.sort(key=lambda r: r["seed"])
    return records
