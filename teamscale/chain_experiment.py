"""Chain runner: real agent teams on the chained-transformation family.

    (driven through teamscale.config_experiment with family:
    error_accumulation -- see experiments/error-accumulation/)

Semantics, mirroring teamscale.real_pilot's strategy definitions on a
sequential task family:

- A run is one chain of N steps over a shared state file (state.py in
  the run's sandbox). Steps execute strictly in order; team size only
  rotates which agent takes each step ((k-1) % team_size). This
  measures dependency and verification, not parallel speedup.
- Chains are BLIND by construction: agents never get a check tool.
  Grading is the runner re-deriving correctness from the generator's
  ground truth after each step -- measurement only; for the
  orchestrated strategy it is never fed back.
- verified = orchestrated + after each step submission the runner
  runs the external check (grade against ground-truth state); on
  failure the check output goes back to the same agent for exactly
  ONE retry, with the state file first reset to the step's input.
  Every check -- pass or fail -- is logged as a recovery event.
- Ground truth is strict and path-independent: step k is correct iff
  the state after it equals the generator's ground-truth state,
  however the agent got there. Local correctness (the transformation
  applied correctly to the input actually received) is recorded
  alongside; see teamscale.chains.grade_step.

Caps and accounting are the pilot's: a per-run token cap and the
shared SpendLedger gate (a run starts only if ledger + worst case
fits the cap).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import chains as chains_mod
from .chains import grade_step, make_chain, parse_state_file, \
    render_state_file
from .real_pilot import Board, SpendLedger, cost_usd, worst_case_run_usd

FAMILY = "error_accumulation"
PER_STEP_TOKEN_BUDGET = 25_000
STRATEGIES = ("orchestrated", "verified")

CHAIN_SYSTEM_PROMPT = (
    "You are a software engineer carrying out one step of a "
    "transformation chain over a shared state file. You work only "
    "inside a sandbox directory, using the provided tools. Read the "
    "state file, apply exactly the transformation your step specifies, "
    "and write the full new state back. You cannot run any check "
    "yourself: your work will be checked externally after you submit "
    "it. Be economical: few, decisive tool calls. When you believe "
    "the new state is correct, stop and reply with a one-line summary."
)

CHAIN_USER_TEMPLATE = (
    "You are step {k} of {n} in a transformation chain. The chain "
    "state is a list of {L} integers stored in the file state.py in "
    "your sandbox, in the form STATE = [...]. Each step transforms "
    "the state, and your output becomes the next step's input, so an "
    "error here propagates to every later step.\n\n"
    "Current state (the content of state.py):\n{state}\n\n"
    "Your transformation for this step:\n{spec}\n\n"
    "Write the FULL new state to state.py in exactly the same form "
    "(STATE = [ ... ]), using the write_file tool, then stop with a "
    "one-line summary. The list must keep exactly {L} integers, each "
    "in 0..999. Do not edit any other file."
)

CHAIN_RETRY_SUFFIX = (
    "\n\nYour previous submission for this step was checked "
    "externally and FAILED. Check output:\n{feedback}\nThe state.py "
    "file has been reset to the current state above. Recompute the "
    "transformation carefully from that state; do not repeat the "
    "same result."
)

PROMPT_TEMPLATES = [CHAIN_SYSTEM_PROMPT, CHAIN_USER_TEMPLATE,
                    CHAIN_RETRY_SUFFIX]


def prompt_hash() -> str:
    """SHA-256 of the chain family's prompt templates (the shared
    board suffix from teamscale.llm_backend included), for manifest
    provenance."""
    from .llm_backend import BOARD_SUFFIX
    return hashlib.sha256(
        "\n".join(PROMPT_TEMPLATES + [BOARD_SUFFIX]).encode()).hexdigest()


def run_id_for_chain(backend_name: str, strategy: str, horizon: int,
                     seed: int) -> str:
    return f"{backend_name}-{FAMILY}-{strategy}-h{horizon}-s{seed}"


def _jsonable_state(value):
    """Store a parsed state for the run record: lists stay lists,
    anything else becomes None (grading only ever needs the list, or
    the fact that it was not one)."""
    return list(value) if isinstance(value, list) else None


def _build_prompt(step, n_steps: int, input_raw_state, board_text: str,
                  feedback: str | None) -> str:
    from .llm_backend import BOARD_SUFFIX
    prompt = CHAIN_USER_TEMPLATE.format(
        k=step.index, n=n_steps, L=chains_mod.STATE_LEN,
        state=repr(input_raw_state), spec=step.spec_text)
    if board_text:
        prompt += BOARD_SUFFIX.format(board_text=board_text)
    if feedback:
        prompt += CHAIN_RETRY_SUFFIX.format(feedback=feedback)
    return prompt


def _call_backend(backend, *, step, prompt, sandbox, token_budget,
                  agent_id, feedback):
    """Dispatch one step attempt: the fake backend takes scripted
    apply/corrupt callables; real backends take the prompt."""
    if hasattr(backend, "scripted_steps") and backend.scripted_steps:
        return backend.fix_step(
            sandbox=sandbox, state_file=chains_mod.STATE_FILE,
            apply_fn=lambda st: step.apply(st),
            corrupt_fn=chains_mod.corrupt_state,
            subtask=step.subtask_id, token_budget=token_budget,
            agent_id=agent_id, feedback=feedback)
    return backend.fix_step(
        prompt=prompt, system_prompt=CHAIN_SYSTEM_PROMPT,
        sandbox=sandbox, token_budget=token_budget, agent_id=agent_id,
        subtask=step.subtask_id)


def run_single_chain(*, strategy: str, horizon: int, seed: int,
                     team_size: int, backend_factory, out_dir: Path,
                     per_run_token_cap: int = 400_000,
                     per_step_budget: int = PER_STEP_TOKEN_BUDGET,
                     model: str = "claude-haiku-5-5") -> dict:
    """Execute one chain run end to end; write result + trace files."""
    out_dir = Path(out_dir)
    backend = backend_factory({"strategy": strategy,
                               "team_size": team_size, "seed": seed,
                               "blind": True, "horizon": horizon})
    run_id = run_id_for_chain(backend.name, strategy, horizon, seed)
    task = make_chain(seed=seed, n_steps=horizon)
    sandbox = out_dir / "sandboxes" / run_id
    if sandbox.exists():
        shutil.rmtree(sandbox)
    sandbox.mkdir(parents=True)
    state_path = sandbox / chains_mod.STATE_FILE
    state_path.write_text(render_state_file(task.initial_state))
    specs_dir = sandbox / "specs"
    specs_dir.mkdir()
    for step in task.steps:
        (specs_dir / f"step_{step.index:02d}.txt").write_text(
            f"Step {step.index} of {horizon}\n{step.spec_text}\n")

    trace_path = out_dir / "traces" / f"{run_id}.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_fh = open(trace_path, "w")

    def trace(event: dict):
        trace_fh.write(json.dumps(event) + "\n")

    started = time.perf_counter()
    trace({"event": "run_start", "run_id": run_id, "family": FAMILY,
           "strategy": strategy, "team_size": team_size, "seed": seed,
           "horizon": horizon, "backend": backend.name, "blind": True,
           "generator_version": task.generator_version,
           "note": "steps execute sequentially; team size rotates the "
                   "agent taking each step, not parallelism"})

    board = Board()
    tokens_in = tokens_out = retries = 0
    step_records = []
    recovery_events = []

    def budget_left() -> int:
        return per_run_token_cap - (tokens_in + tokens_out)

    for step in task.steps:
        agent = (step.index - 1) % team_size
        trace({"event": "assign", "subtask": step.subtask_id,
               "agent": agent, "step_index": step.index})
        board.set(step.subtask_id, f"assigned to agent {agent}")
        input_raw = state_path.read_text()
        input_actual = parse_state_file(input_raw)

        error = None
        step_tin = step_tout = recovery_tokens = 0
        step_retried = 0
        if budget_left() <= 0:
            error = "run token cap reached"
            submitted = parse_state_file(state_path.read_text())
            state_ok, local_ok, first_eclass, msg = grade_step(
                step, input_actual, submitted)
            first_ok = state_ok
        else:
            prompt = _build_prompt(step, horizon, input_actual,
                                   board.render(), None)
            fix = _call_backend(
                backend, step=step, prompt=prompt, sandbox=sandbox,
                token_budget=min(per_step_budget, budget_left()),
                agent_id=agent, feedback=None)
            tokens_in += fix.tokens_in
            tokens_out += fix.tokens_out
            step_tin, step_tout = fix.tokens_in, fix.tokens_out
            for ev in fix.events:
                trace(ev)
            if fix.error:
                error = fix.error
            submitted = parse_state_file(state_path.read_text())
            state_ok, local_ok, first_eclass, msg = grade_step(
                step, input_actual, submitted)
            first_ok = state_ok
            trace({"event": "step_check", "subtask": step.subtask_id,
                   "step_index": step.index, "state_correct": state_ok,
                   "local_correct": local_ok, "strategy": strategy})
            if strategy == "verified":
                trace({"event": "verify", "subtask": step.subtask_id,
                       "step_index": step.index, "passed": state_ok})
                retry_outcome = "not_needed" if state_ok else "no_retry"
                if not state_ok and not fix.error and budget_left() > 0:
                    retries += 1
                    step_retried = 1
                    trace({"event": "retry",
                           "subtask": step.subtask_id,
                           "step_index": step.index})
                    state_path.write_text(input_raw)  # reset, then retry
                    prompt2 = _build_prompt(step, horizon, input_actual,
                                            board.render(), msg)
                    fix2 = _call_backend(
                        backend, step=step, prompt=prompt2,
                        sandbox=sandbox,
                        token_budget=min(per_step_budget, budget_left()),
                        agent_id=agent, feedback=msg)
                    tokens_in += fix2.tokens_in
                    tokens_out += fix2.tokens_out
                    step_tin += fix2.tokens_in
                    step_tout += fix2.tokens_out
                    recovery_tokens = fix2.tokens_in + fix2.tokens_out
                    for ev in fix2.events:
                        trace(ev)
                    if fix2.error:
                        error = fix2.error
                    submitted = parse_state_file(state_path.read_text())
                    state_ok, local_ok, _e2, msg = grade_step(
                        step, input_actual, submitted)
                    retry_outcome = ("recovered" if state_ok
                                     else "still_failing")
                    trace({"event": "step_check",
                           "subtask": step.subtask_id,
                           "step_index": step.index,
                           "state_correct": state_ok,
                           "local_correct": local_ok,
                           "strategy": strategy, "after_retry": True})
                recovery_events.append({
                    "run_id": run_id, "seed": seed, "horizon": horizon,
                    "strategy": strategy, "step_index": step.index,
                    "check_outcome": "pass" if first_ok else "fail",
                    "error_class": None if first_ok else first_eclass,
                    "would_cascade": step.index < horizon,
                    "retry_outcome": retry_outcome,
                    "chain_restored": state_ok,
                    "recovery_tokens": recovery_tokens})
        board.set(step.subtask_id,
                  f"agent {agent} submitted step {step.index}")
        step_records.append({
            "step_index": step.index, "agent": agent,
            "template": step.template,
            "state_correct": state_ok, "local_correct": local_ok,
            "first_state_correct": first_ok,
            "tokens_in": step_tin, "tokens_out": step_tout,
            "recovery_tokens": recovery_tokens,
            "retries": step_retried,
            "input_state": _jsonable_state(input_actual),
            "submitted_state": _jsonable_state(
                parse_state_file(state_path.read_text())),
            "error": error})

    steps_correct = sum(1 for r in step_records if r["state_correct"])
    final_ok = step_records[-1]["state_correct"] if step_records else False
    wall = round(time.perf_counter() - started, 3)
    result = {
        "run_id": run_id, "backend": backend.name, "model": model,
        "family": FAMILY, "difficulty": "hard", "blind": True,
        "generator_version": task.generator_version,
        "strategy": strategy, "team_size": team_size, "seed": seed,
        "horizon": horizon,
        "success": bool(final_ok),
        "final_state_correct": 1 if final_ok else 0,
        "fraction_correct": round(steps_correct / horizon, 4),
        "n_subtasks": horizon, "n_correct": steps_correct,
        "steps_correct": steps_correct,
        "tokens_in": tokens_in, "tokens_out": tokens_out,
        "total_tokens": tokens_in + tokens_out,
        "cost_usd": round(cost_usd(model, tokens_in, tokens_out), 6),
        "wall_time_s": wall,
        "retries": retries,
        "per_step": step_records,
        "recovery_events": recovery_events,
    }
    trace({"event": "run_end", "run_id": run_id,
           "success": result["success"],
           "fraction_correct": result["fraction_correct"],
           "final_state_correct": result["final_state_correct"],
           "total_tokens": result["total_tokens"],
           "cost_usd": result["cost_usd"], "wall_time_s": wall})
    trace_fh.close()
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{run_id}.json").write_text(json.dumps(result, indent=1))
    result["trace_path"] = str(trace_path)
    return result


def run_chain_grid(*, out_dir: Path, backend_factory, model: str,
                   horizons, strategies, seeds, team_size: int = 4,
                   spend_cap_usd: float = 15.0, workers: int = 4,
                   per_run_token_cap: int = 400_000,
                   ledger_name: str = "ledger.json") -> dict:
    """The A-2 grid: strategies x horizons x seeds, spend-cap gated,
    runs concurrent (steps within a run are sequential)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = SpendLedger(out_dir / ledger_name, spend_cap_usd, model)
    configs = [(s, h, seed) for s in strategies for h in horizons
               for seed in seeds]
    results, skipped, failed = [], [], []
    worst = worst_case_run_usd(model, per_run_token_cap)

    def one(cfg):
        strategy, horizon, seed = cfg
        return run_single_chain(
            strategy=strategy, horizon=horizon, seed=seed,
            team_size=team_size, backend_factory=backend_factory,
            out_dir=out_dir, per_run_token_cap=per_run_token_cap,
            model=model)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        reserved = 0.0
        for cfg in configs:
            if ledger.spent_usd + reserved + worst > ledger.cap_usd:
                skipped.append({"strategy": cfg[0], "horizon": cfg[1],
                                "seed": cfg[2],
                                "reason": "spend cap gate"})
                continue
            reserved += worst
            futures[pool.submit(one, cfg)] = cfg
        for fut, cfg in futures.items():
            try:
                result = fut.result()
            except Exception as e:  # one bad run must not kill the grid
                failed.append({"strategy": cfg[0], "horizon": cfg[1],
                               "seed": cfg[2],
                               "error": f"{type(e).__name__}: {e}"})
                continue
            ledger.record(result)
            results.append(result)

    summary = {
        "backend": results[0]["backend"] if results else "none",
        "model": model, "family": FAMILY, "difficulty": "hard",
        "blind": True,
        "grid": {"horizons": list(horizons),
                 "strategies": list(strategies), "seeds": list(seeds),
                 "team_size": team_size},
        "runs_completed": len(results),
        "runs_skipped_spend_cap": skipped, "runs_failed": failed,
        "total_tokens_in": sum(r["tokens_in"] for r in results),
        "total_tokens_out": sum(r["tokens_out"] for r in results),
        "total_cost_usd": round(sum(r["cost_usd"] for r in results), 4),
        "ledger_spent_usd": ledger.spent_usd,
        "spend_cap_usd": ledger.cap_usd,
        "mean_fraction_correct": (
            round(sum(r["fraction_correct"] for r in results)
                  / len(results), 4) if results else 0.0),
        "total_steps": sum(r["horizon"] for r in results),
        "recovery_events": sum(len(r["recovery_events"])
                               for r in results),
        "note": "steps execute sequentially within a run; the external "
                "check (verified) is the only check feedback. ground "
                "truth is the generator's state after each step.",
        "results": [{k: r[k] for k in ("run_id", "strategy", "horizon",
                                        "seed", "success",
                                        "fraction_correct",
                                        "final_state_correct",
                                        "total_tokens", "cost_usd",
                                        "retries")}
                    for r in sorted(results, key=lambda r: r["run_id"])],
    }
    (out_dir / "pilot_summary.json").write_text(json.dumps(summary,
                                                          indent=1))
    return summary
