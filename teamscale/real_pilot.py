"""Pilot runner: REAL agent teams on the real codebase_repair family.

    python -m teamscale.real_pilot                 # the pilot grid, real API
    python -m teamscale.real_pilot --fake          # offline dry run (CI/tests)

Grid (defaults): team sizes {1, 2, 4, 8} x strategies {independent,
orchestrated, verified} x seeds {0, 1, 2} = 36 runs of the real task
generator (8 broken modules, checks executed for ground truth).

Difficulty: --difficulty (or env TEAMSCALE_DIFFICULTY; legacy alias
TEAMSCLAE_DIFFICULTY) selects the task generator's defect
catalogue -- "standard" (default; the original pilot)
or "hard" (subtler, multi-defect and cross-module defects). The grid,
strategies and result shapes are identical either way, and the summary
records which difficulty produced it.

Blind mode: --blind removes the run_check tool from the agents' own tool
set. Agents can read check scripts but cannot execute them; the prompt
says the work is checked externally after submission. The `verified`
strategy is unaffected in mechanism -- the runner always re-ran the check
externally -- but in blind mode that external verification is the ONLY
check feedback in the system. Blind mode is the Phase A laboratory
condition for the question "when does external verification improve team
performance?": the sighted pilot saturated because agents self-checked
inside every attempt, which masked any effect of strategy-level
verification.

Strategy semantics with real agents:
- independent:   agents get disjoint subtasks (round-robin), no shared
                 board, no visibility of each other's work.
- orchestrated:  a lead step assigns subtasks and maintains a shared board
                 (status + outcome per subtask, as text) that every agent
                 sees in its prompt.
- verified:      orchestrated + after a worker declares done, the runner
                 re-runs the check externally; on failure the check output
                 goes back to the same worker for exactly ONE retry.

Honest scope notes, also written into every summary this produces:
- agents execute sequentially inside a run (up to 4 *runs* proceed
  concurrently). This measures team organization -- assignment, visibility,
  verification -- not parallel wall-clock speedup.
- ground truth is strict: the check script's exit code, re-executed by the
  runner at the end of the run, never the agent's own claim.

Provider: env TEAMSCALE_PROVIDER (legacy alias TEAMSCLAE_PROVIDER)
selects the backend -- "openai" (default, OpenAI chat completions)
or "anthropic" (Claude Messages API).
The model comes from env TEAMSCALE_MODEL (legacy alias:
TEAMSCLAE_MODEL); the default follows the
provider (gpt-5-mini / claude-haiku-4-5). Results, traces, and the spend
ledger have the same shape either way, so runs stay comparable.

Caps and accounting:
- per-run token cap (default 400,000) and per-subtask budget.
- a global spend cap (env TEAMSCALE_SPEND_CAP_USD; legacy alias TEAMSCLAE_SPEND_CAP_USD, default $15) enforced
  through a JSON ledger updated after every run: a new run only starts if
  ledger_spent + worst_case_run_estimate <= cap, where the estimate is the
  per-run token cap priced at an 80/20 input/output mix.
- traces contain tool calls and usage only -- never prompts, never keys.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .anthropic_backend import AnthropicLLMBackend
from .fake_backend import FakeLLMBackend
from .llm_backend import OpenAILLMBackend, teamscale_env
from .real_tasks import check_all, make_real_task, materialize, run_check

# USD per 1M tokens: (input, output). Extend here for new models.
PRICES = {
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5-nano": (0.05, 0.40),
    "claude-haiku-4-5": (1.00, 5.00),
    # Haiku 5.5 short-prompt tier (prompts up to 100K tokens); this
    # pilot's subtask prompts stay far below that.
    "claude-haiku-5-5": (0.10, 0.50),
    "fake": (0.0, 0.0),
}

PER_RUN_TOKEN_CAP = 400_000
PER_SUBTASK_TOKEN_BUDGET = 75_000
STRATEGIES = ("independent", "orchestrated", "verified")
TEAM_SIZES = (1, 2, 4, 8)
SEEDS = (0, 1, 2)


def cost_usd(model: str, tokens_in: int, tokens_out: int,
             prices: dict = PRICES) -> float:
    pin, pout = prices.get(model, PRICES["gpt-5-mini"])
    return tokens_in / 1e6 * pin + tokens_out / 1e6 * pout


def worst_case_run_usd(model: str, per_run_token_cap: int = PER_RUN_TOKEN_CAP,
                       prices: dict = PRICES) -> float:
    """One run's worst case: the full token cap at an 80/20 in/out mix."""
    pin, pout = prices.get(model, PRICES["gpt-5-mini"])
    blended = 0.8 * pin + 0.2 * pout
    return per_run_token_cap / 1e6 * blended


class SpendLedger:
    """JSON cost ledger; the spend-cap gate for the whole pilot."""

    def __init__(self, path: Path, cap_usd: float, model: str):
        self.path = Path(path)
        self.lock = threading.Lock()
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.cap_usd = float(data.get("cap_usd", cap_usd))
            self.spent_usd = float(data.get("spent_usd", 0.0))
            self.runs = data.get("runs", [])
        else:
            self.cap_usd = float(cap_usd)
            self.spent_usd = 0.0
            self.runs = []
        self.model = model

    def can_afford(self, model: str | None = None,
                   per_run_token_cap: int = PER_RUN_TOKEN_CAP) -> bool:
        with self.lock:
            estimate = worst_case_run_usd(model or self.model,
                                          per_run_token_cap)
            return self.spent_usd + estimate <= self.cap_usd

    def record(self, result: dict):
        with self.lock:
            self.spent_usd = round(self.spent_usd + result["cost_usd"], 6)
            self.runs.append({"run_id": result["run_id"],
                              "cost_usd": result["cost_usd"],
                              "tokens_in": result["tokens_in"],
                              "tokens_out": result["tokens_out"]})
            self._save()

    def _save(self):
        data = {"cap_usd": self.cap_usd, "spent_usd": self.spent_usd,
                "model": self.model, "runs": self.runs}
        tmp = self.path.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(self.path)


class Board:
    """The shared team board for orchestrated/verified runs (plain text)."""

    def __init__(self):
        self.entries: dict[str, str] = {}
        self.lock = threading.Lock()

    def set(self, subtask_id: str, text: str):
        with self.lock:
            self.entries[subtask_id] = text

    def render(self) -> str:
        with self.lock:
            if not self.entries:
                return "(board is empty)"
            return "\n".join(f"{sid}: {txt}"
                             for sid, txt in sorted(self.entries.items()))


def run_id_for(backend_name: str, strategy: str, team_size: int, seed: int):
    return f"{backend_name}-codebase_repair-{strategy}-n{team_size}-s{seed}"


def run_single_run(*, strategy: str, team_size: int, seed: int,
                   backend_factory, out_dir: Path,
                   per_run_token_cap: int = PER_RUN_TOKEN_CAP,
                   per_subtask_budget: int = PER_SUBTASK_TOKEN_BUDGET,
                   model: str = "gpt-5-mini",
                   difficulty: str = "standard",
                   blind: bool = False) -> dict:
    """Execute one real run end to end and write its result + trace files."""
    out_dir = Path(out_dir)
    backend = backend_factory({"strategy": strategy, "team_size": team_size,
                               "seed": seed, "blind": blind})
    run_id = run_id_for(backend.name, strategy, team_size, seed)
    task = make_real_task(seed=seed, difficulty=difficulty)
    sandbox = out_dir / "sandboxes" / run_id
    if sandbox.exists():
        shutil.rmtree(sandbox)
    materialize(task, sandbox)

    trace_path = out_dir / "traces" / f"{run_id}.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_fh = open(trace_path, "w")

    def trace(event: dict):
        trace_fh.write(json.dumps(event) + "\n")

    started = time.perf_counter()
    trace({"event": "run_start", "run_id": run_id, "family": "codebase_repair",
           "strategy": strategy, "team_size": team_size, "seed": seed,
           "backend": backend.name, "difficulty": difficulty, "blind": blind,
           "note": "agents execute sequentially; this measures organization, "
                   "not parallel wall-clock"})

    initial = check_all(task, sandbox)  # sanity: everything starts broken
    board = Board() if strategy in ("orchestrated", "verified") else None
    buckets: dict[int, list] = {a: [] for a in range(team_size)}
    for i, module in enumerate(task.modules):
        agent = i % team_size
        buckets[agent].append(module)
        trace({"event": "assign", "subtask": module.subtask_id,
               "agent": agent, "file": module.module_file})
        if board is not None:
            board.set(module.subtask_id,
                      f"assigned to agent {agent} ({module.module_file})")

    tokens_in = tokens_out = retries = 0
    attempts: dict[str, int] = {}
    errors: dict[str, str] = {}

    def budget_left() -> int:
        return per_run_token_cap - (tokens_in + tokens_out)

    for agent in range(team_size):
        for module in buckets[agent]:
            if budget_left() <= 0:
                errors[module.subtask_id] = "run token cap reached"
                continue
            budget = min(per_subtask_budget, budget_left())
            fix = backend.fix_module(
                module, sandbox,
                board_text=board.render() if board is not None else "",
                token_budget=budget, agent_id=agent)
            attempts[module.subtask_id] = attempts.get(module.subtask_id, 0) + 1
            tokens_in += fix.tokens_in
            tokens_out += fix.tokens_out
            for ev in fix.events:
                trace(ev)
            if fix.error:
                errors[module.subtask_id] = fix.error
            if board is not None:
                board.set(module.subtask_id,
                          f"agent {agent} declared done"
                          if fix.declared_done else
                          f"agent {agent} stopped ({fix.error or 'budget'})")
            if strategy == "verified":
                passed, out = run_check(sandbox, module)
                trace({"event": "verify", "subtask": module.subtask_id,
                       "agent": agent, "passed": passed})
                if not passed and not fix.error and budget_left() > 0:
                    retries += 1
                    trace({"event": "retry", "subtask": module.subtask_id,
                           "agent": agent})
                    fix2 = backend.fix_module(
                        module, sandbox,
                        board_text=board.render() if board is not None else "",
                        feedback=out, token_budget=min(per_subtask_budget,
                                                       budget_left()),
                        agent_id=agent)
                    attempts[module.subtask_id] += 1
                    tokens_in += fix2.tokens_in
                    tokens_out += fix2.tokens_out
                    for ev in fix2.events:
                        trace(ev)
                    if fix2.error:
                        errors[module.subtask_id] = fix2.error

    final = check_all(task, sandbox)  # strict ground truth, re-executed
    n_correct = sum(1 for v in final.values() if v)
    wall = round(time.perf_counter() - started, 3)
    result = {
        "run_id": run_id, "backend": backend.name, "model": model,
        "family": "codebase_repair", "difficulty": difficulty,
        "blind": blind,
        "strategy": strategy,
        "team_size": team_size, "seed": seed,
        "success": n_correct == len(task.modules),
        "fraction_correct": round(n_correct / len(task.modules), 4),
        "n_subtasks": len(task.modules), "n_correct": n_correct,
        "tokens_in": tokens_in, "tokens_out": tokens_out,
        "total_tokens": tokens_in + tokens_out,
        "cost_usd": round(cost_usd(model, tokens_in, tokens_out), 6),
        "wall_time_s": wall, "retries": retries,
        "duplicates": 0,  # round-robin assignment is disjoint by construction
        "attempts": attempts, "errors": errors,
        "initial_all_broken": not any(initial.values()),
        "per_subtask_correct": final,
    }
    trace({"event": "run_end", "run_id": run_id, "success": result["success"],
           "fraction_correct": result["fraction_correct"],
           "total_tokens": result["total_tokens"],
           "cost_usd": result["cost_usd"], "wall_time_s": wall})
    trace_fh.close()
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{run_id}.json").write_text(json.dumps(result, indent=1))
    result["trace_path"] = str(trace_path)
    return result


def run_pilot(*, out_dir: Path, backend_factory, model: str,
              sizes=TEAM_SIZES, strategies=STRATEGIES, seeds=SEEDS,
              spend_cap_usd: float = 15.0, workers: int = 4,
              per_run_token_cap: int = PER_RUN_TOKEN_CAP,
              ledger_name: str = "ledger.json",
              difficulty: str = "standard",
              blind: bool = False) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = SpendLedger(out_dir / ledger_name, spend_cap_usd, model)
    configs = [(s, n, seed) for s in strategies for n in sizes for seed in seeds]
    results, skipped, failed = [], [], []
    worst = worst_case_run_usd(model, per_run_token_cap)

    def one(cfg):
        strategy, n, seed = cfg
        return run_single_run(strategy=strategy, team_size=n, seed=seed,
                              backend_factory=backend_factory, out_dir=out_dir,
                              per_run_token_cap=per_run_token_cap, model=model,
                              difficulty=difficulty, blind=blind)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        reserved = 0.0  # worst-case estimates of runs submitted, not yet done
        for cfg in configs:
            if ledger.spent_usd + reserved + worst > ledger.cap_usd:
                skipped.append({"strategy": cfg[0], "team_size": cfg[1],
                                "seed": cfg[2], "reason": "spend cap gate"})
                continue
            reserved += worst
            futures[pool.submit(one, cfg)] = cfg
        for fut, cfg in futures.items():
            try:
                result = fut.result()
            except Exception as e:  # one bad run must not kill the pilot
                failed.append({"strategy": cfg[0], "team_size": cfg[1],
                               "seed": cfg[2], "error": f"{type(e).__name__}: {e}"})
                continue
            ledger.record(result)
            results.append(result)

    summary = {
        "backend": results[0]["backend"] if results else "none",
        "model": model, "family": "codebase_repair",
        "difficulty": difficulty, "blind": blind,
        "grid": {"team_sizes": list(sizes), "strategies": list(strategies),
                 "seeds": list(seeds)},
        "runs_completed": len(results), "runs_skipped_spend_cap": skipped,
        "runs_failed": failed,
        "total_tokens_in": sum(r["tokens_in"] for r in results),
        "total_tokens_out": sum(r["tokens_out"] for r in results),
        "total_cost_usd": round(sum(r["cost_usd"] for r in results), 4),
        "ledger_spent_usd": ledger.spent_usd, "spend_cap_usd": ledger.cap_usd,
        "mean_fraction_correct": (
            round(sum(r["fraction_correct"] for r in results) / len(results), 4)
            if results else 0.0),
        "note": "agents execute sequentially within a run; team organization "
                "is what varies. ground truth is re-executed check exit codes.",
        "results": [{k: r[k] for k in ("run_id", "strategy", "team_size",
                                        "seed", "success", "fraction_correct",
                                        "total_tokens", "cost_usd", "retries")}
                    for r in sorted(results, key=lambda r: r["run_id"])],
    }
    (out_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="real-agent pilot grid")
    ap.add_argument("--fake", action="store_true",
                    help="offline fake backend (dry run; no network, $0)")
    ap.add_argument("--fake-competence", type=float, default=0.85)
    ap.add_argument("--sizes", default="1,2,4,8")
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="experiments/real")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--per-run-token-cap", type=int, default=PER_RUN_TOKEN_CAP)
    ap.add_argument("--spend-cap", type=float, default=None,
                    help="default: env TEAMSCALE_SPEND_CAP_USD "
                         "(legacy alias: TEAMSCLAE_SPEND_CAP_USD) or 15.00")
    ap.add_argument("--difficulty", default=None,
                    help="'standard' (default) or 'hard'; "
                         "env TEAMSCALE_DIFFICULTY also works"
                         "(legacy alias: TEAMSCLAE_DIFFICULTY)")
    ap.add_argument("--blind", action="store_true",
                    help="remove the run_check tool from the agents' own "
                         "tool set; checks happen only externally "
                         "(runner-side, as in the verified strategy)")
    args = ap.parse_args(argv)

    spend_cap = (args.spend_cap if args.spend_cap is not None else
                 float(teamscale_env("SPEND_CAP_USD", "15.00")))
    difficulty = (args.difficulty
                  or teamscale_env("DIFFICULTY")
                  or "standard").strip().lower()
    if difficulty not in ("standard", "hard"):
        ap.error(f"unknown difficulty {difficulty!r}: "
                 "use 'standard' or 'hard'")
    sizes = tuple(int(x) for x in args.sizes.split(",") if x)
    strategies = tuple(x.strip() for x in args.strategies.split(",") if x)
    seeds = tuple(int(x) for x in args.seeds.split(",") if x)
    out_dir = Path(args.out)

    if args.fake:
        model = "fake"
        factory = lambda cfg: FakeLLMBackend(
            competence=args.fake_competence,
            seed=cfg["seed"] * 100 + cfg["team_size"],
            blind=cfg.get("blind", False))
        ledger_name = "ledger_fake.json"
    else:
        provider = teamscale_env("PROVIDER", "openai").strip().lower()
        if provider == "anthropic":
            model = teamscale_env("MODEL", "claude-haiku-4-5")
            AnthropicLLMBackend(model=model)  # fail fast if no credential
            factory = lambda cfg: AnthropicLLMBackend(
                model=model, blind=cfg.get("blind", False))
        elif provider == "openai":
            model = teamscale_env("MODEL", "gpt-5-mini")
            OpenAILLMBackend(model=model)  # fail fast if no credential
            factory = lambda cfg: OpenAILLMBackend(
                model=model, blind=cfg.get("blind", False))
        else:
            ap.error(f"unknown TEAMSCALE_PROVIDER (legacy alias: "
                     f"TEAMSCLAE_PROVIDER) {provider!r}: "
                     "use 'openai' or 'anthropic'")
        ledger_name = "ledger.json"

    summary = run_pilot(out_dir=out_dir, backend_factory=factory, model=model,
                        sizes=sizes, strategies=strategies, seeds=seeds,
                        spend_cap_usd=spend_cap, workers=args.workers,
                        per_run_token_cap=args.per_run_token_cap,
                        ledger_name=ledger_name, difficulty=difficulty,
                        blind=args.blind)
    print(f"runs completed: {summary['runs_completed']}  "
          f"skipped (spend cap): {len(summary['runs_skipped_spend_cap'])}")
    print(f"tokens: {summary['total_tokens_in']} in / "
          f"{summary['total_tokens_out']} out   "
          f"cost: ${summary['total_cost_usd']}  "
          f"ledger: ${summary['ledger_spent_usd']} of ${summary['spend_cap_usd']}")
    print(f"mean fraction correct: {summary['mean_fraction_correct']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
