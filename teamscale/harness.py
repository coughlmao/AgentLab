"""Experiment harness: runs one configuration as a discrete-event simulation.

Time is simulated: an attempt occupies its agent for its duration, subtasks
unlock when their dependencies finish, and the run's wall time is the finish
time of the last job. All randomness flows from one seeded RNG, so a config
plus a seed fully determines the run (and its trace).

Budget is a hard cap on total tokens (work + coordination + verification).
A subtask that cannot be afforded is marked failed-unattempted (`skipped`)
and its dependents proceed with untrustworthy inputs -- error propagation,
not a crash.
"""
from __future__ import annotations

import heapq
import itertools
import json
import random
import shutil
import zlib
from dataclasses import dataclass, field
from pathlib import Path

from .agents import SimulatedAgent, AttemptContext
from .metrics import RunMetrics
from .strategies import STRATEGIES, StrategySpec, difficulty_allocation
from .tasks import Task, make_task, strict_verify


@dataclass
class RunConfig:
    family: str
    strategy: str
    team_size: int = 1
    budget: int = 12000
    seed: int = 0
    horizon_scale: float = 1.0
    trace_dir: str | None = None
    sandbox_root: str | None = None

    @property
    def run_id(self) -> str:
        return (f"{self.family}-{self.strategy}-n{self.team_size}"
                f"-b{self.budget}-h{self.horizon_scale}-s{self.seed}")


@dataclass
class RunResult:
    metrics: RunMetrics
    outputs: dict           # subtask_id -> final payload
    correct: dict           # subtask_id -> bool (strict ground truth)
    task: Task
    trace_path: str | None = None


def _seed_for(config: RunConfig) -> int:
    # solo always runs one agent: its stream must not change with the
    # requested team size, or the solo "curve" would be pure seed noise.
    effective_team = 1 if STRATEGIES[config.strategy].force_single_agent else config.team_size
    key = f"{config.family}|{config.strategy}|{effective_team}|{config.seed}|{config.horizon_scale}"
    return zlib.crc32(key.encode()) & 0xFFFFFFFF


class _Job:
    _seq = itertools.count()

    def __init__(self, kind, subtask, agent, finish, payload, correct, tokens):
        self.kind = kind                    # "primary" | "duplicate"
        self.subtask = subtask
        self.agent = agent
        self.finish = finish
        self.payload = payload
        self.correct = correct
        self.tokens = tokens                # work tokens charged for this job
        self.seq = next(_Job._seq)


def run_experiment(config: RunConfig) -> RunResult:
    spec: StrategySpec = STRATEGIES[config.strategy]
    task = make_task(config.family, seed=config.seed, horizon_scale=config.horizon_scale)
    rng = random.Random(_seed_for(config))
    n_agents = 1 if spec.force_single_agent else max(1, config.team_size)
    agents = [SimulatedAgent(agent_id=i) for i in range(n_agents)]
    allocation = difficulty_allocation(task, config.budget) if spec.budget_aware else {}

    sandbox = None
    if config.family == "codebase_repair":
        root = Path(config.sandbox_root or "experiments/sandboxes")
        sandbox = root / config.run_id
        if sandbox.exists():
            shutil.rmtree(sandbox)
        sandbox.mkdir(parents=True, exist_ok=True)

    trace_fh = None
    trace_path = None
    if config.trace_dir:
        Path(config.trace_dir).mkdir(parents=True, exist_ok=True)
        trace_path = str(Path(config.trace_dir) / f"{config.run_id}.jsonl")
        trace_fh = open(trace_path, "w")

    def trace(event: dict):
        if trace_fh:
            trace_fh.write(json.dumps(event) + "\n")

    by_id = task.by_id()
    order = {s.subtask_id: i for i, s in enumerate(task.subtasks)}
    final: dict[str, dict] = {}
    pending = {s.subtask_id for s in task.subtasks}
    in_flight: list = []
    busy: set[int] = set()
    now = 0.0
    work_tokens = 0
    coord_tokens = 0
    retries = 0
    duplicate_attempts = 0
    skipped = 0
    spent = lambda: work_tokens + coord_tokens

    trace({"t": 0.0, "event": "run_start", "run_id": config.run_id,
           "family": config.family, "strategy": config.strategy,
           "team_size": n_agents, "budget": config.budget})

    def finalize(sid, payload, correct, agent_id, tokens, when, note=""):
        final[sid] = {"payload": payload, "correct": correct}
        trace({"t": round(when, 3), "event": "subtask_done", "subtask": sid,
               "agent": agent_id, "tokens": tokens, "correct": correct, "note": note})

    while len(final) < len(task.subtasks) or in_flight:
        ready = [by_id[sid] for sid in pending
                 if all(d in final for d in by_id[sid].deps)]
        if spec.budget_aware:
            ready.sort(key=lambda s: (-s.difficulty, order[s.subtask_id]))
        else:
            ready.sort(key=lambda s: order[s.subtask_id])
        free = [a for a in agents if a.agent_id not in busy and a.available_at <= now + 1e-9]

        progressed = False
        for subtask in ready:
            if not free:
                break
            agent = (rng.choice(free) if spec.name == "independent"
                     else min(free, key=lambda a: (a.tasks_done, a.available_at)))
            ctx = AttemptContext(
                agent_prior_tasks=agent.tasks_done,
                deps_correct=all(final[d]["correct"] for d in subtask.deps),
                depth=task.depth_of(subtask), horizon_scale=config.horizon_scale)
            steps, tokens, duration = agent.plan(subtask, rng)
            coord = spec.coordination_tokens(n_agents)
            afford = spent() + tokens + coord <= config.budget
            if spec.budget_aware:
                afford = afford and tokens <= allocation.get(subtask.subtask_id, 0)
            sid = subtask.subtask_id
            pending.discard(sid)
            progressed = True
            if not afford:
                skipped += 1
                finalize(sid, "", False, agent.agent_id, 0, now, note="skipped_budget")
                continue
            outcome = agent.attempt(subtask, ctx, rng, plan=(steps, tokens, duration))
            job_tokens, job_dur = tokens, outcome.duration
            correct = strict_verify(subtask, outcome.payload, sandbox)
            payload = outcome.payload
            trace({"t": round(now, 3), "event": "assign", "subtask": sid,
                   "agent": agent.agent_id, "tokens": tokens, "coord_tokens": coord})
            if spec.verify:
                verify_cost = int(round(0.12 * tokens)) or 1
                coord_tokens += verify_cost
                job_dur += 0.2 * outcome.duration
                trace({"t": round(now, 3), "event": "verify", "subtask": sid,
                       "agent": agent.agent_id, "tokens": verify_cost, "correct": correct})
                if not correct:
                    remaining = config.budget - (spent() + job_tokens)
                    low = spec.budget_aware and remaining < 0.15 * config.budget
                    if not low:
                        r_steps, r_tokens, r_dur = agent.plan(subtask, rng)
                        if spent() + job_tokens + r_tokens <= config.budget:
                            r_ctx = AttemptContext(agent_prior_tasks=agent.tasks_done,
                                                   deps_correct=ctx.deps_correct,
                                                   depth=ctx.depth,
                                                   horizon_scale=config.horizon_scale,
                                                   is_retry=True)
                            r_out = agent.attempt(subtask, r_ctx, rng,
                                                  plan=(r_steps, r_tokens, r_dur))
                            retries += 1
                            job_tokens += r_tokens
                            job_dur += r_out.duration
                            payload = r_out.payload
                            correct = strict_verify(subtask, payload, sandbox)
                            trace({"t": round(now, 3), "event": "retry", "subtask": sid,
                                   "agent": agent.agent_id, "tokens": r_tokens,
                                   "correct": correct})
            work_tokens += job_tokens
            coord_tokens += coord
            agent.tasks_done += 1
            agent.tokens_spent += job_tokens
            finish = now + job_dur
            agent.available_at = finish
            busy.add(agent.agent_id)
            heapq.heappush(in_flight, (finish, _Job("primary", subtask, agent,
                                                    finish, payload, correct, job_tokens).seq,
                                        {"kind": "primary", "subtask": subtask, "agent": agent,
                                         "payload": payload, "correct": correct, "tokens": job_tokens}))
            free = [a for a in free if a.agent_id != agent.agent_id]
            # independent teams may duplicate work: a shadow attempt whose
            # result is discarded but whose tokens are really spent.
            if spec.duplicate_prob and n_agents > 1 and free and rng.random() < spec.duplicate_prob:
                dup_agent = rng.choice(free)
                d_steps, d_tokens, d_dur = dup_agent.plan(subtask, rng)
                if spent() + d_tokens <= config.budget:
                    d_out = dup_agent.attempt(subtask, ctx, rng, plan=(d_steps, d_tokens, d_dur))
                    work_tokens += d_tokens
                    duplicate_attempts += 1
                    dup_agent.tasks_done += 1
                    dup_agent.tokens_spent += d_tokens
                    d_finish = now + d_out.duration
                    dup_agent.available_at = d_finish
                    busy.add(dup_agent.agent_id)
                    heapq.heappush(in_flight, (d_finish, next(_Job._seq),
                                                {"kind": "duplicate", "subtask": subtask,
                                                 "agent": dup_agent, "payload": d_out.payload,
                                                 "correct": False, "tokens": d_tokens}))
                    trace({"t": round(now, 3), "event": "duplicate", "subtask": sid,
                           "agent": dup_agent.agent_id, "tokens": d_tokens})
                    free = [a for a in free if a.agent_id != dup_agent.agent_id]

        if in_flight:
            finish, _, job = heapq.heappop(in_flight)
            now = max(now, finish)
            busy.discard(job["agent"].agent_id)
            if job["kind"] == "primary":
                finalize(job["subtask"].subtask_id, job["payload"], job["correct"],
                         job["agent"].agent_id, job["tokens"], finish)
            else:
                trace({"t": round(finish, 3), "event": "duplicate_done",
                       "subtask": job["subtask"].subtask_id,
                       "agent": job["agent"].agent_id, "tokens": job["tokens"]})
        elif not progressed:
            # Nothing running and nothing assignable: deps can only be blocked
            # by skipped/finalized nodes, which do finalize -- so this is a
            # guard against a pathological cycle, not an expected path.
            for sid in list(pending):
                pending.discard(sid)
                skipped += 1
                finalize(sid, "", False, -1, 0, now, note="blocked")
            break

    n_correct = sum(1 for v in final.values() if v["correct"])
    metrics = RunMetrics(
        family=config.family, strategy=config.strategy, team_size=n_agents,
        budget=config.budget, seed=config.seed, horizon_scale=config.horizon_scale,
        success=n_correct == len(task.subtasks) and len(task.subtasks) > 0,
        fraction_correct=round(n_correct / len(task.subtasks), 4) if task.subtasks else 0.0,
        n_subtasks=len(task.subtasks), n_correct=n_correct,
        total_tokens=work_tokens + coord_tokens, work_tokens=work_tokens,
        coordination_tokens=coord_tokens, wall_time=round(now, 3),
        duplication_rate=round(duplicate_attempts / len(task.subtasks), 4) if task.subtasks else 0.0,
        duplicate_attempts=duplicate_attempts, retries=retries, skipped=skipped,
        per_agent_load={str(a.agent_id): {"tasks": a.tasks_done, "tokens": a.tokens_spent}
                        for a in agents})
    trace({"t": round(now, 3), "event": "run_end", "run_id": config.run_id,
           "success": metrics.success, "fraction_correct": metrics.fraction_correct,
           "total_tokens": metrics.total_tokens, "wall_time": metrics.wall_time})
    if trace_fh:
        trace_fh.close()
    return RunResult(metrics=metrics,
                     outputs={sid: v["payload"] for sid, v in final.items()},
                     correct={sid: v["correct"] for sid, v in final.items()},
                     task=task, trace_path=trace_path)
