"""Agent model.

v1 agents are *simulated*: an attempt's outcome is drawn from an explicit
probability model (difficulty, accumulated context load, dependency health)
using a seeded RNG, so the whole experiment grid runs offline, deterministically,
with no API keys. The science in this repo is in the harness, the coordination
strategies, and the evaluation -- not in any claim about real LLM behaviour.

`AgentBackend` is the seam for plugging in a real LLM backend later: it only
has to turn a subtask into a payload (an answer string, or source code for the
codebase_repair family).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class AttemptContext:
    """What the harness tells an agent about the conditions of an attempt."""

    agent_prior_tasks: int = 0        # context load carried by this agent
    deps_correct: bool = True         # were this subtask's inputs trustworthy?
    depth: int = 0                    # position in the dependency DAG
    horizon_scale: float = 1.0        # experiment knob: stretches horizon effects
    is_retry: bool = False


@dataclass
class AttemptOutcome:
    steps: int
    tokens: int
    duration: float                   # simulated seconds the agent is occupied
    payload: str                      # answer string, or source code for code tasks
    p_correct: float                  # probability the model assigned to success


@runtime_checkable
class AgentBackend(Protocol):
    """Interface a real LLM backend must implement to replace the simulator.

    `attempt` receives the subtask and the attempt context and must return an
    AttemptOutcome whose payload is the agent's answer (or, for code subtasks,
    the full file contents). Token/step accounting is the backend's job.
    """

    def attempt(self, subtask, ctx: AttemptContext, rng: random.Random) -> AttemptOutcome:
        ...


@dataclass
class SimulatedAgent:
    """Deterministic-under-seed simulated worker.

    Per-step token cost and latency are configurable. Error probability rises
    with subtask difficulty, with the agent's accumulated context load (which
    is why handing subtasks to fresh teammates can help), with depth in a
    dependency chain scaled by horizon_scale, and sharply when a dependency
    produced a wrong result (error propagation along the horizon).
    """

    agent_id: int
    tokens_per_step: int = 100
    step_latency: float = 1.0
    base_skill: float = 0.93
    tasks_done: int = field(default=0, init=False)
    tokens_spent: int = field(default=0, init=False)
    busy_time: float = field(default=0.0, init=False)
    available_at: float = field(default=0.0, init=False)

    def success_probability(self, subtask, ctx: AttemptContext) -> float:
        p = self.base_skill
        p -= 0.35 * subtask.difficulty
        p -= 0.015 * ctx.agent_prior_tasks * ctx.horizon_scale
        p -= 0.010 * ctx.depth * ctx.horizon_scale
        if not ctx.deps_correct:
            p -= 0.30
        if ctx.is_retry:
            p += 0.15  # a retry knows the first attempt failed; it is not blind
        return max(0.20, min(0.98, p))

    def plan(self, subtask, rng: random.Random):
        """Deterministic cost of an attempt, known before it runs.

        The harness uses this for budget checks; `attempt` reuses the same
        numbers so planned and charged costs can never drift apart.
        """
        steps = subtask.expected_steps + (1 if rng.random() < 0.35 else 0)
        tokens = int(round(steps * self.tokens_per_step * (1.0 + 0.5 * subtask.difficulty)))
        duration = steps * self.step_latency * rng.uniform(0.9, 1.1)
        return steps, tokens, duration

    def attempt(self, subtask, ctx: AttemptContext, rng: random.Random,
                plan=None) -> AttemptOutcome:
        steps, tokens, duration = plan if plan is not None else self.plan(subtask, rng)
        p = self.success_probability(subtask, ctx)
        if rng.random() < p:
            payload = subtask.ground_truth_payload()
        else:
            payload = subtask.distractor_payload(rng)
        return AttemptOutcome(steps=steps, tokens=tokens, duration=duration,
                              payload=payload, p_correct=p)


class LLMBackend:
    """Stub for a future real-LLM backend (e.g. an API-driven agent).

    Deliberately unimplemented: v1 ships with simulated agents only, so no
    result in this repository should be read as a measurement of a real model.
    Implement `attempt` against your provider, returning the model's payload
    plus its true token usage, and pass the backend to the harness.
    """

    def __init__(self, model: str = "unset"):
        self.model = model

    def attempt(self, subtask, ctx: AttemptContext, rng: random.Random) -> AttemptOutcome:
        raise NotImplementedError(
            "LLMBackend is a stub: plug in a real model client here. "
            "v1 experiments use SimulatedAgent and run fully offline."
        )
