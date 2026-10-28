"""Run metrics: what one (family, strategy, team, budget, seed) run produced."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class RunMetrics:
    family: str
    strategy: str
    team_size: int
    budget: int
    seed: int
    horizon_scale: float = 1.0
    success: bool = False            # every subtask verified correct
    fraction_correct: float = 0.0
    n_subtasks: int = 0
    n_correct: int = 0
    total_tokens: int = 0            # work + coordination + verification
    work_tokens: int = 0
    coordination_tokens: int = 0     # lead assignment overhead + verifier passes
    wall_time: float = 0.0           # simulated seconds
    duplication_rate: float = 0.0    # duplicate attempts per subtask
    duplicate_attempts: int = 0
    retries: int = 0
    skipped: int = 0                 # subtasks never attempted (budget)
    per_agent_load: dict = field(default_factory=dict)  # agent -> {tasks, tokens}

    def to_dict(self) -> dict:
        return asdict(self)


CSV_FIELDS = ["family", "strategy", "team_size", "budget", "seed", "horizon_scale",
              "success", "fraction_correct", "n_subtasks", "n_correct",
              "total_tokens", "work_tokens", "coordination_tokens", "wall_time",
              "duplication_rate", "duplicate_attempts", "retries", "skipped"]


def csv_row(m: RunMetrics) -> dict:
    d = m.to_dict()
    return {k: d[k] for k in CSV_FIELDS}
