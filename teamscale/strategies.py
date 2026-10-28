"""Coordination strategies.

A strategy is a policy the harness executes:

solo          -- exactly one agent, everything sequential, no overhead.
independent   -- N agents split the work with NO coordination: no dedup
                 (a subtask can be done twice, wasting tokens) and no
                 verification (first result stands, right or wrong).
orchestrated  -- a lead assigns each subtask exactly once to the least-loaded
                 free agent and dedups results; pays coordination tokens.
verified      -- orchestrated + every result is re-checked by the ground-truth
                 verifier; failures are retried once. Costs verification
                 tokens and retry compute, catches errors.
budget_aware  -- orchestrated, but the total token budget is allocated across
                 subtasks in proportion to difficulty before the run starts.
                 A subtask is only attempted while its allocation (and the
                 global budget) covers the attempt's planned cost, and
                 retries are skipped once the run's budget is nearly gone:
                 money goes to first attempts on hard subtasks, not to
                 re-rolling easy ones.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategySpec:
    name: str
    force_single_agent: bool = False
    duplicate_prob: float = 0.0       # chance a subtask gets a shadow duplicate
    verify: bool = False              # verifier pass + one retry on failure
    budget_aware: bool = False

    def coordination_tokens(self, team_size: int) -> int:
        """Per-assignment coordination overhead charged by the lead."""
        if self.name in ("solo", "independent"):
            return 0
        base = 30 + 3 * team_size
        return base + 10 if self.budget_aware else base


STRATEGIES = {
    "solo": StrategySpec("solo", force_single_agent=True),
    "independent": StrategySpec("independent", duplicate_prob=0.30),
    "orchestrated": StrategySpec("orchestrated"),
    "verified": StrategySpec("verified", verify=True),
    "budget_aware": StrategySpec("budget_aware", budget_aware=True),
}


def difficulty_allocation(task, budget: int, tokens_per_step: int = 100) -> dict:
    """Split `budget` across subtasks in proportion to difficulty.

    Each subtask first reserves its worst-case attempt cost (max steps at its
    difficulty price). Whatever budget remains after those reserves is then
    distributed in proportion to difficulty, so hard subtasks hold the largest
    pools for retries and variance. If the budget cannot cover all worst-case
    costs, reserves are scaled down proportionally and the harness's
    difficulty-first ordering decides what gets attempted.
    Returns {subtask_id: tokens}.
    """
    subs = task.subtasks
    if not subs or budget <= 0:
        return {s.subtask_id: 0 for s in subs}
    worst = {s.subtask_id: int(round((s.expected_steps + 2) * tokens_per_step
                                     * (1 + 0.5 * s.difficulty))) for s in subs}
    total_worst = sum(worst.values())
    if budget <= total_worst:
        return {sid: int(budget * w / total_worst) for sid, w in worst.items()}
    alloc = dict(worst)
    leftover = budget - total_worst
    diff_total = sum(s.difficulty for s in subs) or 1.0
    for s in subs:
        alloc[s.subtask_id] += int(leftover * s.difficulty / diff_total)
    return alloc
