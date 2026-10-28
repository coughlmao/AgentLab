"""teamscale: a scaling lab for agent teams.

Simulated agents, real harness: strategies, scheduling, budgets, sandboxed
code verification, and evaluation-artifact audits for studying how agent
teams scale with team size, compute budget, and task horizon.
"""
from .harness import RunConfig, RunResult, run_experiment
from .metrics import RunMetrics
from .strategies import STRATEGIES
from .tasks import FAMILIES, make_task

__version__ = "0.1.0"
__all__ = ["RunConfig", "RunResult", "run_experiment", "RunMetrics",
           "STRATEGIES", "FAMILIES", "make_task"]
