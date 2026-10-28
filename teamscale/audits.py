"""Evaluation-artifact audits.

The point of an audit is to ask what a score is *made of* before trusting it:
how much of it a lenient scorer gives away, how much a team that does no work
at all would score, and how much of it survives changing the seed. Inspired
by the evaluation-artifacts analysis in "Unsolvability Ceiling in Multi-LLM
Routing" -- measured gains are only real once the artifact components are
subtracted.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from .harness import RunConfig, run_experiment
from .tasks import lenient_score


@dataclass
class AuditReport:
    config: RunConfig
    strict_fraction: float
    lenient_fraction: float
    inflation_delta: float          # lenient - strict, in fraction points
    no_work_strict: float
    no_work_lenient: float
    seed_fractions: list
    seed_mean: float
    seed_stdev: float
    seed_range: float

    def summary(self) -> str:
        c = self.config
        lines = [
            f"artifact report: {c.family} / {c.strategy} / team={c.team_size} / budget={c.budget}",
            f"  strict score (this run)      : {self.strict_fraction:.3f}",
            f"  lenient re-score             : {self.lenient_fraction:.3f}  "
            f"(inflation +{self.inflation_delta:.3f})",
            f"  no-work baseline             : strict {self.no_work_strict:.3f} / "
            f"lenient {self.no_work_lenient:.3f}  <- what the eval gives away for free",
            f"  seed sensitivity ({len(self.seed_fractions)} seeds)   : mean {self.seed_mean:.3f} "
            f"+/- {self.seed_stdev:.3f}, range {self.seed_range:.3f}",
            f"  seeds                        : {[round(f, 3) for f in self.seed_fractions]}",
        ]
        return "\n".join(lines)


def lenient_vs_strict(result) -> tuple[float, float]:
    """Re-score a finished run's outputs both ways."""
    task = result.task
    sandbox = None  # lenient code scoring only parses; no execution needed
    strict = result.metrics.fraction_correct
    scores = [lenient_score(s, result.outputs.get(s.subtask_id, ""), sandbox)
              for s in task.subtasks]
    return strict, round(sum(scores) / len(scores), 4) if scores else 0.0


def no_work_baseline(task) -> tuple[float, float]:
    """Score a 'team' that returns topic-only guesses without doing any work.

    Its strict score should sit at the floor (it never produces a real value);
    its lenient score exposes how much partial credit the scorer hands out.
    """
    strict_hits, lenient_total = 0, 0.0
    for s in task.subtasks:
        guess = s.broken_code if s.family == "codebase_repair" else f"{s.topic}:"
        if s.family == "codebase_repair":
            strict_hits += 0  # unmodified broken code passes no checks
        else:
            strict_hits += 1 if guess == s.answer else 0
        lenient_total += lenient_score(s, guess, None)
    n = len(task.subtasks) or 1
    return round(strict_hits / n, 4), round(lenient_total / n, 4)


def seed_sensitivity(config: RunConfig, seeds) -> list[float]:
    fractions = []
    for seed in seeds:
        cfg = RunConfig(family=config.family, strategy=config.strategy,
                        team_size=config.team_size, budget=config.budget,
                        seed=seed, horizon_scale=config.horizon_scale)
        fractions.append(run_experiment(cfg).metrics.fraction_correct)
    return fractions


def run_audits(config: RunConfig, seeds=(0, 1, 2, 3, 4)) -> AuditReport:
    result = run_experiment(config)
    strict, lenient = lenient_vs_strict(result)
    nw_strict, nw_lenient = no_work_baseline(result.task)
    fractions = seed_sensitivity(config, seeds)
    return AuditReport(
        config=config, strict_fraction=strict, lenient_fraction=lenient,
        inflation_delta=round(lenient - strict, 4),
        no_work_strict=nw_strict, no_work_lenient=nw_lenient,
        seed_fractions=fractions,
        seed_mean=round(statistics.fmean(fractions), 4),
        seed_stdev=round(statistics.pstdev(fractions), 4) if len(fractions) > 1 else 0.0,
        seed_range=round(max(fractions) - min(fractions), 4) if fractions else 0.0)
