from teamscale.audits import lenient_vs_strict, no_work_baseline, seed_sensitivity
from teamscale.harness import RunConfig, run_experiment
from teamscale.strategies import difficulty_allocation
from teamscale.tasks import make_task

BIG = 10 ** 9


def test_allocation_respects_budget_and_difficulty():
    task = make_task("research_sweep", seed=0)
    budget = task.total_expected_tokens() * 3
    alloc = difficulty_allocation(task, budget)
    assert sum(alloc.values()) <= budget + len(task.subtasks)  # int rounding slack
    hardest = max(task.subtasks, key=lambda s: s.difficulty)
    easiest = min(task.subtasks, key=lambda s: s.difficulty)
    assert alloc[hardest.subtask_id] >= alloc[easiest.subtask_id]


def test_budget_aware_spends_within_budget():
    for family in ("research_sweep", "long_horizon"):
        task = make_task(family, seed=0)
        budget = int(task.total_expected_tokens() * 1.5)
        for seed in range(3):
            m = run_experiment(RunConfig(family=family, strategy="budget_aware",
                                         team_size=8, budget=budget, seed=seed)).metrics
            assert m.total_tokens <= budget, (family, seed, m.total_tokens, budget)


def test_lenient_never_below_strict():
    result = run_experiment(RunConfig(family="research_sweep", strategy="orchestrated",
                                      team_size=4, budget=BIG, seed=2))
    strict, lenient = lenient_vs_strict(result)
    assert lenient >= strict
    assert strict == result.metrics.fraction_correct


def test_no_work_baseline_is_at_floor():
    task = make_task("research_sweep", seed=0)
    strict, lenient = no_work_baseline(task)
    assert strict == 0.0           # guessing topics never produces a real value
    assert 0.0 < lenient <= 0.5    # but a topic-only guess still farms partial credit


def test_seed_sensitivity_reports_variation():
    cfg = RunConfig(family="research_sweep", strategy="orchestrated",
                    team_size=4, budget=BIG, seed=0)
    fractions = seed_sensitivity(cfg, seeds=(0, 1, 2, 3, 4))
    assert len(fractions) == 5
    assert all(0.0 <= f <= 1.0 for f in fractions)


def test_bend_point_logic():
    from teamscale.experiment import _bend
    assert _bend([1, 2, 4, 8], [0.1, 0.5, 0.51, 0.52]) == 4  # gain 0.01 < 0.02 at n=4
    assert _bend([1, 2, 4, 8], [0.1, 0.4, 0.7, 1.0]) == 8   # never bends -> last size
