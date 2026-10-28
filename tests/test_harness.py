import json

from teamscale.harness import RunConfig, run_experiment

BIG = 10 ** 9


def run(family="research_sweep", strategy="orchestrated", team=4, seed=0, **kw):
    return run_experiment(RunConfig(family=family, strategy=strategy,
                                    team_size=team, budget=BIG, seed=seed, **kw))


def test_deterministic_under_seed():
    a = run(seed=7).metrics.to_dict()
    b = run(seed=7).metrics.to_dict()
    assert a == b


def test_metrics_are_internally_consistent():
    m = run(team=4).metrics
    assert m.total_tokens == m.work_tokens + m.coordination_tokens
    assert abs(m.fraction_correct - m.n_correct / m.n_subtasks) < 1e-9
    assert m.success == (m.n_correct == m.n_subtasks)
    assert sum(v["tokens"] for v in m.per_agent_load.values()) == m.work_tokens
    assert sum(v["tasks"] for v in m.per_agent_load.values()) >= m.n_subtasks


def test_dependency_order_in_trace(tmp_path):
    result = run(family="long_horizon", strategy="orchestrated", team=4,
                 seed=2, trace_dir=str(tmp_path))
    events = [json.loads(l) for l in open(result.trace_path)]
    done = [e["subtask"] for e in events if e["event"] == "subtask_done"]
    assert done == sorted(done)  # chain ids sort in dependency order


def test_orchestrated_never_duplicates():
    for family in ("research_sweep", "long_horizon"):
        for seed in range(3):
            m = run(family=family, strategy="orchestrated", team=8, seed=seed).metrics
            assert m.duplicate_attempts == 0
            assert m.duplication_rate == 0.0


def test_independent_can_duplicate():
    dups = [run(strategy="independent", team=8, seed=s).metrics.duplicate_attempts
            for s in range(4)]
    assert sum(dups) > 0


def test_solo_uses_one_agent():
    m = run(strategy="solo", team=16).metrics
    assert m.team_size == 1
    assert len(m.per_agent_load) == 1


def test_verified_retries_reduce_errors():
    orch, ver = [], []
    for seed in range(5):
        orch.append(run(strategy="orchestrated", team=4, seed=seed).metrics.fraction_correct)
        ver.append(run(strategy="verified", team=4, seed=seed).metrics.fraction_correct)
    assert sum(ver) / len(ver) > sum(orch) / len(orch)


def test_code_family_runs_end_to_end(tmp_path):
    result = run_experiment(RunConfig(
        family="codebase_repair", strategy="verified", team_size=4,
        budget=BIG, seed=1, sandbox_root=str(tmp_path)))
    m = result.metrics
    assert m.n_subtasks == 8
    assert m.fraction_correct >= 0.75  # verification + retry should land most fixes
    # every subtask's file exists in the run sandbox with the final payload
    from pathlib import Path
    run_dirs = list(Path(tmp_path).glob("codebase_repair-*"))
    assert run_dirs
    files = {p.name for p in run_dirs[0].glob("mod_*.py")}
    assert len(files) == 8


def test_trace_summary_renders(tmp_path):
    result = run(strategy="verified", team=4, seed=0, trace_dir=str(tmp_path))
    from teamscale.trace_summary import summarize_trace
    text = summarize_trace(result.trace_path)
    assert "busiest agent" in text
    assert "timeline:" in text
