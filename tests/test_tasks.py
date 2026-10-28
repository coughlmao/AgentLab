import pytest

from teamscale.tasks import (SandboxViolation, ensure_within, lenient_score,
                             make_task, strict_verify, write_in_sandbox)


def test_families_have_valid_dags():
    for family in ("research_sweep", "codebase_repair", "long_horizon"):
        task = make_task(family, seed=3)
        ids = {s.subtask_id for s in task.subtasks}
        assert task.subtasks
        for s in task.subtasks:
            assert all(d in ids for d in s.deps)
            assert 0.0 <= s.difficulty <= 1.0


def test_long_horizon_is_a_chain():
    task = make_task("long_horizon", seed=0)
    for i, s in enumerate(task.subtasks):
        expected = (task.subtasks[i - 1].subtask_id,) if i else ()
        assert s.deps == expected
    assert task.depth_of(task.subtasks[-1]) == len(task.subtasks) - 1


def test_sandbox_containment(tmp_path):
    write_in_sandbox(tmp_path, "ok/inner.py", "x = 1")
    assert (tmp_path / "ok" / "inner.py").exists()
    with pytest.raises(SandboxViolation):
        ensure_within(tmp_path, "../escape.py")
    with pytest.raises(SandboxViolation):
        write_in_sandbox(tmp_path, "sub/../../escape.py", "x = 2")


def test_code_verifier_really_executes(tmp_path):
    task = make_task("codebase_repair", seed=1)
    sub = task.subtasks[0]
    assert strict_verify(sub, sub.fixed_code, tmp_path) is True
    assert strict_verify(sub, sub.broken_code, tmp_path) is False


def test_lenient_gives_partial_credit_for_topic_only():
    task = make_task("research_sweep", seed=1)
    sub = task.subtasks[0]
    guess = f"{sub.topic}:0000" if not sub.answer.endswith(":0000") else f"{sub.topic}:1111"
    assert lenient_score(sub, guess, None) == 0.5
    assert lenient_score(sub, sub.answer, None) == 1.0
    assert lenient_score(sub, "nope:xxxx", None) == 0.0
