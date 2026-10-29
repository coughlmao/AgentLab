"""Hard-mode real tasks: generation, proofs, and difficulty plumbing.

Everything runs offline on the fake backend. No network.
"""
import pytest

from teamscale.fake_backend import FakeLLMBackend
import teamscale.real_pilot as rp
from teamscale.real_tasks import (check_all, make_real_task, materialize,
                                  run_check)


def _apply_fixed(sandbox, task):
    for m in task.modules:
        (sandbox / m.module_file).write_text(m.fixed_code)
        for fname, (_broken, fixed) in m.extra_files.items():
            (sandbox / fname).write_text(fixed)


def _factory(competence=1.0):
    return lambda cfg: FakeLLMBackend(
        competence=competence, seed=cfg["seed"] * 100 + cfg["team_size"])


# ------------------------------------------------------------- generation
def test_hard_generation_is_deterministic():
    a = make_real_task(seed=3, difficulty="hard")
    b = make_real_task(seed=3, difficulty="hard")
    key = lambda t: [(m.module_file, m.defect_type, m.broken_code,
                      m.fixed_code, m.check_code, m.extra_files)
                     for m in t.modules]
    assert key(a) == key(b)
    assert a.difficulty == "hard" and b.difficulty == "hard"


def test_hard_varies_with_seed_and_differs_from_standard():
    hard0 = make_real_task(seed=0, difficulty="hard")
    hard1 = make_real_task(seed=1, difficulty="hard")
    std0 = make_real_task(seed=0)
    assert [m.broken_code for m in hard0.modules] != \
        [m.broken_code for m in hard1.modules]
    assert {m.defect_type for m in hard0.modules}.isdisjoint(
        {m.defect_type for m in std0.modules})
    assert std0.difficulty == "standard"


def test_unknown_difficulty_rejected():
    with pytest.raises(ValueError, match="unknown difficulty"):
        make_real_task(seed=0, difficulty="nightmare")


def test_hard_composition_is_guaranteed():
    for seed in (0, 1, 2, 3):
        task = make_real_task(seed=seed, difficulty="hard")
        assert len(task.modules) == 8
        # at least one cross-module subtask (defect lives in a helper)
        assert any(m.extra_files for m in task.modules)
        # exactly one multi-defect subtask
        assert sum(1 for m in task.modules if m.n_defects >= 2) == 1
        # cross-module repairs target the helper, not the caller
        for m in task.modules:
            if m.extra_files:
                assert m.repair_file in m.extra_files
                assert m.broken_code == m.fixed_code  # caller is correct


# ------------------------------------------------------------------ proofs
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_hard_tasks_fail_before_and_pass_after(tmp_path, seed):
    task = make_real_task(seed=seed, difficulty="hard")
    sandbox = materialize(task, tmp_path / f"h{seed}")
    before = check_all(task, sandbox)
    assert not any(before.values()), f"seed {seed}: a check passed pre-fix"
    _apply_fixed(sandbox, task)
    after = check_all(task, sandbox)
    assert all(after.values()), f"seed {seed}: a check still fails post-fix"


def test_hard_repair_pairs_reproduce_fixed_code():
    for seed in (0, 1, 2):
        task = make_real_task(seed=seed, difficulty="hard")
        for m in task.modules:
            target = m.repair_file or m.module_file
            if target == m.module_file:
                t_broken, t_fixed = m.broken_code, m.fixed_code
            else:
                t_broken, t_fixed = m.extra_files[target]
            find, repl = m.repair
            assert find and t_broken.replace(find, repl, 1) == t_fixed


def test_multi_defect_partial_fixes_still_fail(tmp_path):
    task = make_real_task(seed=0, difficulty="hard")
    multi = next(m for m in task.modules if m.n_defects >= 2)
    assert len(multi.partial_fixes) == multi.n_defects
    sandbox = materialize(task, tmp_path)
    passed, _ = run_check(sandbox, multi)
    assert not passed
    for fname, code in multi.partial_fixes:
        (sandbox / fname).write_text(code)
        passed, _ = run_check(sandbox, multi)
        assert not passed, "a partial fix passed: defects are not interacting"
        (sandbox / fname).write_text(multi.broken_code)
    (sandbox / multi.module_file).write_text(multi.fixed_code)
    passed, _ = run_check(sandbox, multi)
    assert passed


# -------------------------------------------------------------- fake backend
def test_fake_backend_repairs_cross_module_defect(tmp_path):
    task = make_real_task(seed=0, difficulty="hard")
    cross = next(m for m in task.modules if m.extra_files)
    sandbox = materialize(task, tmp_path)
    backend = FakeLLMBackend(competence=1.0, seed=0)
    fix = backend.fix_module(cross, sandbox)
    assert fix.declared_done
    passed, _ = run_check(sandbox, cross)
    assert passed  # the repair landed in the helper file


def test_fake_backend_fixes_hard_run_end_to_end(tmp_path):
    result = rp.run_single_run(strategy="verified", team_size=2, seed=0,
                               backend_factory=_factory(), out_dir=tmp_path,
                               model="fake", difficulty="hard")
    assert result["difficulty"] == "hard"
    assert result["initial_all_broken"]
    assert result["success"] and result["fraction_correct"] == 1.0


# ---------------------------------------------------------------- plumbing
def test_run_pilot_summary_records_difficulty(tmp_path):
    summary = rp.run_pilot(out_dir=tmp_path / "hard", backend_factory=_factory(),
                           model="fake", sizes=(1,), strategies=("orchestrated",),
                           seeds=(0,), difficulty="hard",
                           ledger_name="ledger_fake.json")
    assert summary["difficulty"] == "hard"
    assert summary["runs_completed"] == 1
    summary_std = rp.run_pilot(out_dir=tmp_path / "std",
                               backend_factory=_factory(), model="fake",
                               sizes=(1,), strategies=("orchestrated",),
                               seeds=(0,), ledger_name="ledger_fake.json")
    assert summary_std["difficulty"] == "standard"
