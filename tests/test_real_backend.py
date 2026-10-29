"""Real-path tests: everything runs on the fake backend. No network."""
import json

import pytest

from teamscale.fake_backend import FakeLLMBackend
from teamscale.llm_backend import SandboxTools, resolve_api_key, BackendError
import teamscale.real_pilot as rp
from teamscale.real_tasks import check_all, make_real_task, materialize, run_check
from teamscale.tasks import SandboxViolation


def _factory(competence=1.0):
    return lambda cfg: FakeLLMBackend(competence=competence,
                                      seed=cfg["seed"] * 100 + cfg["team_size"])


# ------------------------------------------------------------- task generation
def test_generated_tasks_fail_before_and_pass_after(tmp_path):
    for seed in (0, 1, 2, 3):
        task = make_real_task(seed=seed)
        assert len(task.modules) == 8
        sandbox = materialize(task, tmp_path / f"s{seed}")
        before = check_all(task, sandbox)
        assert not any(before.values()), f"seed {seed}: a check passed pre-fix"
        for m in task.modules:  # apply the ground-truth fix by hand
            (sandbox / m.module_file).write_text(m.fixed_code)
        after = check_all(task, sandbox)
        assert all(after.values()), f"seed {seed}: a check still fails post-fix"


def test_task_instances_vary_with_seed():
    a = make_real_task(seed=0)
    b = make_real_task(seed=1)
    assert [m.module_file for m in a.modules] != [m.module_file for m in b.modules] \
        or [m.broken_code for m in a.modules] != [m.broken_code for m in b.modules]


# ---------------------------------------------------------------- containment
def test_real_write_tool_containment(tmp_path):
    tools = SandboxTools(tmp_path)
    with pytest.raises(SandboxViolation):
        tools.write_file("../evil.py", "x = 1\n")
    with pytest.raises(SandboxViolation):
        tools.read_file("../secret.py")
    with pytest.raises(SandboxViolation):
        tools.write_file("/tmp/evil.py", "x = 1\n")
    assert "wrote" in tools.write_file("ok.py", "x = 1\n")


# ---------------------------------------------------------------- accounting
def test_fake_backend_token_accounting_sums(tmp_path):
    task = make_real_task(seed=0)
    sandbox = materialize(task, tmp_path)
    backend = FakeLLMBackend(competence=1.0, seed=0)
    fix = backend.fix_module(task.modules[0], sandbox)
    ev_in = sum(e["tokens_in"] for e in fix.events if e["event"] == "model_call")
    ev_out = sum(e["tokens_out"] for e in fix.events if e["event"] == "model_call")
    assert (ev_in, ev_out) == (fix.tokens_in, fix.tokens_out)
    assert fix.tokens > 0 and fix.declared_done
    passed, _ = run_check(sandbox, task.modules[0])
    assert passed  # competence 1.0 applies the correct repair


def test_fake_backend_budget_enforced(tmp_path):
    task = make_real_task(seed=0)
    sandbox = materialize(task, tmp_path)
    backend = FakeLLMBackend(competence=1.0, seed=0)
    fix = backend.fix_module(task.modules[0], sandbox, token_budget=100)
    assert fix.budget_exceeded or fix.tokens <= 100 + 2000  # stops early
    fix2 = backend.fix_module(task.modules[1], sandbox, token_budget=1200)
    assert fix2.budget_exceeded and not fix2.declared_done


# ------------------------------------------------------------------ end to end
@pytest.mark.parametrize("team_size", [1, 2])
def test_fake_real_path_run_end_to_end(tmp_path, team_size):
    result = rp.run_single_run(strategy="orchestrated", team_size=team_size,
                               seed=0, backend_factory=_factory(),
                               out_dir=tmp_path, model="fake")
    assert result["initial_all_broken"]
    assert result["n_subtasks"] == 8
    assert result["success"] and result["fraction_correct"] == 1.0
    assert result["total_tokens"] == result["tokens_in"] + result["tokens_out"]
    assert result["cost_usd"] == 0.0
    # trace accounting reconciles with the result totals
    trace = [json.loads(l) for l in
             (tmp_path / "traces" / f"{result['run_id']}.jsonl").read_text().splitlines()]
    t_in = sum(e.get("tokens_in", 0) for e in trace if e["event"] == "model_call")
    t_out = sum(e.get("tokens_out", 0) for e in trace if e["event"] == "model_call")
    assert (t_in, t_out) == (result["tokens_in"], result["tokens_out"])
    assert not any("prompt" in json.dumps(e).lower() for e in trace)


def test_verified_strategy_retries_and_reports(tmp_path):
    # competence 0: first attempts all fail, retry (with feedback boost in the
    # fake) is exercised and counted either way
    result = rp.run_single_run(strategy="verified", team_size=2, seed=1,
                               backend_factory=_factory(competence=0.0),
                               out_dir=tmp_path, model="fake")
    assert result["retries"] >= 1
    assert result["fraction_correct"] < 1.0


# ------------------------------------------------------------------ spend cap
def test_worst_case_and_cost_math():
    assert rp.worst_case_run_usd("gpt-5-mini", 400_000) == pytest.approx(0.24)
    assert rp.cost_usd("gpt-5-mini", 1_000_000, 1_000_000) == pytest.approx(2.25)


def test_spend_cap_gate_triggers(tmp_path):
    ledger = rp.SpendLedger(tmp_path / "ledger.json", cap_usd=15.0,
                            model="gpt-5-mini")
    assert ledger.can_afford("gpt-5-mini")
    ledger.spent_usd = 14.90
    assert not ledger.can_afford("gpt-5-mini")  # 14.90 + 0.24 worst > 15.00
    ledger.record({"run_id": "x", "cost_usd": 0.05, "tokens_in": 10,
                   "tokens_out": 5})
    saved = json.loads((tmp_path / "ledger.json").read_text())
    assert saved["spent_usd"] == pytest.approx(14.95)
    assert saved["runs"][0]["run_id"] == "x"


def test_no_key_configured_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("TEAMSCLAE_CRED_CMD", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(BackendError, match="no API key configured"):
        resolve_api_key()
