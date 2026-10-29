"""Anthropic-backend tests: scripted mock transport only. No network."""
import json
from pathlib import Path

import pytest

from teamscale.anthropic_backend import (AnthropicLLMBackend,
                                         resolve_anthropic_api_key)
from teamscale.llm_backend import BackendError
import teamscale.real_pilot as rp
from teamscale.real_tasks import make_real_task, materialize, run_check

USAGE = [(100, 10), (200, 50), (150, 20), (120, 15)]  # per scripted step


def _field(text: str, label: str) -> str:
    for line in text.splitlines():
        if line.startswith(label):
            return line.split(":", 1)[1].strip()
    raise AssertionError(f"label not found: {label}")


class ScriptedAnthropicTransport:
    """Offline stand-in for the Messages API.

    Plays a competent agent per subtask conversation: read the module,
    write the repair, run the check, then stop with a text reply. Usage
    per call is fixed (USAGE), so accounting can be asserted exactly.
    """

    def __init__(self, search_root, repairs):
        self.search_root = Path(search_root)
        self.repairs = repairs  # {module_file: (find, repl)}
        self.requests = []

    def _current_content(self, module_file: str) -> str:
        for p in self.search_root.rglob(module_file):
            return p.read_text()
        raise AssertionError(f"module not found under {self.search_root}")

    def __call__(self, url, headers, payload):
        self.requests.append({"url": url, "headers": headers,
                              "payload": payload})
        messages = payload["messages"]
        first_user = messages[0]["content"]
        module_file = _field(first_user, "Target module:")
        check_file = _field(first_user, "Check script:")
        step = sum(1 for m in messages if m["role"] == "assistant")
        tin, tout = USAGE[min(step, len(USAGE) - 1)]
        usage = {"input_tokens": tin, "output_tokens": tout}
        tid = f"toolu_{step + 1}"
        if step == 0:
            block = {"type": "tool_use", "id": tid, "name": "read_file",
                     "input": {"path": module_file}}
        elif step == 1:
            find, repl = self.repairs[module_file]
            fixed = self._current_content(module_file).replace(find, repl, 1)
            block = {"type": "tool_use", "id": tid, "name": "write_file",
                     "input": {"path": module_file, "content": fixed}}
        elif step == 2:
            block = {"type": "tool_use", "id": tid, "name": "run_check",
                     "input": {"check_file": check_file}}
        else:
            return {"content": [{"type": "text", "text": "check passes"}],
                    "stop_reason": "end_turn", "usage": usage}
        return {"content": [block], "stop_reason": "tool_use", "usage": usage}


def _setup(tmp_path, seed=0):
    task = make_real_task(seed=seed)
    sandbox = materialize(task, tmp_path / "sandbox")
    repairs = {m.module_file: m.repair for m in task.modules}
    transport = ScriptedAnthropicTransport(sandbox, repairs)
    backend = AnthropicLLMBackend(model="claude-haiku-4-5",
                                  api_key="test-key", transport=transport)
    return task, sandbox, transport, backend


# ---------------------------------------------------------------- the loop
def test_anthropic_loop_completes_fix(tmp_path):
    task, sandbox, transport, backend = _setup(tmp_path)
    module = task.modules[0]
    fix = backend.fix_module(module, sandbox)
    assert fix.declared_done and fix.error is None
    assert fix.model_calls == 4
    passed, _ = run_check(sandbox, module)
    assert passed
    assert (sandbox / module.module_file).read_text() == module.fixed_code


def test_anthropic_usage_accounting_sums(tmp_path):
    task, sandbox, transport, backend = _setup(tmp_path)
    fix = backend.fix_module(task.modules[0], sandbox)
    assert fix.tokens_in == sum(u[0] for u in USAGE)
    assert fix.tokens_out == sum(u[1] for u in USAGE)
    ev_in = sum(e["tokens_in"] for e in fix.events if e["event"] == "model_call")
    ev_out = sum(e["tokens_out"] for e in fix.events
                 if e["event"] == "model_call")
    assert (ev_in, ev_out) == (fix.tokens_in, fix.tokens_out)
    tools = [e["tool"] for e in fix.events if e["event"] == "tool_call"]
    assert tools == ["read_file", "write_file", "run_check"]


# ----------------------------------------------------------- request shape
def test_anthropic_request_shape(tmp_path):
    task, sandbox, transport, backend = _setup(tmp_path)
    backend.fix_module(task.modules[0], sandbox)
    url, headers, payload = (transport.requests[0][k]
                             for k in ("url", "headers", "payload"))
    assert url == "https://api.anthropic.com/v1/messages"
    assert headers["x-api-key"] == "test-key"
    assert headers["anthropic-version"] == "2023-06-01"
    assert payload["model"] == "claude-haiku-4-5"
    assert "fixing one broken Python module" in payload["system"]
    assert payload["messages"][0]["role"] == "user"
    tool0 = payload["tools"][0]
    assert set(tool0) == {"name", "description", "input_schema"}
    assert {t["name"] for t in payload["tools"]} == {
        "list_files", "read_file", "write_file", "run_check"}
    # second request carries the assistant tool_use block and a matching
    # user tool_result block
    msgs = transport.requests[1]["payload"]["messages"]
    assistant = [m for m in msgs if m["role"] == "assistant"][0]
    use = assistant["content"][0]
    assert use["type"] == "tool_use" and use["name"] == "read_file"
    follow = msgs[msgs.index(assistant) + 1]
    assert follow["role"] == "user"
    res = follow["content"][0]
    assert res["type"] == "tool_result"
    assert res["tool_use_id"] == use["id"]
    assert res["is_error"] is False


def test_anthropic_tool_error_marks_is_error(tmp_path):
    def transport(url, headers, payload):
        msgs = payload["messages"]
        step = sum(1 for m in msgs if m["role"] == "assistant")
        usage = {"input_tokens": 10, "output_tokens": 5}
        if step == 0:
            return {"content": [{"type": "tool_use", "id": "toolu_1",
                                 "name": "read_file",
                                 "input": {"path": "../outside.py"}}],
                    "stop_reason": "tool_use", "usage": usage}
        # capture the tool_result the backend sent back, then finish
        transport.last = msgs[-1]
        return {"content": [{"type": "text", "text": "cannot proceed"}],
                "stop_reason": "end_turn", "usage": usage}

    task = make_real_task(seed=0)
    sandbox = materialize(task, tmp_path)
    backend = AnthropicLLMBackend(api_key="test-key", transport=transport)
    fix = backend.fix_module(task.modules[0], sandbox)
    assert fix.declared_done
    res = transport.last["content"][0]
    assert res["type"] == "tool_result" and res["is_error"] is True
    assert res["content"].startswith("error:")


# ---------------------------------------------------------------- budget
def test_anthropic_budget_cap_stops_calls(tmp_path):
    def greedy(url, headers, payload):
        return {"content": [{"type": "tool_use", "id": "toolu_x",
                             "name": "read_file",
                             "input": {"path": "whatever.py"}}],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 40_000, "output_tokens": 5_000}}

    task = make_real_task(seed=0)
    sandbox = materialize(task, tmp_path)
    backend = AnthropicLLMBackend(api_key="test-key", transport=greedy)
    fix = backend.fix_module(task.modules[0], sandbox, token_budget=75_000)
    assert fix.budget_exceeded and not fix.declared_done
    assert fix.model_calls == 2  # 45K + 45K; the third call never happens
    assert fix.tokens == 90_000


def test_anthropic_max_tokens_stop_is_an_error(tmp_path):
    def truncated(url, headers, payload):
        return {"content": [{"type": "text", "text": "partial..."}],
                "stop_reason": "max_tokens",
                "usage": {"input_tokens": 10, "output_tokens": 4096}}

    task = make_real_task(seed=0)
    sandbox = materialize(task, tmp_path)
    backend = AnthropicLLMBackend(api_key="test-key", transport=truncated)
    fix = backend.fix_module(task.modules[0], sandbox)
    assert not fix.declared_done and "max_tokens" in fix.error


# ----------------------------------------------------------------- errors
def test_anthropic_api_error_surfaces_cleanly(tmp_path):
    def failing(url, headers, payload):
        raise BackendError("API returned HTTP 401: invalid x-api-key")

    task = make_real_task(seed=0)
    sandbox = materialize(task, tmp_path)
    backend = AnthropicLLMBackend(api_key="test-key", transport=failing)
    fix = backend.fix_module(task.modules[0], sandbox)
    assert fix.error and "401" in fix.error
    assert fix.tokens == 0 and fix.model_calls == 0


def test_anthropic_key_resolution(monkeypatch):
    monkeypatch.delenv("TEAMSCLAE_ANTHROPIC_CRED_CMD", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(BackendError, match="no API key configured"):
        resolve_anthropic_api_key()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-env-key")
    assert resolve_anthropic_api_key() == "dummy-env-key"
    monkeypatch.setenv("TEAMSCLAE_ANTHROPIC_CRED_CMD", "printf cred-cmd-key")
    assert resolve_anthropic_api_key() == "cred-cmd-key"


# ------------------------------------------------------- pricing and caps
def test_claude_pricing_math():
    assert rp.worst_case_run_usd("claude-haiku-4-5", 400_000) \
        == pytest.approx(0.72)
    assert rp.worst_case_run_usd("claude-haiku-5-5", 400_000) \
        == pytest.approx(0.072)
    assert rp.cost_usd("claude-haiku-4-5", 1_000_000, 1_000_000) \
        == pytest.approx(6.00)
    assert rp.cost_usd("claude-haiku-5-5", 1_000_000, 1_000_000) \
        == pytest.approx(0.60)


def test_claude_spend_cap_gate_uses_model_price(tmp_path):
    ledger = rp.SpendLedger(tmp_path / "ledger.json", cap_usd=15.0,
                            model="claude-haiku-4-5")
    assert ledger.can_afford()
    ledger.spent_usd = 14.50
    assert not ledger.can_afford()  # 14.50 + 0.72 worst case > 15.00
    ledger.spent_usd = 14.00
    assert ledger.can_afford()      # 14.00 + 0.72 <= 15.00


# -------------------------------------------------------------- end to end
def test_anthropic_real_path_run_end_to_end(tmp_path):
    task = make_real_task(seed=0)
    repairs = {m.module_file: m.repair for m in task.modules}
    transport = ScriptedAnthropicTransport(tmp_path, repairs)
    factory = lambda cfg: AnthropicLLMBackend(
        model="claude-haiku-4-5", api_key="test-key", transport=transport)
    result = rp.run_single_run(strategy="orchestrated", team_size=2, seed=0,
                               backend_factory=factory, out_dir=tmp_path,
                               model="claude-haiku-4-5")
    assert result["backend"] == "anthropic"
    assert result["run_id"].startswith("anthropic-")
    assert result["initial_all_broken"]
    assert result["success"] and result["fraction_correct"] == 1.0
    assert result["tokens_in"] == 8 * sum(u[0] for u in USAGE)
    assert result["tokens_out"] == 8 * sum(u[1] for u in USAGE)
    per_module = 570 / 1e6 * 1.00 + 95 / 1e6 * 5.00
    assert result["cost_usd"] == pytest.approx(8 * per_module, abs=1e-6)
    trace = [json.loads(l) for l in
             (tmp_path / "traces" / f"{result['run_id']}.jsonl")
             .read_text().splitlines()]
    t_in = sum(e.get("tokens_in", 0) for e in trace if e["event"] == "model_call")
    assert t_in == result["tokens_in"]
    assert not any("test-key" in json.dumps(e) for e in trace)


#--------------------------------------------------------------------
def test_anthropic_credential_command_prefers_new_prefix(monkeypatch):
    from teamscale.anthropic_backend import resolve_anthropic_api_key

    monkeypatch.setenv("TEAMSCALE_ANTHROPIC_CRED_CMD", "printf new-key")
    monkeypatch.setenv("TEAMSCLAE_ANTHROPIC_CRED_CMD", "printf legacy-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert resolve_anthropic_api_key() == "new-key"


def test_anthropic_model_prefers_new_prefix(monkeypatch):
    from teamscale.anthropic_backend import AnthropicLLMBackend

    monkeypatch.setenv("TEAMSCALE_MODEL", "new-model")
    monkeypatch.setenv("TEAMSCLAE_MODEL", "legacy-model")

    backend = AnthropicLLMBackend(api_key="test-key")
    assert backend.model == "new-model"


def test_anthropic_credential_command_falls_back_to_legacy(monkeypatch):
    from teamscale.anthropic_backend import resolve_anthropic_api_key

    monkeypatch.delenv("TEAMSCALE_ANTHROPIC_CRED_CMD", raising=False)
    monkeypatch.setenv("TEAMSCLAE_ANTHROPIC_CRED_CMD", "printf legacy-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert resolve_anthropic_api_key() == "legacy-key"
