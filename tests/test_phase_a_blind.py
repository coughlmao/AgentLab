"""Phase A blind mode: run_check removed from the agents' own tool set
in both real backends and the fake; the runner's external verification
(the verified strategy) still works. Everything offline. No network.
"""
import json

from teamscale.anthropic_backend import AnthropicLLMBackend
from teamscale.fake_backend import FakeLLMBackend
from teamscale.llm_backend import OpenAILLMBackend
import teamscale.real_pilot as rp
from teamscale.real_tasks import make_real_task, materialize, run_check


def _blind_factory(competence=1.0):
    return lambda cfg: FakeLLMBackend(
        competence=competence, seed=cfg["seed"] * 100 + cfg["team_size"],
        blind=cfg.get("blind", False))


# ------------------------------------------------------------ OpenAI backend
class _OpenAICapture:
    """Returns an immediate text reply; records every request payload."""

    def __init__(self):
        self.requests = []

    def __call__(self, url, headers, payload):
        self.requests.append(payload)
        return {"choices": [{"message": {"role": "assistant",
                                         "content": "done"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def _openai_tools(payload):
    return [t["function"]["name"] for t in payload["tools"]]


def test_openai_blind_removes_run_check(tmp_path):
    task = make_real_task(seed=0)
    sandbox = materialize(task, tmp_path)
    cap = _OpenAICapture()
    backend = OpenAILLMBackend(api_key="test-key", transport=cap, blind=True)
    fix = backend.fix_module(task.modules[0], sandbox)
    assert fix.declared_done
    payload = cap.requests[0]
    assert "run_check" not in _openai_tools(payload)
    assert set(_openai_tools(payload)) == {"list_files", "read_file",
                                           "write_file"}
    system = payload["messages"][0]["content"]
    assert "checked externally after you submit" in system
    user = payload["messages"][1]["content"]
    assert "You cannot run the check yourself" in user


def test_openai_sighted_still_offers_run_check(tmp_path):
    task = make_real_task(seed=0)
    sandbox = materialize(task, tmp_path)
    cap = _OpenAICapture()
    backend = OpenAILLMBackend(api_key="test-key", transport=cap)
    backend.fix_module(task.modules[0], sandbox)
    assert "run_check" in _openai_tools(cap.requests[0])
    assert "run the check to confirm" in cap.requests[0]["messages"][0]["content"]


def test_openai_blind_run_check_request_is_refused(tmp_path):
    task = make_real_task(seed=0)
    sandbox = materialize(task, tmp_path)
    seen = {}

    def transport(url, headers, payload):
        messages = payload["messages"]
        step = sum(1 for m in messages if m.get("role") == "assistant")
        if step == 0:
            return {"choices": [{"message": {
                        "role": "assistant", "content": None,
                        "tool_calls": [{"id": "c1", "type": "function",
                                        "function": {
                                            "name": "run_check",
                                            "arguments": json.dumps(
                                                {"check_file": "x.py"})}}]}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        seen["tool_reply"] = messages[-1]["content"]
        return {"choices": [{"message": {"role": "assistant",
                                         "content": "ok"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    backend = OpenAILLMBackend(api_key="test-key", transport=transport,
                               blind=True)
    backend.fix_module(task.modules[0], sandbox)
    assert "unavailable tool: run_check" in seen["tool_reply"]


# --------------------------------------------------------- Anthropic backend
class _AnthropicCapture:
    def __init__(self):
        self.requests = []

    def __call__(self, url, headers, payload):
        self.requests.append(payload)
        return {"content": [{"type": "text", "text": "done"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5}}


def test_anthropic_blind_removes_run_check(tmp_path):
    task = make_real_task(seed=0)
    sandbox = materialize(task, tmp_path)
    cap = _AnthropicCapture()
    backend = AnthropicLLMBackend(api_key="test-key", transport=cap,
                                  blind=True)
    fix = backend.fix_module(task.modules[0], sandbox)
    assert fix.declared_done
    payload = cap.requests[0]
    assert {t["name"] for t in payload["tools"]} == {
        "list_files", "read_file", "write_file"}
    assert "checked externally after you submit" in payload["system"]
    assert "You cannot run the check yourself" in \
        payload["messages"][0]["content"]


def test_anthropic_sighted_still_offers_run_check(tmp_path):
    task = make_real_task(seed=0)
    sandbox = materialize(task, tmp_path)
    cap = _AnthropicCapture()
    backend = AnthropicLLMBackend(api_key="test-key", transport=cap)
    backend.fix_module(task.modules[0], sandbox)
    assert "run_check" in {t["name"] for t in cap.requests[0]["tools"]}


def test_anthropic_blind_run_check_request_is_refused(tmp_path):
    task = make_real_task(seed=0)
    sandbox = materialize(task, tmp_path)
    seen = {}

    def transport(url, headers, payload):
        messages = payload["messages"]
        step = sum(1 for m in messages if m["role"] == "assistant")
        usage = {"input_tokens": 10, "output_tokens": 5}
        if step == 0:
            return {"content": [{"type": "tool_use", "id": "t1",
                                 "name": "run_check",
                                 "input": {"check_file": "x.py"}}],
                    "stop_reason": "tool_use", "usage": usage}
        seen["tool_result"] = messages[-1]["content"][0]
        return {"content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn", "usage": usage}

    backend = AnthropicLLMBackend(api_key="test-key", transport=transport,
                                  blind=True)
    backend.fix_module(task.modules[0], sandbox)
    assert seen["tool_result"]["is_error"] is True
    assert "unavailable tool: run_check" in seen["tool_result"]["content"]


# -------------------------------------------------------------- fake backend
def test_fake_blind_never_runs_check_but_still_fixes(tmp_path):
    task = make_real_task(seed=0)
    sandbox = materialize(task, tmp_path)
    backend = FakeLLMBackend(competence=1.0, seed=0, blind=True)
    fix = backend.fix_module(task.modules[0], sandbox)
    assert fix.declared_done
    tools = [e["tool"] for e in fix.events if e["event"] == "tool_call"]
    assert "run_check" not in tools
    assert tools == ["read_file", "write_file"]
    passed, _ = run_check(sandbox, task.modules[0])  # the runner can check
    assert passed


def test_fake_sighted_does_run_check(tmp_path):
    task = make_real_task(seed=0)
    sandbox = materialize(task, tmp_path)
    backend = FakeLLMBackend(competence=1.0, seed=0)
    fix = backend.fix_module(task.modules[0], sandbox)
    tools = [e["tool"] for e in fix.events if e["event"] == "tool_call"]
    assert "run_check" in tools


# ------------------------------------------------------- runner integration
def test_blind_verified_still_retries_via_external_check(tmp_path):
    result = rp.run_single_run(strategy="verified", team_size=2, seed=1,
                               backend_factory=_blind_factory(competence=0.0),
                               out_dir=tmp_path, model="fake", blind=True)
    assert result["blind"] is True
    assert result["retries"] >= 1  # external check drove the retries
    assert result["fraction_correct"] < 1.0
    trace = [json.loads(l) for l in
             (tmp_path / "traces" / f"{result['run_id']}.jsonl")
             .read_text().splitlines()]
    assert trace[0]["blind"] is True
    # the agents themselves never executed a check anywhere in the trace
    assert not any(e.get("tool") == "run_check" for e in trace
                   if e["event"] == "tool_call")
    # but the runner's verification events are there
    assert any(e["event"] == "verify" for e in trace)


def test_blind_defaults_off_and_is_recorded(tmp_path):
    sighted = rp.run_single_run(strategy="independent", team_size=1, seed=0,
                                backend_factory=_blind_factory(),
                                out_dir=tmp_path / "a", model="fake")
    assert sighted["blind"] is False
    summary = rp.run_pilot(out_dir=tmp_path / "b",
                           backend_factory=_blind_factory(), model="fake",
                           sizes=(1,), strategies=("verified",), seeds=(0,),
                           workers=1, blind=True)
    assert summary["blind"] is True
    assert summary["results"][0]  # the run completed
