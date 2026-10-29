"""An Anthropic (Claude) counterpart of the OpenAI real backend.

Same work, second provider: a model reads a broken module inside a sandbox
directory, edits it with tools, and runs the module's check script until it
passes or the attempt budget runs out. This module implements the Messages
API variant of `teamscale.llm_backend.OpenAILLMBackend` so the pilot runner
(teamscale.real_pilot) can drive either provider through the same
`fix_module` interface and produce identically shaped results and traces.

Design constraints mirror llm_backend.py exactly:

- Standard library only (urllib). No SDK dependency.
- The API key is NEVER read from a file, logged, printed, or written into
  traces or results. It comes from exactly two places:
    1. the shell command in env TEAMSCLAE_ANTHROPIC_CRED_CMD (its stdout
       is the key), or
    2. env ANTHROPIC_API_KEY, if already present in the process environment.
  If neither is configured, construction raises a clear error.
- Model: env TEAMSCLAE_MODEL, default "claude-haiku-4-5".
- Every tool call is sandbox-contained (teamscale.tasks.ensure_within),
  via the shared SandboxTools surface from llm_backend.
- Token usage is taken from the API responses themselves (usage
  input/output tokens per call), summed exactly. The same per-subtask
  token budget is enforced: once usage reaches it, the loop stops and the
  attempt is reported with usage so far.

Messages API shape notes: the system prompt goes in the top-level `system`
field; tools use {name, description, input_schema}; assistant tool calls
come back as `tool_use` content blocks and tool outputs go back as a user
message of `tool_result` blocks (with is_error set on failures).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .llm_backend import (BLIND_SYSTEM_PROMPT, MAX_MODEL_CALLS, SYSTEM_PROMPT,
                           TOOL_SCHEMAS, BackendError, FixResult, SandboxTools,
                           _default_transport, build_user_prompt,
                           tool_schemas)
from .real_tasks import RealModule
from .tasks import SandboxViolation

API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-haiku-4-5"
MAX_TOKENS = 4096  # per response; the Messages API requires a cap


def resolve_anthropic_api_key() -> str:
    """Return the API key from the configured credential source.

    The key is returned to the caller (the backend constructor) and nowhere
    else; it is never logged or persisted by this project.
    """
    cmd = os.environ.get("TEAMSCLAE_ANTHROPIC_CRED_CMD", "").strip()
    if cmd:
        proc = subprocess.run(cmd, shell=True, capture_output=True,
                              text=True, timeout=30)
        key = proc.stdout.strip()
        if proc.returncode != 0 or not key:
            raise BackendError(
                "TEAMSCLAE_ANTHROPIC_CRED_CMD ran but produced no key on "
                f"stdout (exit {proc.returncode}); check the credential "
                "command.")
        return key
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    raise BackendError(
        "no API key configured: set TEAMSCLAE_ANTHROPIC_CRED_CMD to a "
        "command whose stdout is the key, or export ANTHROPIC_API_KEY.")


def _to_anthropic(schemas) -> list:
    return [{"name": spec["function"]["name"],
             "description": spec["function"]["description"],
             "input_schema": spec["function"]["parameters"]}
            for spec in schemas]


# The OpenAI tool schemas converted to the Messages API tool format: same
# names, same descriptions, same parameter schemas. The blind variant
# drops run_check, exactly as on the OpenAI side.
ANTHROPIC_TOOL_SCHEMAS = _to_anthropic(tool_schemas(False))
ANTHROPIC_BLIND_TOOL_SCHEMAS = _to_anthropic(tool_schemas(True))


class AnthropicLLMBackend:
    """Real backend: Anthropic Messages API with a bounded tool loop."""

    name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 transport=None, max_model_calls: int = MAX_MODEL_CALLS,
                 blind: bool = False):
        self.model = model or os.environ.get("TEAMSCLAE_MODEL", DEFAULT_MODEL)
        self._api_key = (api_key if api_key is not None
                         else resolve_anthropic_api_key())
        self._transport = transport or _default_transport
        self.max_model_calls = max_model_calls
        self.blind = blind

    # -- one API round trip; returns content blocks + stop reason + usage
    def _call_model(self, messages: list, system: str | None = None,
                    schemas=None) -> tuple[list, str | None, int, int]:
        payload = {"model": self.model, "max_tokens": MAX_TOKENS,
                   "system": (system if system is not None else
                              BLIND_SYSTEM_PROMPT if self.blind
                              else SYSTEM_PROMPT),
                   "messages": messages,
                   "tools": (schemas if schemas is not None else
                             ANTHROPIC_BLIND_TOOL_SCHEMAS if self.blind
                             else ANTHROPIC_TOOL_SCHEMAS)}
        headers = {"x-api-key": self._api_key,
                   "anthropic-version": ANTHROPIC_VERSION,
                   "Content-Type": "application/json"}
        resp = self._transport(API_URL, headers, payload)
        usage = resp.get("usage") or {}
        return (resp.get("content") or [], resp.get("stop_reason"),
                int(usage.get("input_tokens", 0)),
                int(usage.get("output_tokens", 0)))

    def fix_module(self, module: RealModule, sandbox: Path, *,
                   board_text: str = "", feedback: str | None = None,
                   token_budget: int = 75_000, agent_id: int = 0) -> FixResult:
        tools = SandboxTools(sandbox)
        result = FixResult()
        user = build_user_prompt(module, self.blind, board_text, feedback)
        messages = [{"role": "user", "content": user}]
        dispatch = {
            "list_files": lambda a: tools.list_files(),
            "read_file": lambda a: tools.read_file(a["path"]),
            "write_file": lambda a: tools.write_file(a["path"], a["content"]),
        }
        if not self.blind:
            dispatch["run_check"] = \
                lambda a: tools.run_check(a["check_file"])
        for _ in range(self.max_model_calls):
            if result.tokens >= token_budget:
                result.budget_exceeded = True
                return result
            try:
                blocks, stop_reason, tin, tout = self._call_model(messages)
            except BackendError as e:
                result.error = str(e)
                return result
            result.tokens_in += tin
            result.tokens_out += tout
            result.model_calls += 1
            result.events.append({"event": "model_call", "agent": agent_id,
                                  "subtask": module.subtask_id,
                                  "tokens_in": tin, "tokens_out": tout})
            messages.append({"role": "assistant", "content": blocks})
            tool_uses = [b for b in blocks
                         if isinstance(b, dict) and b.get("type") == "tool_use"]
            if not tool_uses:
                if stop_reason == "max_tokens":
                    result.error = "model hit max_tokens before finishing"
                elif stop_reason == "refusal":
                    result.error = "model refused the subtask"
                else:
                    result.declared_done = True
                return result
            tool_results = []
            for block in tool_uses:
                fn = block.get("name", "")
                args = block.get("input") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                result.events.append({
                    "event": "tool_call", "agent": agent_id,
                    "subtask": module.subtask_id, "tool": fn,
                    "file": args.get("path") or args.get("check_file") or ""})
                is_error = False
                try:
                    output = dispatch[fn](args)
                except KeyError:
                    output = f"error: unknown or unavailable tool: {fn}"
                    is_error = True
                except SandboxViolation as e:
                    output = f"error: {e}"
                    is_error = True
                except Exception as e:  # tool errors are data for the model
                    output = f"error: {type(e).__name__}: {e}"
                    is_error = True
                if isinstance(output, str) and output.startswith("error:"):
                    is_error = True
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.get("id", ""),
                    "content": str(output)[:8000],
                    "is_error": is_error})
            messages.append({"role": "user", "content": tool_results})
        result.error = f"exceeded {self.max_model_calls} model calls"
        return result

    def fix_step(self, *, prompt: str, system_prompt: str, sandbox: Path,
                 token_budget: int = 25_000, agent_id: int = 0,
                 subtask: str = "step") -> FixResult:
        """One chained-task step (see OpenAILLMBackend.fix_step): the
        caller builds the prompt; the loop runs the blind tool surface
        only -- steps have no run_check, ever."""
        tools = SandboxTools(sandbox)
        result = FixResult()
        messages = [{"role": "user", "content": prompt}]
        dispatch = {
            "list_files": lambda a: tools.list_files(),
            "read_file": lambda a: tools.read_file(a["path"]),
            "write_file": lambda a: tools.write_file(a["path"],
                                                      a["content"]),
        }
        for _ in range(self.max_model_calls):
            if result.tokens >= token_budget:
                result.budget_exceeded = True
                return result
            try:
                blocks, stop_reason, tin, tout = self._call_model(
                    messages, system=system_prompt,
                    schemas=ANTHROPIC_BLIND_TOOL_SCHEMAS)
            except BackendError as e:
                result.error = str(e)
                return result
            result.tokens_in += tin
            result.tokens_out += tout
            result.model_calls += 1
            result.events.append({"event": "model_call", "agent": agent_id,
                                  "subtask": subtask,
                                  "tokens_in": tin, "tokens_out": tout})
            messages.append({"role": "assistant", "content": blocks})
            tool_uses = [b for b in blocks
                         if isinstance(b, dict)
                         and b.get("type") == "tool_use"]
            if not tool_uses:
                if stop_reason == "max_tokens":
                    result.error = "model hit max_tokens before finishing"
                elif stop_reason == "refusal":
                    result.error = "model refused the subtask"
                else:
                    result.declared_done = True
                return result
            tool_results = []
            for block in tool_uses:
                fn = block.get("name", "")
                args = block.get("input") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                result.events.append({
                    "event": "tool_call", "agent": agent_id,
                    "subtask": subtask, "tool": fn,
                    "file": args.get("path") or ""})
                is_error = False
                try:
                    output = dispatch[fn](args)
                except KeyError:
                    output = f"error: unknown or unavailable tool: {fn}"
                    is_error = True
                except SandboxViolation as e:
                    output = f"error: {e}"
                    is_error = True
                except Exception as e:  # tool errors are data for the model
                    output = f"error: {type(e).__name__}: {e}"
                    is_error = True
                if isinstance(output, str) and output.startswith("error:"):
                    is_error = True
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.get("id", ""),
                    "content": str(output)[:8000],
                    "is_error": is_error})
            messages.append({"role": "user", "content": tool_results})
        result.error = f"exceeded {self.max_model_calls} model calls"
        return result

    # -- AgentBackend protocol compatibility: same guard rail as the
    # OpenAI backend; real code-fixing agents work on live sandboxes.
    def attempt(self, subtask, ctx, rng):  # pragma: no cover - guard rail
        raise NotImplementedError(
            "the real backend works on live sandboxes, not simulated "
            "payloads: drive it through teamscale.real_pilot (fix_module).")
