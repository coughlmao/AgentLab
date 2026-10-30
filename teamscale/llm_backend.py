"""A real LLM backend: an API-driven agent that fixes code in a sandbox.

This is the production counterpart of `SimulatedAgent` (see agents.py). It
implements the same idea -- turn a subtask into work, account for every
token -- but the work is real: the model reads a broken module inside a
sandbox directory, edits it with tools, and runs the module's check script
until it passes or the attempt budget runs out.

Design constraints:

- Standard library only (urllib). No SDK dependency.
- The API key is NEVER read from a file, logged, printed, or written into
  traces or results. It comes from exactly two places:
    1. the shell command in env TEAMSCALE_CRED_CMD (its stdout is the key;
       TEAMSCLAE_CRED_CMD is supported as a legacy alias), or
    2. env OPENAI_API_KEY, if already present in the process environment.
  If neither is configured, construction raises a clear error.
- Model: env TEAMSCALE_MODEL, default "gpt-5-mini"
  (TEAMSCLAE_MODEL is supported as a legacy alias).
- Every tool call is sandbox-contained (teamscale.tasks.ensure_within):
  the model can only touch files inside the run's sandbox directory.
- Token usage is taken from the API responses themselves (prompt/completion
  tokens per call), summed exactly, and returned with the result. A
  per-subtask token budget is enforced: once usage reaches it, the loop
  stops and the attempt is reported as failed with usage so far.
- Blind mode (constructor flag `blind=True`): the run_check tool is not
  offered and not dispatched, and the system prompt says the work will be
  checked externally after submission. Agents can still read the check
  script; they just cannot execute it. This is the Phase A laboratory
  condition: with self-checking removed, external verification (the
  runner's re-execution in the `verified` strategy) is the only check
  feedback in the system.

The runner that drives this backend over full team configurations is
teamscale.real_pilot; this module is the per-subtask worker.
"""
from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .real_tasks import RealModule
from .tasks import SandboxViolation, ensure_within

API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-5-mini"
MAX_MODEL_CALLS = 10
CHECK_TIMEOUT = 10


def teamscale_env(name: str, default: str | None = None) -> str | None:
    """Read a TEAMSCALE_ variable, falling back to the legacy TEAMSCLAE_ name."""
    value = os.environ.get(f"TEAMSCALE_{name}")
    if value is not None:
        return value
    return os.environ.get(f"TEAMSCLAE_{name}", default)


class BackendError(RuntimeError):
    pass


def resolve_api_key() -> str:
    """Return the API key from the configured credential source.

    The key is returned to the caller (the backend constructor) and nowhere
    else; it is never logged or persisted by this project.
    """
    cmd = (teamscale_env("CRED_CMD", "") or "").strip()
    if cmd:
        proc = subprocess.run(cmd, shell=True, capture_output=True,
                              text=True, timeout=30)
        key = proc.stdout.strip()
        if proc.returncode != 0 or not key:
            raise BackendError(
                "TEAMSCALE_CRED_CMD (legacy alias: TEAMSCLAE_CRED_CMD) "
                "ran but produced no key on stdout "
                f"(exit {proc.returncode}); check the credential command.")
        return key
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    raise BackendError(
        "no API key configured: set TEAMSCALE_CRED_CMD "
        "(legacy alias: TEAMSCLAE_CRED_CMD) to a command whose "
        "stdout is the key, or export OPENAI_API_KEY.")


# --------------------------------------------------------------------- tools
class SandboxTools:
    """The tool surface an agent gets over one run sandbox.

    Every path is resolved through ensure_within, so ../ escapes, absolute
    paths outside the sandbox, and symlink tricks raise SandboxViolation.
    """

    def __init__(self, sandbox: Path):
        self.sandbox = Path(sandbox)

    def list_files(self) -> str:
        names = sorted(p.name for p in self.sandbox.iterdir() if p.is_file())
        return "\n".join(names)

    def read_file(self, path: str) -> str:
        target = ensure_within(self.sandbox, Path(path))
        if not target.is_file():
            return f"error: no such file: {path}"
        return target.read_text()[:20000]

    def write_file(self, path: str, content: str) -> str:
        if not str(path).endswith(".py"):
            raise SandboxViolation(f"agents may only write .py files: {path}")
        target = ensure_within(self.sandbox, Path(path))
        target.write_text(content)
        return f"wrote {path} ({len(content)} bytes)"

    def run_check(self, check_file: str) -> str:
        """Execute a check script in the sandbox; return exit status + tail."""
        import sys
        target = ensure_within(self.sandbox, Path(check_file))
        if not target.is_file():
            return f"error: no such check file: {check_file}"
        import shutil
        shutil.rmtree(self.sandbox / "__pycache__", ignore_errors=True)
        try:
            proc = subprocess.run([sys.executable, "-B", str(target)],
                                  cwd=str(self.sandbox), capture_output=True,
                                  text=True, timeout=CHECK_TIMEOUT)
            tail = (proc.stdout + proc.stderr).strip()[-2000:]
            status = "PASS" if proc.returncode == 0 else f"FAIL (exit {proc.returncode})"
            return f"{status}\n{tail}"
        except subprocess.TimeoutExpired:
            return f"FAIL (check timed out after {CHECK_TIMEOUT}s)"


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "list_files", "description": "List files in the sandbox.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file from the sandbox.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Overwrite a .py file in the sandbox with new content.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_check",
        "description": "Run a check script; returns PASS/FAIL and output.",
        "parameters": {"type": "object",
                       "properties": {"check_file": {"type": "string"}},
                       "required": ["check_file"]}}},
]

# Blind mode offers every tool except run_check: the agent can read the
# check script but cannot execute it.
BLIND_TOOL_SCHEMAS = [s for s in TOOL_SCHEMAS
                      if s["function"]["name"] != "run_check"]


def tool_schemas(blind: bool) -> list:
    return BLIND_TOOL_SCHEMAS if blind else TOOL_SCHEMAS


# ------------------------------------------------------------------- results
@dataclass
class FixResult:
    """Outcome of one backend attempt at one subtask."""
    declared_done: bool = False        # the agent stopped on its own, satisfied
    tokens_in: int = 0
    tokens_out: int = 0
    model_calls: int = 0
    budget_exceeded: bool = False
    error: str | None = None
    events: list = field(default_factory=list)  # model_call / tool_call records

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out


SYSTEM_PROMPT = (
    "You are a software engineer fixing one broken Python module. You work "
    "only inside a sandbox directory, using the provided tools. Read the "
    "module and its check script, find the defect, rewrite the module so "
    "the check passes, and run the check to confirm. Only edit the target "
    "module file. Be economical: few, decisive tool calls. When the check "
    "passes, stop and reply with a one-line summary."
)

BLIND_SYSTEM_PROMPT = (
    "You are a software engineer fixing one broken Python module. You work "
    "only inside a sandbox directory, using the provided tools. Read the "
    "module and its check script, find the defect, and rewrite the module "
    "so the check would pass. You cannot run the check yourself: your "
    "work will be checked externally after you submit it. Only edit the "
    "target module file. Be economical: few, decisive tool calls. When "
    "you believe the fix is correct, stop and reply with a one-line "
    "summary."
)

# The user-prompt templates, as named constants, so the exact prompt
# text is versioned, shared by both real backends (one definition, no
# drift), and hashable for run-manifest provenance (ENGINEERING.md,
# invariant 3). build_user_prompt reproduces the strings the backends
# have always sent, byte for byte.
USER_TEMPLATE = (
    "Subtask: {description}\n"
    "Target module: {module_file}\n"
    "Check script: {check_file}\n"
    "Fix {module_file} so that running {check_file} passes. "
    "Do not edit the check script."
)
BLIND_USER_TEMPLATE = (
    "Subtask: {description}\n"
    "Target module: {module_file}\n"
    "Check script: {check_file}\n"
    "Fix {module_file} so that {check_file} passes when it is run "
    "externally after you submit. You cannot run the check yourself. "
    "Do not edit the check script."
)
BOARD_SUFFIX = "\n\nTeam board (shared status):\n{board_text}"
FEEDBACK_SUFFIX = (
    "\n\nYour previous attempt was checked externally and FAILED. "
    "Check output:\n{feedback}\nFix the remaining problem; do not "
    "repeat the same change blindly."
)

PROMPT_TEMPLATES = [SYSTEM_PROMPT, BLIND_SYSTEM_PROMPT, USER_TEMPLATE,
                    BLIND_USER_TEMPLATE, BOARD_SUFFIX, FEEDBACK_SUFFIX]


def build_user_prompt(module, blind: bool, board_text: str = "",
                      feedback: str | None = None) -> str:
    template = BLIND_USER_TEMPLATE if blind else USER_TEMPLATE
    user = template.format(description=module.description,
                           module_file=module.module_file,
                           check_file=module.check_file)
    if board_text:
        user += BOARD_SUFFIX.format(board_text=board_text)
    if feedback:
        user += FEEDBACK_SUFFIX.format(feedback=feedback)
    return user


def prompt_hash() -> str:
    """SHA-256 of every prompt template the repair-family backends can
    send (system + user + suffixes), for manifest provenance."""
    import hashlib
    return hashlib.sha256(
        "\n".join(PROMPT_TEMPLATES).encode()).hexdigest()


def _default_transport(url: str, headers: dict, payload: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:500]
        raise BackendError(f"API returned HTTP {e.code}: {body}") from None
    except urllib.error.URLError as e:
        raise BackendError(f"API request failed: {e.reason}") from None


class OpenAILLMBackend:
    """Real backend: OpenAI Chat Completions with a bounded tool loop."""

    name = "openai"

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 transport=None, max_model_calls: int = MAX_MODEL_CALLS,
                 blind: bool = False):
        self.model = model or teamscale_env("MODEL", DEFAULT_MODEL)
        self._api_key = api_key if api_key is not None else resolve_api_key()
        self._transport = transport or _default_transport
        self.max_model_calls = max_model_calls
        self.blind = blind

    # -- one API round trip; returns the parsed message + usage
    def _call_model(self, messages: list, schemas=None) -> tuple[dict, int, int]:
        payload = {"model": self.model, "messages": messages,
                   "tools": (schemas if schemas is not None
                             else tool_schemas(self.blind)),
                   "tool_choice": "auto"}
        headers = {"Authorization": f"Bearer {self._api_key}",
                   "Content-Type": "application/json"}
        resp = self._transport(API_URL, headers, payload)
        usage = resp.get("usage") or {}
        return (resp["choices"][0]["message"],
                int(usage.get("prompt_tokens", 0)),
                int(usage.get("completion_tokens", 0)))

    def fix_module(self, module: RealModule, sandbox: Path, *,
                   board_text: str = "", feedback: str | None = None,
                   token_budget: int = 75_000, agent_id: int = 0) -> FixResult:
        tools = SandboxTools(sandbox)
        result = FixResult()
        user = build_user_prompt(module, self.blind, board_text, feedback)
        messages = [{"role": "system",
                     "content": BLIND_SYSTEM_PROMPT if self.blind
                     else SYSTEM_PROMPT},
                    {"role": "user", "content": user}]
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
                message, tin, tout = self._call_model(messages)
            except BackendError as e:
                result.error = str(e)
                return result
            result.tokens_in += tin
            result.tokens_out += tout
            result.model_calls += 1
            result.events.append({"event": "model_call", "agent": agent_id,
                                  "subtask": module.subtask_id,
                                  "tokens_in": tin, "tokens_out": tout})
            messages.append(message)
            calls = message.get("tool_calls") or []
            if not calls:
                result.declared_done = True
                return result
            for call in calls:
                fn = call["function"]["name"]
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                result.events.append({
                    "event": "tool_call", "agent": agent_id,
                    "subtask": module.subtask_id, "tool": fn,
                    "file": args.get("path") or args.get("check_file") or ""})
                try:
                    output = dispatch[fn](args)
                except KeyError:
                    output = f"error: unknown or unavailable tool: {fn}"
                except SandboxViolation as e:
                    output = f"error: {e}"
                except Exception as e:  # tool errors are data for the model
                    output = f"error: {type(e).__name__}: {e}"
                messages.append({"role": "tool",
                                 "tool_call_id": call.get("id", ""),
                                 "content": str(output)[:8000]})
        result.error = f"exceeded {self.max_model_calls} model calls"
        return result

    def fix_step(self, *, prompt: str, system_prompt: str, sandbox: Path,
                 token_budget: int = 25_000, agent_id: int = 0,
                 subtask: str = "step") -> FixResult:
        """One chained-task step: the caller (teamscale.chain_experiment)
        builds the prompt; this loop just runs the blind tool surface
        (list/read/write -- there is no run_check for steps, ever)
        against the sandbox until the model stops or the budget ends.
        Same accounting and event shapes as fix_module."""
        tools = SandboxTools(sandbox)
        result = FixResult()
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
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
                message, tin, tout = self._call_model(
                    messages, schemas=tool_schemas(True))
            except BackendError as e:
                result.error = str(e)
                return result
            result.tokens_in += tin
            result.tokens_out += tout
            result.model_calls += 1
            result.events.append({"event": "model_call", "agent": agent_id,
                                  "subtask": subtask,
                                  "tokens_in": tin, "tokens_out": tout})
            messages.append(message)
            calls = message.get("tool_calls") or []
            if not calls:
                result.declared_done = True
                return result
            for call in calls:
                fn = call["function"]["name"]
                try:
                    args = json.loads(call["function"].get("arguments")
                                      or "{}")
                except json.JSONDecodeError:
                    args = {}
                result.events.append({
                    "event": "tool_call", "agent": agent_id,
                    "subtask": subtask, "tool": fn,
                    "file": args.get("path") or ""})
                try:
                    output = dispatch[fn](args)
                except KeyError:
                    output = f"error: unknown or unavailable tool: {fn}"
                except SandboxViolation as e:
                    output = f"error: {e}"
                except Exception as e:  # tool errors are data for the model
                    output = f"error: {type(e).__name__}: {e}"
                messages.append({"role": "tool",
                                 "tool_call_id": call.get("id", ""),
                                 "content": str(output)[:8000]})
        result.error = f"exceeded {self.max_model_calls} model calls"
        return result

    # -- AgentBackend protocol compatibility: the simulated harness calls
    # attempt() with no sandbox, which a real code-fixing agent cannot use.
    def attempt(self, subtask, ctx, rng):  # pragma: no cover - guard rail
        raise NotImplementedError(
            "the real backend works on live sandboxes, not simulated "
            "payloads: drive it through teamscale.real_pilot (fix_module).")
