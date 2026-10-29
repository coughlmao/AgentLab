"""A deterministic fake LLM backend for tests, CI, and dry runs.

It mimics the real backend's shape -- same fix_module signature, same tool
surface (SandboxTools), same event/token accounting -- without any network.
Instead of calling a model, it "reasons" by script:

  call 1: read the target module (and the team board, if one is provided)
  call 2: apply a scripted repair for the module's known defect type, by
          applying the (find -> replace) pair the generator embedded
  call 3: run the module's check script to confirm

Competence is configurable: with probability `competence` (seeded RNG) an
attempt applies the correct repair; otherwise it makes a plausible but
wrong edit (rewrites the file with a harmless comment appended), so checks
fail the way a real miss would. Token usage is synthetic but follows the
same accounting rules as the real backend: every model call charges
input/output tokens, recorded as events, and the per-subtask token budget
is enforced before each call.

Blind parity: with `blind=True` the scripted third call submits without
running the check, mirroring the real backends, which do not offer the
run_check tool at all in blind mode.
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

from .llm_backend import FixResult, SandboxTools
from .real_tasks import RealModule


class FakeLLMBackend:
    name = "fake"
    model = "fake"
    # The chain runner dispatches fix_step differently for scripted
    # backends (it passes apply/corrupt callables instead of a prompt).
    scripted_steps = True

    def __init__(self, competence: float = 0.85, seed: int = 0,
                 blind: bool = False):
        self.competence = competence
        self.seed = seed
        self.blind = blind

    def _rng_for(self, subtask_id: str, retry: bool) -> random.Random:
        digest = hashlib.sha256(
            f"{self.seed}|{subtask_id}|{retry}".encode()).hexdigest()
        return random.Random(int(digest[:8], 16))

    def fix_module(self, module: RealModule, sandbox: Path, *,
                   board_text: str = "", feedback: str | None = None,
                   token_budget: int = 75_000, agent_id: int = 0) -> FixResult:
        tools = SandboxTools(sandbox)
        result = FixResult()
        rng = self._rng_for(module.subtask_id, bool(feedback))
        board_tokens = len(board_text) // 4 if board_text else 0

        def charge(tin: int, tout: int) -> bool:
            """Record one synthetic model call; False if budget is blown."""
            if result.tokens + tin + tout >= token_budget and result.model_calls:
                result.budget_exceeded = True
                return False
            result.tokens_in += tin
            result.tokens_out += tout
            result.model_calls += 1
            result.events.append({"event": "model_call", "agent": agent_id,
                                  "subtask": module.subtask_id,
                                  "tokens_in": tin, "tokens_out": tout})
            return True

        def tool(name: str, file: str, fn):
            result.events.append({"event": "tool_call", "agent": agent_id,
                                  "subtask": module.subtask_id, "tool": name,
                                  "file": file})
            return fn()

        # call 1: read the file the repair applies to (and the board).
        # Usually that is the module itself; for cross-module defects it
        # is the helper the module imports.
        target_file = module.repair_file or module.module_file
        code = tool("read_file", target_file,
                    lambda: tools.read_file(target_file))
        if not charge(700 + len(code) // 2 + board_tokens, 150):
            return result
        # call 2: decide and apply an edit
        competent = rng.random() < self.competence
        if feedback:
            competent = competent or rng.random() < 0.5  # feedback helps
        if competent:
            find, repl = module.repair
            new_code = code.replace(find, repl, 1)
        else:
            new_code = code + "# attempted fix (ineffective)\n"
        tool("write_file", target_file,
             lambda: tools.write_file(target_file, new_code))
        if not charge(600 + len(new_code) // 2, 320):
            return result
        # call 3: run the check to confirm -- unless blind, where the
        # agent cannot run checks and simply submits the fix.
        if not self.blind:
            tool("run_check", module.check_file,
                 lambda: tools.run_check(module.check_file))
        if not charge(450, 90):
            return result
        result.declared_done = True
        return result

    def fix_step(self, *, sandbox: Path, state_file: str = "state.py",
                 apply_fn, corrupt_fn, subtask: str = "step",
                 token_budget: int = 25_000, agent_id: int = 0,
                 feedback: str | None = None) -> FixResult:
        """Scripted chain step, mirroring fix_module's accounting: read
        the state file, then write either the correct transformation
        of the actual input (apply_fn -- the scripted repair) or a
        plausible corruption of it, with `competence` probability on
        the first and boosted on a feedback retry."""
        from .chains import parse_state_file, render_state_file
        tools = SandboxTools(sandbox)
        result = FixResult()
        rng = self._rng_for(subtask, bool(feedback))

        def charge(tin: int, tout: int) -> bool:
            if result.tokens + tin + tout >= token_budget \
                    and result.model_calls:
                result.budget_exceeded = True
                return False
            result.tokens_in += tin
            result.tokens_out += tout
            result.model_calls += 1
            result.events.append({"event": "model_call", "agent": agent_id,
                                  "subtask": subtask,
                                  "tokens_in": tin, "tokens_out": tout})
            return True

        def tool(name: str, file: str, fn):
            result.events.append({"event": "tool_call", "agent": agent_id,
                                  "subtask": subtask, "tool": name,
                                  "file": file})
            return fn()

        raw = tool("read_file", state_file,
                   lambda: tools.read_file(state_file))
        if not charge(400 + len(raw) // 2, 120):
            return result
        current = parse_state_file(raw)
        if not isinstance(current, list):
            current = None
        competent = rng.random() < self.competence
        if feedback:
            competent = competent or rng.random() < 0.5
        if competent and current is not None:
            new_state = apply_fn(current)
        else:
            new_state = corrupt_fn(current, rng)
        payload = render_state_file(new_state)
        tool("write_file", state_file,
             lambda: tools.write_file(state_file, payload))
        if not charge(400 + len(payload) // 2, 200):
            return result
        if not charge(300, 80):
            return result
        result.declared_done = True
        return result
