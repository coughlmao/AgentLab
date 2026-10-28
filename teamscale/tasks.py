"""Task families: DAGs of subtasks with dependencies and ground-truth verifiers.

Families
--------
research_sweep   -- many independent fact lookups over a generated corpus.
                    Embarrassingly parallel; team size should help until
                    coordination/duplication costs dominate.
codebase_repair  -- subtasks that write small Python modules inside a per-run
                    sandbox directory; verified for real by executing the
                    module's check script in a subprocess with a timeout.
long_horizon     -- a chain of dependent subtasks; later steps consume earlier
                    outputs, so errors propagate and parallelism barely helps.
"""
from __future__ import annotations

import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------- sandboxing
class SandboxViolation(Exception):
    pass


def ensure_within(base: Path, target: Path) -> Path:
    """Resolve `target` and refuse anything that escapes `base`."""
    base_r = base.resolve()
    tgt_r = (base_r / target).resolve() if not Path(target).is_absolute() else Path(target).resolve()
    if tgt_r != base_r and base_r not in tgt_r.parents:
        raise SandboxViolation(f"path escapes sandbox: {target}")
    return tgt_r


def write_in_sandbox(base: Path, relname: str, contents: str) -> Path:
    path = ensure_within(base, Path(relname))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    return path


# ---------------------------------------------------------------- subtasks
@dataclass
class Subtask:
    subtask_id: str
    deps: tuple = ()
    difficulty: float = 0.3        # 0..1
    expected_steps: int = 3
    family: str = ""
    # fact / chain families: the correct payload and its "topic" prefix
    answer: str = ""
    topic: str = ""
    # codebase family: file to write and the check script that verifies it
    filename: str = ""
    fixed_code: str = ""
    broken_code: str = ""
    check_code: str = ""

    def ground_truth_payload(self) -> str:
        return self.fixed_code if self.family == "codebase_repair" else self.answer

    def distractor_payload(self, rng: random.Random) -> str:
        """A plausible-but-wrong payload.

        For fact/chain subtasks the distractor keeps the topic prefix and gets
        the value wrong -- exactly the shape that lenient (substring/partial
        credit) verifiers over-reward, which the audits measure.
        """
        if self.family == "codebase_repair":
            return self.broken_code
        value = self.answer.split(":", 1)[-1]
        wrong = f"{int(rng.random() * 9000) + 1000}" if value.strip().isdigit() else "unknown"
        return f"{self.topic}:{wrong}"


@dataclass
class Task:
    family: str
    subtasks: list = field(default_factory=list)

    def by_id(self):
        return {s.subtask_id: s for s in self.subtasks}

    def depth_of(self, subtask) -> int:
        by_id = self.by_id()
        depth, cur = 0, subtask
        seen = set()
        while cur.deps and cur.subtask_id not in seen:
            seen.add(cur.subtask_id)
            cur = by_id[cur.deps[0]]
            depth += 1
        return depth

    def total_expected_tokens(self, tokens_per_step: int = 100) -> int:
        return int(sum(s.expected_steps * tokens_per_step * (1 + 0.5 * s.difficulty)
                       for s in self.subtasks))


# ---------------------------------------------------------------- verifiers
def strict_verify(subtask: Subtask, payload: str, sandbox: Path | None) -> bool:
    """Ground truth. For code subtasks this really executes the check script."""
    if subtask.family == "codebase_repair":
        assert sandbox is not None
        write_in_sandbox(sandbox, subtask.filename, payload)
        check = write_in_sandbox(sandbox, f"check_{subtask.filename}", subtask.check_code)
        try:
            proc = subprocess.run([sys.executable, str(check)], cwd=str(sandbox),
                                  capture_output=True, timeout=10)
            return proc.returncode == 0
        except subprocess.TimeoutExpired:
            return False
    return payload == subtask.answer


def lenient_score(subtask: Subtask, payload: str, sandbox: Path | None) -> float:
    """Partial credit, the way generous LLM-judge / substring scorers behave.

    Fact/chain: full credit for the exact answer, half credit for getting the
    topic right with a wrong value. Code: half credit if the file at least
    parses and defines the expected function, even if its checks fail.
    """
    if subtask.family == "codebase_repair":
        if sandbox is not None and strict_verify(subtask, payload, sandbox):
            return 1.0
        if payload == subtask.fixed_code:
            return 1.0
        try:
            compile(payload, subtask.filename, "exec")
            return 0.5 if "def " in payload else 0.0
        except SyntaxError:
            return 0.0
    if payload == subtask.answer:
        return 1.0
    return 0.5 if payload.split(":", 1)[0] == subtask.topic else 0.0


# ---------------------------------------------------------------- generators
def make_research_sweep(n: int = 16, seed: int = 0) -> Task:
    rng = random.Random(seed ^ 0x51)
    subs = []
    for i in range(n):
        topic = f"fact-{i:02d}"
        subs.append(Subtask(
            subtask_id=f"rs-{i:02d}", family="research_sweep", topic=topic,
            answer=f"{topic}:{rng.randint(1000, 9999)}",
            difficulty=round(rng.uniform(0.05, 0.55), 2),
            expected_steps=rng.randint(2, 4)))
    return Task("research_sweep", subs)


_CODE_TEMPLATES = [
    # (broken, fixed) bodies for `def f_{i}(x):` verified by check script
    ("    return x + {k} + 1", "    return x + {k}"),
    ("    return x * {k} + x", "    return x * {k}"),
    ("    return x - {k}", "    return x + {k}"),
]


def _code_pair(i: int, rng: random.Random):
    k = rng.randint(2, 9)
    broken_t, fixed_t = _CODE_TEMPLATES[i % len(_CODE_TEMPLATES)]
    header = f"def f_{i}(x):\n"
    broken = header + broken_t.format(k=k) + "\n"
    fixed = header + fixed_t.format(k=k) + "\n"
    expr = fixed_t.format(k=k).strip().removeprefix("return ")
    lines = []
    for v in (0, 1, 3, 7):
        expected = eval(expr, {"__builtins__": {}}, {"x": v})  # tiny arithmetic only
        lines.append(f"assert f_{i}({v}) == {expected}")
    check_code = f"from mod_{i} import f_{i}\n" + "\n".join(lines) + "\nprint('ok')\n"
    return broken, fixed, check_code


def make_codebase_repair(n: int = 8, seed: int = 0) -> Task:
    rng = random.Random(seed ^ 0xC0DE)
    subs = []
    for i in range(n):
        broken, fixed, check = _code_pair(i, rng)
        subs.append(Subtask(
            subtask_id=f"cr-{i:02d}", family="codebase_repair",
            filename=f"mod_{i}.py", fixed_code=fixed, broken_code=broken,
            check_code=check,
            difficulty=round(rng.uniform(0.25, 0.7), 2),
            expected_steps=rng.randint(3, 5)))
    return Task("codebase_repair", subs)


def make_long_horizon(n: int = 12, seed: int = 0) -> Task:
    rng = random.Random(seed ^ 0x11)
    subs = []
    acc = rng.randint(1000, 9999)
    for i in range(n):
        acc = (acc * 7 + 13) % 10000
        topic = f"step-{i:02d}"
        subs.append(Subtask(
            subtask_id=f"lh-{i:02d}", family="long_horizon", topic=topic,
            deps=(f"lh-{i - 1:02d}",) if i else (),
            answer=f"{topic}:{acc:04d}",
            difficulty=round(rng.uniform(0.2, 0.6), 2),
            expected_steps=rng.randint(3, 5)))
    return Task("long_horizon", subs)


FAMILIES = {
    "research_sweep": make_research_sweep,
    "codebase_repair": make_codebase_repair,
    "long_horizon": make_long_horizon,
}


def make_task(family: str, seed: int = 0, horizon_scale: float = 1.0) -> Task:
    """Build a task; horizon_scale stretches the long-horizon chain length
    (and modestly grows the other families) so horizon effects can be swept."""
    if family not in FAMILIES:
        raise ValueError(f"unknown task family: {family}")
    if family == "long_horizon":
        return FAMILIES[family](n=max(4, int(round(12 * horizon_scale))), seed=seed)
    if family == "research_sweep":
        return FAMILIES[family](n=max(4, int(round(16 * horizon_scale))), seed=seed)
    return FAMILIES[family](n=max(3, int(round(8 * horizon_scale))), seed=seed)
