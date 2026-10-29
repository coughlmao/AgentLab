"""Chained-transformation tasks: the Phase A-2 task family.

A chain is a sequence of N steps over one shared state (a list of
STATE_LEN integers, each in 0..999). Step k's specification is a
transformation defined on the state produced by step k-1, so the
correct input to step k exists only if step k-1 ended in the correct
state. An unrecovered error therefore propagates: that is the error
accumulation this family exists to study (see
experiments/error-accumulation/DESIGN.md).

Generator determinism (ENGINEERING.md, invariant 1): a chain is a pure
function of (seed, generator_version). Every random draw flows from one
RNG seeded by a SHA-256 digest of those two values; no timestamps, no
host or environment data, no global RNG state. Templates are constants
in this module; changing any template, parameter distribution, the
state length, or the warmup increments GENERATOR_VERSION.

Steps are sampled independently of position: the template and its
parameters are drawn iid at every step, and a fixed warmup (random
transformations applied before step 1, not part of the chain) puts the
initial state into the same distribution that later steps see, so
position carries no difficulty signal by construction. The template
set is restricted to transformations that preserve the uniform
marginal distribution of state values (permutations, modular
arithmetic with coprime multipliers, pairwise modular sums and
differences): contractive maps (absolute differences, clamps) were
rejected during development because they concentrate the state
distribution over time, which makes defect visibility drift with
position -- exactly the confound the calibration gate exists to
exclude. The offline calibration gate (teamscale.calibrate) verifies
position-independence empirically.

Grading distinguishes two correctness notions per step:
- state correctness: the submitted state equals the ground-truth state
  after step k (computed by the generator along the true chain);
- local correctness: the submitted state equals the step's
  transformation applied to the input the agent actually received --
  an agent can be locally correct while the chain state is wrong
  (it inherited a corrupted state), and that distinction is what makes
  cascades interpretable.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field

GENERATOR_VERSION = "chain-gen-1"
STATE_LEN = 12
MOD = 1000
WARMUP_OPS = 8        # transformations applied before step 1 (see above)
PROBE_COUNT = STATE_LEN + 1   # one probe per index, plus a length probe


# ------------------------------------------------------------- templates
# Each template: name, make_params(rng), apply(state, params),
# mutant_apply(state, params) -- a canonical single-defect variant used
# by the calibration gate's probe-visibility proxy -- spec_text(params)
# and op_count(params). All applies are pure and length-preserving.

def _affine_params(rng, n=STATE_LEN):
    return {"m": rng.choice([3, 7, 9, 11, 13]), "c": rng.randrange(100)}


def _affine_apply(state, p):
    return [(v * p["m"] + p["c"]) % MOD for v in state]


def _affine_mutant(state, p):
    return [(v * p["m"] + (p["c"] + 7) % MOD) % MOD for v in state]


def _rotate_params(rng, n=STATE_LEN):
    return {"r": rng.randrange(1, n)}


def _rotate_apply(state, p):
    r = p["r"] % len(state)
    return state[r:] + state[:r]


def _rotate_mutant(state, p):
    n = len(state)
    r = (p["r"] % (n - 1)) + 1
    if r == p["r"]:
        r = r % (n - 1) + 1
    return state[r:] + state[:r]


def _reverse_apply(state, p):
    return state[::-1]


def _reverse_mutant(state, p):
    return state[1:] + state[:1]


def _neighbor_add_apply(state, p):
    n = len(state)
    return [(state[i] + state[(i + 1) % n]) % MOD for i in range(n)]


def _neighbor_add_mutant(state, p):
    n = len(state)
    return [(state[i] + state[(i - 1) % n]) % MOD for i in range(n)]


def _neighbor_sub_apply(state, p):
    n = len(state)
    return [(state[i] - state[(i + 1) % n]) % MOD for i in range(n)]


def _neighbor_sub_mutant(state, p):
    n = len(state)
    return [(state[i] - state[(i - 1) % n]) % MOD for i in range(n)]


def _threshold_params(rng, n=STATE_LEN):
    return {"h": rng.choice(list(range(200, 801, 50))),
            "t": rng.randrange(5, 51)}


def _threshold_apply(state, p):
    return [((v + p["t"]) % MOD if v < p["h"] else (v - p["t"]) % MOD)
            for v in state]


def _threshold_mutant(state, p):
    h = p["h"] + 50 if p["h"] < 800 else p["h"] - 50
    q = {"h": h, "t": p["t"]}
    return _threshold_apply(state, q)


def _add_index_params(rng, n=STATE_LEN):
    return {"s": rng.randrange(30, 91)}


def _add_index_apply(state, p):
    return [(v + i * p["s"]) % MOD for i, v in enumerate(state)]


def _add_index_mutant(state, p):
    return [(v + i * (p["s"] + 7)) % MOD for i, v in enumerate(state)]


def _swap_params(rng, n=STATE_LEN):
    a, b, c, d = rng.sample(range(n), 4)
    return {"pairs": [a, b, c, d]}


def _swap_apply(state, p):
    a, b, c, d = p["pairs"]
    out = list(state)
    out[a], out[b] = out[b], out[a]
    out[c], out[d] = out[d], out[c]
    return out


def _swap_mutant(state, p):
    a, b, c, d = p["pairs"]
    out = list(state)
    out[a], out[c] = out[c], out[a]
    out[b], out[d] = out[d], out[b]
    return out


def _sort_params(rng, n=STATE_LEN):
    a = rng.randrange(0, 6)
    return {"a": a, "b": rng.randrange(a + 3, n)}


def _sort_apply(state, p):
    a, b = p["a"], p["b"]
    return state[:a] + sorted(state[a:b + 1]) + state[b + 1:]


def _sort_mutant(state, p):
    a, b = p["a"], p["b"]
    return state[:a] + sorted(state[a:b + 1], reverse=True) + state[b + 1:]


def _no_params(rng, n=STATE_LEN):
    return {}


@dataclass(frozen=True)
class _Template:
    name: str
    make_params: object
    apply: object
    mutant_apply: object
    spec_text: object
    op_count: object


TEMPLATES = [
    _Template(
        "affine", _affine_params, _affine_apply, _affine_mutant,
        lambda p: (f"Replace every value v in the state with "
                   f"(v * {p['m']} + {p['c']}) mod 1000."),
        lambda p: STATE_LEN),
    _Template(
        "rotate", _rotate_params, _rotate_apply, _rotate_mutant,
        lambda p: (f"Rotate the state left by {p['r']} positions: the "
                   f"first {p['r']} values move to the end, keeping "
                   f"their relative order."),
        lambda p: STATE_LEN),
    _Template(
        "reverse", _no_params, _reverse_apply, _reverse_mutant,
        lambda p: "Reverse the order of the values in the state.",
        lambda p: STATE_LEN),
    _Template(
        "neighbor_add", _no_params, _neighbor_add_apply,
        _neighbor_add_mutant,
        lambda p: ("Replace each value with the sum, mod 1000, of "
                   "itself and the next value in the state; the last "
                   "value pairs with the first."),
        lambda p: STATE_LEN),
    _Template(
        "neighbor_sub", _no_params, _neighbor_sub_apply,
        _neighbor_sub_mutant,
        lambda p: ("Replace each value with itself minus the next "
                   "value in the state, mod 1000; the last value "
                   "pairs with the first."),
        lambda p: STATE_LEN),
    _Template(
        "threshold_add", _threshold_params, _threshold_apply,
        _threshold_mutant,
        lambda p: (f"For each value v in the state: if v < {p['h']}, "
                   f"replace it with (v + {p['t']}) mod 1000; otherwise "
                   f"replace it with (v - {p['t']}) mod 1000."),
        lambda p: STATE_LEN),
    _Template(
        "add_index", _add_index_params, _add_index_apply,
        _add_index_mutant,
        lambda p: (f"Add i * {p['s']} (mod 1000) to the value at "
                   f"position i, where i is the 0-based position: "
                   f"position 0 is unchanged, position 1 gains "
                   f"{p['s']}, position 2 gains {2 * p['s']}, and so "
                   f"on."),
        lambda p: STATE_LEN),
    _Template(
        "swap_pairs", _swap_params, _swap_apply, _swap_mutant,
        lambda p: (f"Swap the values at positions {p['pairs'][0]} and "
                   f"{p['pairs'][1]}, and swap the values at positions "
                   f"{p['pairs'][2]} and {p['pairs'][3]}. Positions are "
                   f"0-based."),
        lambda p: 4),
    _Template(
        "sort_window", _sort_params, _sort_apply, _sort_mutant,
        lambda p: (f"Sort the values at positions {p['a']} through "
                   f"{p['b']} (inclusive, 0-based) into ascending "
                   f"order; leave every other position unchanged."),
        lambda p: p["b"] - p["a"] + 1),
]
TEMPLATE_NAMES = [t.name for t in TEMPLATES]
_BY_NAME = {t.name: t for t in TEMPLATES}


# ----------------------------------------------------------------- steps
@dataclass
class ChainStep:
    index: int                 # 1-based position in the chain
    template: str
    params: dict
    spec_text: str
    input_state: list          # ground-truth state before this step
    expected_state: list       # ground-truth state after this step
    op_count: int

    def apply(self, state):
        """This step's transformation applied to an arbitrary state
        (the local-correctness reference and the fake backend's
        scripted repair). None if `state` is not a valid state."""
        if not is_valid_state(state):
            return None
        return _BY_NAME[self.template].apply(list(state), self.params)

    def mutant_apply(self, state):
        return _BY_NAME[self.template].mutant_apply(list(state),
                                                    self.params)

    @property
    def subtask_id(self) -> str:
        return f"step-{self.index:02d}"

    def to_dict(self) -> dict:
        return {"index": self.index, "template": self.template,
                "params": self.params, "spec_text": self.spec_text,
                "input_state": self.input_state,
                "expected_state": self.expected_state,
                "op_count": self.op_count}


@dataclass
class ChainTask:
    seed: int
    n_steps: int
    generator_version: str
    initial_state: list
    steps: list = field(default_factory=list)
    family: str = "error_accumulation"

    def to_dict(self) -> dict:
        return {"family": self.family, "seed": self.seed,
                "n_steps": self.n_steps,
                "generator_version": self.generator_version,
                "initial_state": self.initial_state,
                "steps": [s.to_dict() for s in self.steps]}

    def canonical_json(self) -> str:
        """Byte-stable serialization: the determinism contract."""
        return json.dumps(self.to_dict(), sort_keys=True,
                          separators=(",", ":"))


STATE_FILE = "state.py"


def render_state_file(state) -> str:
    """The on-disk form of a chain state: a Python module assigning
    STATE. Agents may only write .py files in a sandbox, so the chain
    artifact is one -- the state is data, carried as code."""
    return f"STATE = {list(state)!r}\n"


def parse_state_file(text: str):
    """Inverse of render_state_file: the STATE value, or None if the
    text is not a module assigning STATE a literal. Never executes
    anything (ast + literal_eval only)."""
    import ast
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "STATE"
                for t in node.targets):
            try:
                return ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                return None
    return None


def is_valid_state(state) -> bool:
    return (isinstance(state, list) and len(state) == STATE_LEN
            and all(isinstance(v, int) and not isinstance(v, bool)
                    and 0 <= v < MOD for v in state))


def _seed_rng(seed: int, generator_version: str) -> random.Random:
    digest = hashlib.sha256(
        f"chains|{generator_version}|{seed}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def make_chain(seed: int, n_steps: int,
               generator_version: str = GENERATOR_VERSION) -> ChainTask:
    """Generate one chain: a pure function of (seed, generator_version)."""
    if generator_version != GENERATOR_VERSION:
        raise ValueError(
            f"unknown generator_version {generator_version!r}: "
            f"this code implements {GENERATOR_VERSION!r}")
    if n_steps < 1:
        raise ValueError("a chain needs at least one step")
    rng = _seed_rng(seed, generator_version)
    state = [rng.randrange(MOD) for _ in range(STATE_LEN)]
    for _ in range(WARMUP_OPS):   # warmup: not part of the chain
        t = TEMPLATES[rng.randrange(len(TEMPLATES))]
        state = t.apply(state, t.make_params(rng))
    initial_state = list(state)
    steps = []
    for k in range(1, n_steps + 1):
        t = TEMPLATES[rng.randrange(len(TEMPLATES))]
        params = t.make_params(rng)
        expected = t.apply(state, params)
        steps.append(ChainStep(
            index=k, template=t.name, params=params,
            spec_text=t.spec_text(params), input_state=list(state),
            expected_state=list(expected), op_count=t.op_count(params)))
        state = expected
    return ChainTask(seed=seed, n_steps=n_steps,
                     generator_version=generator_version,
                     initial_state=initial_state, steps=steps)


# --------------------------------------------------------------- grading
def grade_step(step: ChainStep, input_actual, submitted):
    """Grade one step submission.

    input_actual: the state the agent actually received (list or None
    if what was in the state file was not a valid state).
    submitted: the parsed content of the state file after the attempt
    (any JSON value, or None if unparseable).

    Returns (state_correct, local_correct, error_class, message).
    error_class is None on success; otherwise one of "state"
    (unparseable / not a list), "contract" (wrong length), "type"
    (non-integer or out-of-range elements), "assertion" (values differ
    from the ground-truth state). The message is the external check
    output: it names the first discrepancy, like a check script's
    assertion tail.
    """
    if not isinstance(submitted, list):
        return (False, False, "state",
                "state check failed: the submitted state is not a "
                "JSON list")
    if len(submitted) != STATE_LEN:
        return (False, False, "contract",
                f"state check failed: expected a list of {STATE_LEN} "
                f"values, got {len(submitted)}")
    if not all(isinstance(v, int) and not isinstance(v, bool)
               for v in submitted):
        return (False, False, "type",
                "state check failed: every value must be an integer")
    if not all(0 <= v < MOD for v in submitted):
        return (False, False, "type",
                "state check failed: every value must be in 0..999")
    expected_from_actual = step.apply(input_actual)
    local_correct = (expected_from_actual is not None
                     and list(submitted) == expected_from_actual)
    if list(submitted) == step.expected_state:
        # state correct; local stays as computed -- an agent that
        # re-derives the right state from a wrong input is state-
        # correct but not locally correct, and that distinction is
        # exactly what this grading exists to keep visible.
        return (True, local_correct, None, "state check passed")
    for i, (got, want) in enumerate(zip(submitted, step.expected_state)):
        if got != want:
            return (False, local_correct, "assertion",
                    f"state check failed: position {i} should be "
                    f"{want}, got {got}")
    raise AssertionError("unreachable: unequal lists with no mismatch")


def probe_visibility(step: ChainStep) -> float:
    """Calibration proxy: the fraction of this step's check probes
    failed by the canonical single-defect mutant of the step, applied
    to the step's ground-truth input state. Probes are one per index
    plus a length probe (PROBE_COUNT total)."""
    mutant = step.mutant_apply(step.input_state)
    failed = sum(1 for a, b in zip(mutant, step.expected_state) if a != b)
    if len(mutant) != len(step.expected_state):
        failed += 1
    return failed / PROBE_COUNT


def corrupt_state(state, rng: random.Random):
    """A plausible-but-wrong submission for the fake backend: the kind
    of near-miss a real agent makes (identity, an off-by-one on one
    element, a swapped pair, a duplicated tail element)."""
    if not is_valid_state(state):
        return [rng.randrange(MOD) for _ in range(STATE_LEN)]
    choice = rng.randrange(4)
    out = list(state)
    if choice == 0:
        return out                                   # no change at all
    if choice == 1:
        out[0] = (out[0] + 1) % MOD                  # off by one
        return out
    if choice == 2:
        out[0], out[1] = out[1], out[0]              # wrong order
        return out
    out[-1] = out[-2]                                # duplicated element
    return out
