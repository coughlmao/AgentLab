"""The Phase A-3 difficulty ladder: standalone transformation steps
at graded generator settings.

Phase A-2 showed that the chained family's base step distribution
sits below claude-haiku-5-5's difficulty threshold (0 first-attempt
failures in 870 steps), so the error-accumulation mechanism was
unidentifiable in that regime. Phase A-3 turns difficulty into a
measured, controlled variable: this module generates standalone
steps at an ordered ladder of settings, and teamscale.model_calibrate
probes the model's first-attempt failure rate at each setting to find
the ones that land in a requested band (see
experiments/model-calibrated-generation/DESIGN.md).

A ladder setting combines the design's three levers:

- composite steps: k transformations applied in sequence per step;
- state size: 12 -> 32 -> 64 elements;
- interacting parts: at the "inter" rungs the transformation pool
  also contains state-dependent transformations, whose effect is a
  function of the current state (rotate by an amount read from the
  state, scale by a factor read from the state, ...), so a step
  cannot be executed by pattern alone.

Rung 0 is the A-2 base setting: one plain transformation drawn iid
from the chained family's template set, over a warmed length-12
state -- the distribution whose failure rate A-2 measured at ~0.
Plain transformations ARE the chained family's templates
(teamscale.chains.TEMPLATES), reused via their length-generic form,
so the floor is the same distribution by construction, not by
imitation.

Determinism (ENGINEERING.md, invariant 1): a step is a pure function
of (seed, setting, generator_version). The ladder itself -- the
settings and both template pools -- is part of the versioned
generator definition: changing any of it increments
GENERATOR_VERSION. Every random draw flows from one RNG seeded by a
SHA-256 digest of those three values.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field

from . import chains as chains_mod

GENERATOR_VERSION = "ladder-gen-1"
MOD = chains_mod.MOD            # state values are integers in 0..999
WARMUP_OPS = 8                  # same warmup as the chained family


# ------------------------------------------------- interacting templates
# Each: name, apply(state) -> new state, spec_text(state_len) -> str.
# Applies are pure, length-preserving, and computed simultaneously
# from the input state (decisions never read partially-updated
# values). Parameters come from the state itself; that is the point.

def _rotate_by_first_apply(state):
    n = len(state)
    r = state[0] % (n - 1) + 1
    return state[r:] + state[:r]


def _add_first_apply(state):
    return [(v + state[0]) % MOD for v in state]


def _neighbor_parity_apply(state):
    n = len(state)
    out = []
    for i, v in enumerate(state):
        if state[(i + 1) % n] % 2 == 0:
            out.append((v * 3) % MOD)
        else:
            out.append((v + 29) % MOD)
    return out


def _swap_min_max_apply(state):
    out = list(state)
    i_min = min(range(len(state)), key=lambda i: state[i])
    i_max = max(range(len(state)), key=lambda i: state[i])
    out[i_min], out[i_max] = out[i_max], out[i_min]
    return out


def _scale_by_last_apply(state):
    m = state[-1] % 5 + 2
    return [(v * m) % MOD for v in state]


@dataclass(frozen=True)
class _InterTemplate:
    name: str
    apply: object
    spec_text: object


INTERACTING_TEMPLATES = [
    _InterTemplate(
        "rotate_by_first", _rotate_by_first_apply,
        lambda n: ("Rotate the state left by r positions, where r is "
                   "computed from the state itself: r = (the first "
                   f"value of the state mod {n - 1}) + 1. The first r "
                   "values move to the end, keeping their relative "
                   "order.")),
    _InterTemplate(
        "add_first", _add_first_apply,
        lambda n: ("Add the first value of the state to every value "
                   "in the state, mod 1000. (The first value therefore "
                   "doubles, mod 1000.)")),
    _InterTemplate(
        "neighbor_parity", _neighbor_parity_apply,
        lambda n: ("For each position i, look at the NEXT value in "
                   "the state as given (position i+1; the last "
                   "position's next is the first). If that next value "
                   "is even, replace the value at i with (value * 3) "
                   "mod 1000; if it is odd, replace it with "
                   "(value + 29) mod 1000. All decisions use the "
                   "state as given, not partially updated values.")),
    _InterTemplate(
        "swap_min_max", _swap_min_max_apply,
        lambda n: ("Swap the smallest value in the state with the "
                   "largest value. If the smallest (or largest) value "
                   "occurs more than once, use its first occurrence. "
                   "Every other position is unchanged.")),
    _InterTemplate(
        "scale_by_last", _scale_by_last_apply,
        lambda n: ("Multiply every value in the state by m, mod "
                   "1000, where m is computed from the state itself: "
                   "m = (the last value of the state mod 5) + 2.")),
]
INTERACTING_NAMES = [t.name for t in INTERACTING_TEMPLATES]
_INTER_BY_NAME = {t.name: t for t in INTERACTING_TEMPLATES}
_PLAIN_BY_NAME = {t.name: t for t in chains_mod.TEMPLATES}


# ------------------------------------------------------------------ ladder
@dataclass(frozen=True)
class LadderSetting:
    name: str
    state_len: int
    composite_k: int        # transformations applied per step
    interacting: bool       # pool includes state-dependent templates

    def to_dict(self) -> dict:
        return {"name": self.name, "state_len": self.state_len,
                "composite_k": self.composite_k,
                "interacting": self.interacting}


LADDER = [
    LadderSetting("base-s12-k1", 12, 1, False),    # the A-2 floor
    LadderSetting("plain-s12-k2", 12, 2, False),
    LadderSetting("plain-s12-k3", 12, 3, False),
    LadderSetting("plain-s12-k4", 12, 4, False),
    LadderSetting("plain-s32-k2", 32, 2, False),
    LadderSetting("plain-s32-k4", 32, 4, False),
    LadderSetting("inter-s32-k2", 32, 2, True),
    LadderSetting("inter-s32-k4", 32, 4, True),
    LadderSetting("inter-s64-k4", 64, 4, True),
    LadderSetting("inter-s64-k6", 64, 6, True),
]
LADDER_NAMES = [s.name for s in LADDER]
_BY_SETTING = {s.name: s for s in LADDER}


def setting_at(index: int) -> LadderSetting:
    return LADDER[index]


def setting_index(name: str) -> int:
    return LADDER_NAMES.index(name)


# -------------------------------------------------------------------- step
@dataclass
class ProbeStep:
    seed: int
    setting: str
    generator_version: str
    state_len: int
    subs: list = field(default_factory=list)   # [{kind, template, params}]
    spec_text: str = ""
    input_state: list = field(default_factory=list)
    expected_state: list = field(default_factory=list)
    op_count: int = 0

    def apply_to(self, state):
        """This step's full transformation sequence applied to an
        arbitrary state of the right length (the local reference the
        fake backend's scripted repair uses). None if the state is
        not a valid state for this step."""
        if not _is_valid(state, self.state_len):
            return None
        out = list(state)
        for sub in self.subs:
            out = _apply_sub(sub, out)
        return out

    @property
    def subtask_id(self) -> str:
        return f"probe-{self.setting}-s{self.seed}"

    def to_dict(self) -> dict:
        return {"seed": self.seed, "setting": self.setting,
                "generator_version": self.generator_version,
                "state_len": self.state_len, "subs": self.subs,
                "spec_text": self.spec_text,
                "input_state": self.input_state,
                "expected_state": self.expected_state,
                "op_count": self.op_count}

    def canonical_json(self) -> str:
        """Byte-stable serialization: the determinism contract."""
        return json.dumps(self.to_dict(), sort_keys=True,
                          separators=(",", ":"))


def _is_valid(state, n) -> bool:
    return (isinstance(state, list) and len(state) == n
            and all(isinstance(v, int) and not isinstance(v, bool)
                    and 0 <= v < MOD for v in state))


def _apply_sub(sub: dict, state: list) -> list:
    if sub["kind"] == "plain":
        return _PLAIN_BY_NAME[sub["template"]].apply(
            list(state), sub["params"])
    return _INTER_BY_NAME[sub["template"]].apply(list(state))


def _sub_spec_text(sub: dict, n: int) -> str:
    if sub["kind"] == "plain":
        return _PLAIN_BY_NAME[sub["template"]].spec_text(sub["params"])
    return _INTER_BY_NAME[sub["template"]].spec_text(n)


def _sub_op_count(sub: dict, n: int) -> int:
    if sub["kind"] == "interacting":
        return 2 if sub["template"] == "swap_min_max" else n
    name = sub["template"]
    if name == "swap_pairs":
        return 4
    if name == "sort_window":
        p = sub["params"]
        return p["b"] - p["a"] + 1
    return n


def _seed_rng(seed: int, setting_name: str,
              generator_version: str) -> random.Random:
    digest = hashlib.sha256(
        f"ladder|{generator_version}|{setting_name}|{seed}"
        .encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def make_probe_step(seed: int, setting_name: str,
                    generator_version: str = GENERATOR_VERSION
                    ) -> ProbeStep:
    """Generate one standalone probe step: a pure function of
    (seed, setting, generator_version)."""
    if generator_version != GENERATOR_VERSION:
        raise ValueError(
            f"unknown generator_version {generator_version!r}: "
            f"this code implements {GENERATOR_VERSION!r}")
    if setting_name not in _BY_SETTING:
        raise ValueError(f"unknown ladder setting {setting_name!r}: "
                         f"use one of {LADDER_NAMES}")
    setting = _BY_SETTING[setting_name]
    n = setting.state_len
    rng = _seed_rng(seed, setting_name, generator_version)
    state = [rng.randrange(MOD) for _ in range(n)]
    plain = chains_mod.TEMPLATES
    for _ in range(WARMUP_OPS):   # warmup: not part of the step
        t = plain[rng.randrange(len(plain))]
        state = t.apply(state, t.make_params(rng, n))
    input_state = list(state)
    pool = ([("plain", t.name) for t in plain]
            + ([("interacting", t.name) for t in INTERACTING_TEMPLATES]
               if setting.interacting else []))
    subs = []
    for _ in range(setting.composite_k):
        kind, name = pool[rng.randrange(len(pool))]
        if kind == "plain":
            params = _PLAIN_BY_NAME[name].make_params(rng, n)
            subs.append({"kind": "plain", "template": name,
                         "params": params})
        else:
            subs.append({"kind": "interacting", "template": name,
                         "params": {}})
    out = list(input_state)
    for sub in subs:
        out = _apply_sub(sub, out)
    if setting.composite_k == 1:
        spec_text = _sub_spec_text(subs[0], n)
    else:
        lines = [f"Apply these {setting.composite_k} transformations "
                 "to the state IN ORDER (each one operates on the "
                 "result of the previous one):"]
        for i, sub in enumerate(subs, 1):
            lines.append(f"{i}. {_sub_spec_text(sub, n)}")
        spec_text = "\n".join(lines)
    return ProbeStep(
        seed=seed, setting=setting_name,
        generator_version=generator_version, state_len=n, subs=subs,
        spec_text=spec_text, input_state=input_state,
        expected_state=out,
        op_count=sum(_sub_op_count(s, n) for s in subs))


# ----------------------------------------------------------------- grading
def grade_probe_step(step: ProbeStep, submitted):
    """Grade one probe submission against the generator's ground
    truth, with the same checks and error classes as the chained
    family's grade_step (teamscale.chains): (correct, error_class,
    message). Correct means the submitted state equals the expected
    state exactly."""
    n = step.state_len
    if not isinstance(submitted, list):
        return (False, "state",
                "state check failed: the submitted state is not a "
                "JSON list")
    if len(submitted) != n:
        return (False, "contract",
                f"state check failed: expected a list of {n} "
                f"values, got {len(submitted)}")
    if not all(isinstance(v, int) and not isinstance(v, bool)
               for v in submitted):
        return (False, "type",
                "state check failed: every value must be an integer")
    if not all(0 <= v < MOD for v in submitted):
        return (False, "type",
                "state check failed: every value must be in 0..999")
    if list(submitted) == step.expected_state:
        return (True, None, "state check passed")
    for i, (got, want) in enumerate(zip(submitted,
                                        step.expected_state)):
        if got != want:
            return (False, "assertion",
                    f"state check failed: position {i} should be "
                    f"{want}, got {got}")
    raise AssertionError("unreachable: unequal lists with no mismatch")


def corrupt_state(state, rng: random.Random):
    """Length-generic version of the chained family's plausible-but-
    wrong submission, for the fake backend: identity, an off-by-one
    on one element, a swapped pair, a duplicated tail element."""
    if not _is_valid(state, len(state) if isinstance(state, list)
                     else -1):
        n = len(state) if isinstance(state, list) and state else 12
        return [rng.randrange(MOD) for _ in range(n)]
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
