"""Phase A-3 ladder: generator determinism (the (seed, setting,
generator_version) contract, across processes), the A-2 floor rung,
grading semantics, and the interacting templates. Everything
offline."""
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from teamscale import ladder

PINNED = {  # (seed, setting) -> sha256 of canonical_json, ladder-gen-1
    (7, "base-s12-k1"):
        "942c42d74dcc235386fee21b34df7e4a1bda50182f7a46e976798ab701dd35eb",
    (3, "inter-s32-k4"):
        "fb7aa2625e9a5bdc8660e6fdac211da0c57749feb687ff97fb76f59e05d8bbf9",
    (11, "inter-s64-k6"):
        "1b5520ca67efc45dda92c594b2eae6b8c7e5b39eac0942494c80242ac2f4d120",
}


def test_ladder_shape_and_floor():
    assert len(ladder.LADDER) == 10
    floor = ladder.LADDER[0]
    # rung 0 is the A-2 base setting: length 12, one plain
    # transformation, no interacting pool
    assert (floor.state_len, floor.composite_k, floor.interacting) \
        == (12, 1, False)
    # the rungs, in order (checked rung-by-rung)
    assert [s.composite_k for s in ladder.LADDER] == \
        [1, 2, 3, 4, 2, 4, 2, 4, 4, 6]
    assert [s.state_len for s in ladder.LADDER] == \
        [12, 12, 12, 12, 32, 32, 32, 32, 64, 64]
    assert [s.interacting for s in ladder.LADDER] == \
        [False] * 6 + [True] * 4


def test_determinism_in_process_and_pinned_hashes():
    for (seed, name), want in PINNED.items():
        a = ladder.make_probe_step(seed, name).canonical_json()
        b = ladder.make_probe_step(seed, name).canonical_json()
        assert a == b
        assert hashlib.sha256(a.encode()).hexdigest() == want


def test_determinism_across_processes():
    code = ("import hashlib, sys; sys.path.insert(0, '.');"
            "from teamscale import ladder;"
            "print(hashlib.sha256(ladder.make_probe_step"
            "(3, 'inter-s32-k4').canonical_json().encode())"
            ".hexdigest())")
    outs = set()
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True,
            text=True, timeout=60,
            cwd=str(Path(__file__).resolve().parents[1]))
        assert proc.returncode == 0, proc.stderr
        outs.add(proc.stdout.strip())
    assert outs == {PINNED[(3, "inter-s32-k4")]}


def test_unknown_setting_and_version_rejected():
    with pytest.raises(ValueError, match="ladder setting"):
        ladder.make_probe_step(0, "nope")
    with pytest.raises(ValueError, match="generator_version"):
        ladder.make_probe_step(0, "base-s12-k1",
                               generator_version="ladder-gen-999")


def test_every_rung_applies_and_grades():
    for name in ladder.LADDER_NAMES:
        for seed in (0, 1, 2):
            step = ladder.make_probe_step(seed, name)
            assert len(step.input_state) == step.state_len
            assert len(step.expected_state) == step.state_len
            assert step.apply_to(step.input_state) \
                == step.expected_state
            ok, ec, _ = ladder.grade_probe_step(
                step, step.expected_state)
            assert ok and ec is None
            assert step.apply_to([0] * (step.state_len + 1)) is None


def test_grading_error_classes():
    step = ladder.make_probe_step(5, "plain-s12-k2")
    assert ladder.grade_probe_step(step, "nope")[:2] \
        == (False, "state")
    assert ladder.grade_probe_step(step, [1, 2])[:2] \
        == (False, "contract")
    bad_type = list(step.expected_state)
    bad_type[0] = "x"
    assert ladder.grade_probe_step(step, bad_type)[:2] \
        == (False, "type")
    bad_range = list(step.expected_state)
    bad_range[0] = 1000
    assert ladder.grade_probe_step(step, bad_range)[:2] \
        == (False, "type")
    bad_val = list(step.expected_state)
    bad_val[0] = (bad_val[0] + 1) % 1000
    assert ladder.grade_probe_step(step, bad_val)[:2] \
        == (False, "assertion")


def test_interacting_templates_semantics():
    by_name = {t.name: t for t in ladder.INTERACTING_TEMPLATES}
    state = [10, 3, 7, 42]
    # rotate_by_first: r = 10 % 3 + 1 = 2
    assert by_name["rotate_by_first"].apply(state) == [7, 42, 10, 3]
    # add_first: +10 everywhere
    assert by_name["add_first"].apply(state) == [20, 13, 17, 52]
    # scale_by_last: m = 42 % 5 + 2 = 4
    assert by_name["scale_by_last"].apply(state) == [40, 12, 28, 168]
    # swap_min_max: min 3 (idx 1) <-> max 42 (idx 3)
    assert by_name["swap_min_max"].apply(state) == [10, 42, 7, 3]
    # neighbor_parity reads the ORIGINAL next values: next of 10 is
    # 3 (odd) -> +29; next of 3 is 7 (odd) -> +29; next of 7 is 42
    # (even) -> *3; next of 42 is 10 (even) -> *3
    assert by_name["neighbor_parity"].apply(state) == [39, 32, 21, 126]


def test_plain_floor_uses_chain_templates():
    from teamscale import chains
    for seed in range(5):
        step = ladder.make_probe_step(seed, "base-s12-k1")
        assert len(step.subs) == 1
        assert step.subs[0]["kind"] == "plain"
        assert step.subs[0]["template"] in chains.TEMPLATE_NAMES
