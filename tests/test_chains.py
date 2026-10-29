"""Phase A-2 chained family: generator determinism (the (seed,
generator_version) contract, across processes), grading semantics
(state vs local correctness, error classes), and chain structure.
Everything offline.
"""
import hashlib
import subprocess
import sys

import pytest

from teamscale import chains

PINNED = {  # (seed, n_steps) -> sha256 of canonical_json, chain-gen-1
    (7, 50): "4c423689b14130baec702b3e6ed293029a5ab5005afb4fa617a572d544873edd",
    (0, 2): "42f00a117c0d1f3d69d8ce34e94e99058d46839149258316cf41d810724e9bbf",
}


def test_chain_structure_links_steps():
    task = chains.make_chain(3, 10)
    assert task.steps[0].input_state == task.initial_state
    for prev, cur in zip(task.steps, task.steps[1:]):
        assert cur.input_state == prev.expected_state
        assert cur.index == prev.index + 1
    for step in task.steps:
        assert step.template in chains.TEMPLATE_NAMES
        assert chains.is_valid_state(step.input_state)
        assert chains.is_valid_state(step.expected_state)
        assert step.apply(step.input_state) == step.expected_state


def test_determinism_in_process_and_pinned_hashes():
    for (seed, n), want in PINNED.items():
        a = chains.make_chain(seed, n).canonical_json()
        b = chains.make_chain(seed, n).canonical_json()
        assert a == b
        assert hashlib.sha256(a.encode()).hexdigest() == want


def test_determinism_across_processes():
    code = ("import hashlib, sys; sys.path.insert(0, '.');"
            "from teamscale import chains;"
            "print(hashlib.sha256(chains.make_chain(7, 50)"
            ".canonical_json().encode()).hexdigest())")
    outs = set()
    for _ in range(2):
        proc = subprocess.run([sys.executable, "-c", code],
                              capture_output=True, text=True, timeout=60,
                              cwd=str(__import__("pathlib").Path(
                                  __file__).resolve().parents[1]))
        assert proc.returncode == 0, proc.stderr
        outs.add(proc.stdout.strip())
    assert outs == {PINNED[(7, 50)]}


def test_unknown_generator_version_rejected():
    with pytest.raises(ValueError, match="generator_version"):
        chains.make_chain(0, 5, generator_version="chain-gen-999")


def test_grading_state_vs_local_and_error_classes():
    task = chains.make_chain(5, 6)
    s1, s2 = task.steps[0], task.steps[1]
    ok = chains.grade_step(s1, s1.input_state, s1.expected_state)
    assert ok[:3] == (True, True, None)
    # corrupt step 1's output: state wrong, and (input was right) local
    # wrong too
    bad = list(s1.expected_state)
    bad[0] = (bad[0] + 1) % 1000
    got = chains.grade_step(s1, s1.input_state, bad)
    assert got[0] is False and got[1] is False
    assert got[2] == "assertion"
    # step 2 receives the corrupted state and applies its transformation
    # correctly: locally correct, state still wrong -- the cascade case
    local_ok = s2.apply(bad)
    got2 = chains.grade_step(s2, bad, local_ok)
    assert got2[0] is False and got2[1] is True
    # error classes
    assert chains.grade_step(s2, bad, "junk")[2] == "state"
    assert chains.grade_step(s2, bad, None)[2] == "state"
    assert chains.grade_step(s2, bad, [1, 2, 3])[2] == "contract"
    assert chains.grade_step(s2, bad, ["x"] * 12)[2] == "type"
    assert chains.grade_step(s2, bad, [True] * 12)[2] == "type"
    assert chains.grade_step(s2, bad, [1000] * 12)[2] == "type"


def test_state_file_round_trip_and_rejects():
    state = [1, 22, 333, 0, 999, 5, 6, 7, 8, 9, 10, 11]
    text = chains.render_state_file(state)
    assert text.startswith("STATE = [")
    assert chains.parse_state_file(text) == state
    assert chains.parse_state_file("STATE = (1, 2)") == (1, 2)
    assert chains.parse_state_file("raise SystemExit") is None
    assert chains.parse_state_file("X = [1]") is None
    assert chains.parse_state_file("STATE = [1,") is None
    # never executes: this would create a file if it ran
    assert chains.parse_state_file(
        "import os; os.system('true')\nSTATE = [1]") == [1]


def test_probe_visibility_bounds():
    task = chains.make_chain(11, 50)
    for step in task.steps:
        v = chains.probe_visibility(step)
        assert 0.0 <= v <= 1.0
