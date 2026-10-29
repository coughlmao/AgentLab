"""Phase A replay + analysis: deterministic replay of an offline fake
experiment, tamper detection, bootstrap CIs, and figure regeneration.
Everything offline. No network.
"""
import csv
import json
from pathlib import Path

import pytest

import teamscale.analyze as an
import teamscale.config_experiment as ce
import teamscale.replay as replay


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("phase-a")
    spec = ce.spec_from_dict({
        "name": "tiny", "question": "when does verification help?",
        "provider": "fake", "difficulty": "standard", "blind": True,
        "sizes": [1, 2], "strategies": ["orchestrated", "verified"],
        "seeds": [0, 1], "output_root": str(tmp / "results"),
        "workers": 2, "fake_competence": 0.5})
    return ce.run_config_experiment(spec, run_id="replay-me")


# ------------------------------------------------------------------- replay
def test_replay_passes_on_a_real_run_directory(run_dir):
    assert replay.replay_run(run_dir) == []
    assert replay.main([str(run_dir)]) == 0


def test_replay_detects_a_tampered_metric(run_dir, tmp_path):
    import shutil
    copy = tmp_path / "tampered"
    shutil.copytree(run_dir, copy)
    metrics = copy / "metrics.csv"
    rows = list(csv.DictReader(open(metrics)))
    rows[0]["fraction_correct"] = "0.1234"  # impossible: fractions are k/8
    with open(metrics, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    diffs = replay.replay_run(copy)
    assert diffs and any("fraction" in d for d in diffs)
    assert replay.main([str(copy)]) == 1


def test_replay_detects_a_tampered_sandbox(run_dir, tmp_path):
    import shutil
    copy = tmp_path / "tampered-sandbox"
    shutil.copytree(run_dir, copy)
    # break every module in every saved sandbox: replayed grading must
    # drop to zero, which cannot match the recorded metrics (at
    # competence 0.5 across 64 subtasks, an all-zero record is not a
    # plausible genuine outcome, and success flags would flip too).
    for sandbox in (copy / "sandboxes").iterdir():
        for module_file in sandbox.glob("*.py"):
            if not module_file.name.startswith("check_"):
                module_file.write_text("raise RuntimeError('tampered')\n")
    diffs = replay.replay_run(copy)
    assert diffs and any("fraction" in d or "success" in d for d in diffs)


# ------------------------------------------------------------------ analyze
def test_analyze_writes_everything_and_is_deterministic(run_dir):
    assert an.main([str(run_dir)]) == 0
    analysis = json.loads((run_dir / "analysis.json").read_text())
    assert analysis["n_runs"] == 8
    assert len(analysis["cells"]) == 4
    for cell in analysis["cells"]:
        assert cell["ci95"][0] <= cell["mean_fraction_correct"] + 1e-9
        assert cell["ci95"][0] <= cell["ci95"][1]
    contrast = analysis["primary_contrast"]
    assert contrast["strategies"] == ["verified", "orchestrated"]
    assert set(contrast["by_size"]) == {"1", "2"}
    assert contrast["overall"]["ci95"][0] <= contrast["overall"]["ci95"][1]

    first_json = (run_dir / "analysis.json").read_bytes()
    first_md = (run_dir / "SUMMARY.md").read_bytes()
    assert an.main([str(run_dir)]) == 0
    assert (run_dir / "analysis.json").read_bytes() == first_json
    assert (run_dir / "SUMMARY.md").read_bytes() == first_md

    for name in ("scaling_curve.svg", "cost_breakdown.svg"):
        svg = (run_dir / "figures" / name).read_text()
        assert svg.startswith("<svg") and "verified" in svg
    md = (run_dir / "SUMMARY.md").read_text()
    assert "when does verification help?" in md
    assert "Primary contrast" in md and "Caveats" in md


def test_analyze_intervals_shrink_with_more_runs(run_dir):
    # sanity of the bootstrap itself: a cell whose runs all agree has a
    # degenerate interval; a mixed cell has a strictly wider one.
    import random
    rng = random.Random(0)
    lo, hi = an.bootstrap_ci([0.5, 0.5, 0.5, 0.5], rng, resamples=2000)
    assert (lo, hi) == (0.5, 0.5)
    lo2, hi2 = an.bootstrap_ci([0.0, 1.0, 0.0, 1.0], rng, resamples=2000)
    assert lo2 < 0.5 < hi2
