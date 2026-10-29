"""Phase A-2 pipeline, end to end on the fake backend: shipped config,
bundle layout (steps table, recovery events, calibration, seal),
provenance fields, the calibration release-gate enforcement, replay
(incl. hash tamper detection), and the generic analysis producing
survival/hazard/across outputs. Everything offline.
"""
import csv
import hashlib
import json
from pathlib import Path

import pytest

import teamscale.analyze as an
import teamscale.calibrate as cal
import teamscale.config_experiment as ce
import teamscale.replay as replay

REPO = Path(__file__).resolve().parents[1]
SHIPPED = REPO / "experiments" / "error-accumulation" / "experiment.yaml"


@pytest.fixture(scope="module")
def calibration_file(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("a2-cal")
    path = tmp / "calibration.json"
    report = cal.run_calibration(n_chains=30, n_steps=50)
    assert report["gate"] == "pass"
    path.write_text(json.dumps(report, indent=1))
    return path


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory, calibration_file):
    tmp = tmp_path_factory.mktemp("a2-run")
    spec = ce.spec_from_dict({
        "name": "error-accumulation", "question": "does it grow?",
        "provider": "fake", "family": "error_accumulation",
        "difficulty": "hard", "blind": True,
        "horizons": [2, 5], "strategies": ["orchestrated", "verified"],
        "seeds": [0, 1], "team_size": 4,
        "output_root": str(tmp / "results"), "workers": 2,
        "fake_competence": 0.7,
        "analysis": {
            "series": "strategy", "group_by": "horizon",
            "value": "final_state_correct",
            "contrast": {"positive": "verified",
                         "negative": "orchestrated",
                         "across": [2, 5]},
            "steps": {"file": "steps.csv", "index": "step_index",
                      "value": "state_correct"},
            "caveats_position": "top"},
        "caveats": ["test caveat: fake backend only"]})
    return ce.run_config_experiment(
        spec, run_id="a2-tiny", calibration_path=str(calibration_file))


def test_shipped_a2_config():
    spec, _text, _sha = ce.load_config(SHIPPED)
    assert spec.family == "error_accumulation"
    assert spec.horizons == (2, 5, 10, 20, 50)
    assert spec.strategies == ("orchestrated", "verified")
    assert spec.seeds == (0, 1, 2, 3, 4)
    assert spec.team_size == 4 and spec.blind is True
    assert spec.n_runs == 50 and spec.spend_cap_usd == 15.00
    decl = spec.analysis_declaration
    assert decl["group_by"] == "horizon"
    assert decl["contrast"]["across"] == [2, 50]
    assert len(spec.caveats) >= 4


def test_bundle_layout_and_provenance(run_dir, calibration_file):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["family"] == "error_accumulation"
    assert manifest["generator_version"] == "chain-gen-1"
    assert manifest["calibration_version"] == cal.CALIBRATION_VERSION
    assert manifest["replay_version"] == replay.REPLAY_VERSION
    assert manifest["analysis_version"] == an.ANALYSIS_VERSION
    assert len(manifest["prompt_hash"]) == 64
    assert manifest["calibration"]["gate"] == "pass"
    assert manifest["calibration"]["sha256"] == hashlib.sha256(
        calibration_file.read_bytes()).hexdigest()
    assert (run_dir / "calibration.json").read_bytes() == \
        calibration_file.read_bytes()
    assert (run_dir / "hashes.json").exists()

    rows = list(csv.DictReader(open(run_dir / "metrics.csv")))
    assert len(rows) == 8                      # 2 strat x 2 hor x 2 seeds
    assert {r["horizon"] for r in rows} == {"2", "5"}
    steps = list(csv.DictReader(open(run_dir / "steps.csv")))
    assert len(steps) == 2 * (2 + 5) * 2       # per run: horizon rows
    assert {s["state_correct"] for s in steps} <= {"0", "1"}
    events = [json.loads(l) for l in
              (run_dir / "recovery_events.jsonl").read_text().splitlines()
              if l.strip()]
    assert events and all(e["strategy"] == "verified" for e in events)
    assert len(events) == 2 * (2 + 5)          # one check per verified step
    for e in events:
        assert e["check_outcome"] in ("pass", "fail")
        assert e["would_cascade"] is (e["step_index"] < e["horizon"])
        if e["check_outcome"] == "pass":
            assert e["error_class"] is None
            assert e["retry_outcome"] == "not_needed"
            assert e["chain_restored"] is True


def test_replay_and_analysis(run_dir):
    assert replay.replay_run(run_dir) == []
    assert replay.main([str(run_dir)]) == 0
    analysis = an.analyze_run(run_dir)
    assert analysis["n_runs"] == 8
    steps = analysis["steps_analysis"]
    assert steps is not None
    for group in ("2", "5"):
        for series in ("orchestrated", "verified"):
            s = steps["survival"][group][series]
            assert len(s["survival"]) == int(group)
            assert all(0.0 <= v <= 1.0 for v in s["survival"])
            assert s["survival"] == sorted(s["survival"], reverse=True)
            h = steps["hazard"][group][series]
            assert len(h["hazard"]) == int(group)
    contrast = analysis["primary_contrast"]
    assert contrast["strategies"] == ["verified", "orchestrated"]
    assert contrast["across"]["groups"] == [2, 5]
    md = (run_dir / "SUMMARY.md").read_text()
    # caveats at the TOP: the caveat block precedes the Design section
    assert md.index("test caveat: fake backend only") < md.index("## Design")
    for name in ("scaling_curve.svg", "cost_breakdown.svg",
                 "survival_2.svg", "hazard_50.svg"):
        path = run_dir / "figures" / name
        if name != "hazard_50.svg":
            assert path.exists() and path.read_text().startswith("<svg")


def test_replay_detects_tampered_steps(run_dir, tmp_path):
    import shutil
    copy = tmp_path / "tampered-a2"
    shutil.copytree(run_dir, copy)
    steps = copy / "steps.csv"
    rows = list(csv.DictReader(open(steps)))
    rows[0]["state_correct"] = "1" if rows[0]["state_correct"] == "0" \
        else "0"
    with open(steps, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    diffs = replay.replay_run(copy)
    assert diffs and any("hash mismatch" in d or "steps.csv" in d
                         for d in diffs)


def test_curve_svg_handles_all_zero_hazard():
    # a group where no run ever fails has hazard 0 everywhere; the
    # figure must not divide by a zero ymax (found by the fake dry run
    # of the shipped config, before the release freeze).
    per_series = {"verified": {"steps": [1, 2], "hazard": [0.0, 0.0],
                               "ci95": [[0.0, 0.0], [0.0, 0.0]]}}
    svg = an._curve_svg(per_series, "hazard", 2,
                        {"group_by": "horizon"})
    assert svg.startswith("<svg")


def test_live_run_requires_passing_calibration(tmp_path):
    spec = ce.spec_from_dict({
        "name": "error-accumulation", "provider": "anthropic",
        "model": "claude-haiku-5-5", "family": "error_accumulation",
        "difficulty": "hard", "blind": True, "horizons": [2],
        "strategies": ["verified"], "seeds": [0],
        "output_root": str(tmp_path / "results")})
    with pytest.raises(ValueError, match="calibration"):
        ce.run_config_experiment(spec, run_id="must-not-run")
    assert not (tmp_path / "results").exists()


def test_generic_analysis_on_unfamiliar_columns(tmp_path):
    # The purity contract: columns and names the analysis has never
    # seen, declared in the manifest, still produce contrasts and
    # survival -- no experiment vocabulary inside the machinery.
    run = tmp_path / "synth"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({
        "experiment": "synth", "question": "q?", "run_id": "s1",
        "provider": "fake", "model": "fake",
        "analysis": {"series": "arm", "group_by": "depth",
                     "value": "score",
                     "contrast": {"positive": "treated",
                                  "negative": "control",
                                  "across": [1, 2]},
                     "steps": {"file": "steps.csv",
                               "index": "k", "value": "ok"}}}))
    with open(run / "metrics.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "depth", "seed", "score", "cost_usd"])
        for depth in (1, 2):
            for arm, base in (("control", 0.4), ("treated", 0.6)):
                for seed in range(4):
                    w.writerow([arm, depth, seed, base, 0.01])
    with open(run / "steps.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run_id", "arm", "depth", "k", "ok"])
        for depth in (1, 2):
            for arm in ("control", "treated"):
                for seed in range(4):
                    rid = f"{arm}-{depth}-{seed}"
                    # treated never fails; control fails at k=2
                    oks = [1, 1 if arm == "treated" else 0]
                    for k, ok in enumerate(oks, start=1):
                        w.writerow([rid, arm, depth, k, ok])
    analysis = an.analyze_run(run)
    assert analysis["primary_contrast"]["overall"]["difference"] == 0.2
    assert analysis["primary_contrast"]["across"]["difference"] == 0.0
    surv = analysis["steps_analysis"]["survival"]["2"]
    assert surv["treated"]["survival"] == [1.0, 1.0]
    assert surv["control"]["survival"] == [1.0, 0.0]
    haz = analysis["steps_analysis"]["hazard"]["2"]
    assert haz["control"]["hazard"] == [0.0, 1.0]
    assert analysis["steps_analysis"]["divergence_step"]["2"] is None or \
        isinstance(analysis["steps_analysis"]["divergence_step"]["2"],
                   int)
