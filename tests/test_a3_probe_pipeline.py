"""Phase A-3 probe harness + full fake bundle: probe records with a
scripted backend, the transport-error outcome taxonomy, and an
end-to-end fake run of the calibration driver whose bundle replays
clean. Everything offline."""
import json
from pathlib import Path

from teamscale.config_experiment import ExperimentSpec, \
    run_config_experiment
from teamscale.fake_backend import FakeLLMBackend
from teamscale.llm_backend import FixResult
from teamscale.probe import run_probe_batch
from teamscale.replay import replay_run


def test_probe_batch_competent_fake_all_correct(tmp_path):
    backend = FakeLLMBackend(competence=1.0, seed=1, blind=True)
    records = run_probe_batch(
        backend=backend, setting_name="plain-s12-k3",
        seeds=list(range(8)), label="t10", target=0.10,
        phase="search", out_dir=tmp_path, model="fake", workers=2)
    assert len(records) == 8
    assert all(r["outcome"] == "correct" for r in records)
    assert all(r["tokens_in"] > 0 and r["model_calls"] > 0
               for r in records)
    for r in records:
        sandbox = tmp_path / "sandboxes" / r["probe_id"]
        assert (sandbox / "state.py").exists()
        trace = tmp_path / "traces" / f"{r['probe_id']}.jsonl"
        assert trace.exists()


def test_probe_batch_incompetent_fake_mostly_incorrect(tmp_path):
    backend = FakeLLMBackend(competence=0.0, seed=1, blind=True)
    records = run_probe_batch(
        backend=backend, setting_name="plain-s12-k3",
        seeds=list(range(8)), label="t10", target=0.10,
        phase="search", out_dir=tmp_path, model="fake", workers=2)
    failures = sum(1 for r in records if r["outcome"] == "incorrect")
    assert failures >= 6
    assert all(r["error_class"] for r in records
               if r["outcome"] == "incorrect")


class _TransportFailBackend:
    name = "transportfail"

    def fix_step(self, *, prompt, system_prompt, sandbox,
                 token_budget=25_000, agent_id=0, subtask="step"):
        result = FixResult()
        result.error = "API returned HTTP 500: simulated outage"
        return result


def test_probe_transport_error_is_not_a_model_failure(tmp_path):
    records = run_probe_batch(
        backend=_TransportFailBackend(), setting_name="base-s12-k1",
        seeds=[0, 1], label="t5", target=0.05, phase="search",
        out_dir=tmp_path, model="fake", workers=1)
    assert all(r["outcome"] == "error" and r["correct"] is None
               for r in records)


def _fake_spec() -> ExperimentSpec:
    return ExperimentSpec(
        name="model-calibrated-generation-test",
        question="can calibration hit a wide synthetic band?",
        provider="fake", family="model_calibrated_generation",
        blind=True, band_targets=(0.5,), band_tolerances=(0.3,),
        batch_size=10, max_settings=6, validation_n=20,
        spend_cap_usd=10.0, workers=2, output_root="results",
        fake_competence=0.5, caveats=("test caveat",),
        raw={"name": "model-calibrated-generation-test"})


def test_fake_bundle_end_to_end_and_replay(tmp_path):
    spec = _fake_spec()
    run_dir = run_config_experiment(spec, fake=True, base_dir=tmp_path)
    for name in ("manifest.json", "probes.csv", "metrics.csv",
                 "traces.jsonl", "calibration.json", "summary.json",
                 "SUMMARY.md", "hashes.json", "ledger_fake.json"):
        assert (run_dir / name).exists(), name
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["family"] == "model_calibrated_generation"
    assert manifest["generator_version"] == "ladder-gen-1"
    assert manifest["calibration_version"] == "model-calibration-1"
    assert manifest["prompt_hash"]
    calibration = json.loads((run_dir / "calibration.json").read_text())
    assert calibration["version"] == "model-calibration-1"
    (band,) = calibration["bands"]
    # fake competence 0.5 -> ~50% failures at every rung; the wide
    # synthetic band [0.2, 0.8] must converge on the first probe
    assert band["status"] == "converged"
    assert band["calibration_version"].startswith("band-t50@ladder-")
    assert band["efficiency"]["probes"] == 10
    assert band["validation"]["status"] == "completed"
    assert band["validation"]["n"] == 20
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["total_probes"] == 30
    assert summary["total_cost_usd"] == 0.0
    # caveats sit at the top of SUMMARY.md
    summary_md = (run_dir / "SUMMARY.md").read_text()
    assert summary_md.index("## Caveats") \
        < summary_md.index("## Bands")
    # the whole bundle replays exactly, offline
    assert replay_run(run_dir) == []
