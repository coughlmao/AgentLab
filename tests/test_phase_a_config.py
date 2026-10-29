"""Phase A config-driven experiments: the YAML subset parser, the
shipped Phase A config, and the immutable run-directory layout.
Everything offline on the fake backend. No network.
"""
import csv
import hashlib
import json
from pathlib import Path

import pytest

import teamscale.config_experiment as ce
import teamscale.experiment as exp_cli
from teamscale.yaml_subset import YamlSubsetError, parse_yaml

REPO = Path(__file__).resolve().parents[1]
SHIPPED = REPO / "experiments" / "verification-question" / "experiment.yaml"


# ------------------------------------------------------------------- parser
def test_parser_scalars_and_typing():
    doc = parse_yaml(
        "name: demo\n"
        "count: 42\n"
        "ratio: 0.5\n"
        "cap: 15.00\n"
        "blind: true\n"
        "other: false\n"
        "nothing: null\n"
        "plain: hello world\n")
    assert doc == {"name": "demo", "count": 42, "ratio": 0.5, "cap": 15.0,
                   "blind": True, "other": False, "nothing": None,
                   "plain": "hello world"}


def test_parser_lists_flow_block_and_comments():
    doc = parse_yaml(
        "# leading comment\n"
        "sizes: [1, 2, 4, 8]  # trailing comment\n"
        "strategies:\n"
        "  - independent\n"
        "  - verified\n")
    assert doc["sizes"] == [1, 2, 4, 8]
    assert doc["strategies"] == ["independent", "verified"]


def test_parser_nested_maps_and_quoted_strings():
    doc = parse_yaml(
        'question: "When: does it help?"\n'
        "outer:\n"
        "  inner: value\n"
        "  deep:\n"
        "    n: 3\n")
    assert doc["question"] == "When: does it help?"
    assert doc["outer"] == {"inner": "value", "deep": {"n": 3}}


def test_parser_rejects_out_of_subset_constructs():
    with pytest.raises(YamlSubsetError):
        parse_yaml("a: {b: 1}\n")            # flow mapping
    with pytest.raises(YamlSubsetError):
        parse_yaml("a: 1\na: 2\n")           # duplicate key
    with pytest.raises(YamlSubsetError):
        parse_yaml("- name: x\n")            # list of mappings
    with pytest.raises(YamlSubsetError):
        parse_yaml("a: [1, [2]]\n")          # nested flow list


# ----------------------------------------------------------- shipped config
def test_shipped_phase_a_config_parses_to_the_expected_grid():
    spec, text, sha = ce.load_config(SHIPPED)
    assert sha == hashlib.sha256(text.encode()).hexdigest()
    assert spec.name == "verification-question"
    assert "external verification" in spec.question
    assert spec.provider == "anthropic"
    assert spec.model == "claude-haiku-5-5"
    assert spec.difficulty == "hard"
    assert spec.blind is True
    assert spec.strategies == ("independent", "orchestrated", "verified")
    assert spec.sizes == (1, 2, 4, 8)
    assert spec.seeds == (0, 1, 2, 3, 4)
    assert spec.n_runs == 60
    assert spec.spend_cap_usd == 15.00
    assert spec.per_run_token_cap == 400_000


def test_config_validation_rejects_bad_values():
    with pytest.raises(ValueError, match="unknown provider"):
        ce.spec_from_dict({"name": "x", "provider": "carrier-pigeon"})
    with pytest.raises(ValueError, match="unknown difficulty"):
        ce.spec_from_dict({"name": "x", "difficulty": "nightmare"})
    with pytest.raises(ValueError, match="unknown strategies"):
        ce.spec_from_dict({"name": "x", "strategies": ["mob"]})
    with pytest.raises(ValueError, match="requires a string 'name'"):
        ce.spec_from_dict({"provider": "fake"})


# -------------------------------------------------------------- run layout
def _tiny_spec(tmp_path, **overrides):
    data = {"name": "tiny", "question": "does it run?",
            "provider": "fake", "difficulty": "standard", "blind": True,
            "sizes": [1], "strategies": ["verified"], "seeds": [0, 1],
            "output_root": str(tmp_path / "results"), "workers": 2,
            "fake_competence": 0.5}
    data.update(overrides)
    return ce.spec_from_dict(data)


def test_run_directory_layout_manifest_and_metrics(tmp_path):
    spec = _tiny_spec(tmp_path)
    run_dir = ce.run_config_experiment(spec, run_id="fixed-run-id")
    assert run_dir == tmp_path / "results" / "tiny" / "fixed-run-id"

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["experiment"] == "tiny"
    assert manifest["config"] == spec.raw          # config verbatim
    assert manifest["provider"] == "fake"
    assert manifest["blind"] is True
    assert manifest["git"]["commit"]                # a hash or "unknown"
    assert manifest["pricing"]["model"] == "fake"
    expected_sha = hashlib.sha256(
        json.dumps(spec.raw, sort_keys=True).encode()).hexdigest()
    assert manifest["config_sha256"] == expected_sha

    rows = list(csv.DictReader(open(run_dir / "metrics.csv")))
    assert len(rows) == 2                            # 1 size x 1 strat x 2 seeds
    assert {r["strategy"] for r in rows} == {"verified"}
    assert all(0.0 <= float(r["fraction_correct"]) <= 1.0 for r in rows)

    traces = [json.loads(l) for l in
              (run_dir / "traces.jsonl").read_text().splitlines() if l.strip()]
    assert sum(1 for e in traces if e["event"] == "run_start") == 2
    assert (run_dir / "figures").is_dir()
    assert (run_dir / "sandboxes").is_dir()
    assert (run_dir / "ledger_fake.json").exists()

    summary = json.loads((run_dir / "summary.json").read_text())
    acct = summary["cost_accounting"]
    assert acct["runs_completed"] == 2
    assert acct["total_cost_usd"] == 0.0             # fake backend is free
    assert acct["by_strategy"]["verified"]["subtasks_correct"] == \
        sum(int(float(r["fraction_correct"]) * 8) for r in rows)
    md = (run_dir / "SUMMARY.md").read_text()
    assert "does it run?" in md and "Cost accounting" in md


def test_run_directory_refuses_overwrite(tmp_path):
    spec = _tiny_spec(tmp_path)
    ce.run_config_experiment(spec, run_id="same-id")
    with pytest.raises(FileExistsError, match="refusing to write"):
        ce.run_config_experiment(spec, run_id="same-id")


def test_experiment_cli_dispatches_yaml(tmp_path):
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(
        "name: tiny\n"
        "provider: fake\n"
        "sizes: [1]\n"
        "strategies: [independent]\n"
        "seeds: [0]\n"
        "workers: 1\n"
        f"output_root: {tmp_path / 'results'}\n")
    assert exp_cli.main([str(cfg), "--run-id", "via-cli"]) == 0
    run_dir = tmp_path / "results" / "tiny" / "via-cli"
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "metrics.csv").exists()
