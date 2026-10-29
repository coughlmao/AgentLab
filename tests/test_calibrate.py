"""Calibration gate: the hand-rolled statistics against known values,
and the gate itself on a reduced (fast) generation. Offline."""
import pytest

from teamscale import calibrate, stats


def test_chi2_sf_known_values():
    assert stats.chi2_sf(0.0, 5) == 1.0
    assert stats.chi2_sf(3.841, 1) == pytest.approx(0.05, abs=2e-3)
    assert stats.chi2_sf(16.919, 9) == pytest.approx(0.05, abs=2e-3)
    assert stats.chi2_sf(2.706, 1) == pytest.approx(0.10, abs=2e-3)


def test_chi_square_uniform_and_skewed():
    uniform = stats.chi_square_test([[50, 50], [50, 50]])
    assert uniform["statistic"] == 0.0 and uniform["p_value"] == 1.0
    assert uniform["effect_size"] == 0.0
    skewed = stats.chi_square_test([[90, 10], [10, 90]])
    assert skewed["p_value"] < 0.001 and skewed["effect_size"] > 0.5


def test_kruskal_wallis_identical_and_separated():
    same = stats.kruskal_wallis([[1, 2, 3, 4], [1, 2, 3, 4],
                                 [1, 2, 3, 4]])
    assert same["statistic"] == 0.0 and same["p_value"] == 1.0
    apart = stats.kruskal_wallis([[1, 2, 3, 4, 5], [50, 51, 52, 53, 54],
                                  [90, 91, 92, 93, 94]])
    assert apart["p_value"] < 0.01 and apart["effect_size"] > 0.5
    const = stats.kruskal_wallis([[7, 7], [7, 7]])
    assert const["p_value"] == 1.0  # degenerate: all values identical


def test_benjamini_hochberg():
    adj = stats.benjamini_hochberg([0.001, 0.04, 0.5], q=0.05)
    assert adj[0] == pytest.approx(0.003)
    assert adj[1] == pytest.approx(0.06)   # not significant at q=0.05
    assert adj[2] == pytest.approx(0.5)
    assert stats.benjamini_hochberg([], q=0.05) == []


def test_gate_passes_on_reduced_generation():
    report = calibrate.run_calibration(n_chains=60, n_steps=50)
    assert report["gate"] == "pass"
    assert report["generator_version"] == "chain-gen-1"
    assert report["n_steps_total"] == 60 * 50
    assert report["shuffle_control"]["null"] is True
    names = {t["name"] for t in report["tests"]}
    assert names == {"template_mix", "op_count", "state_size",
                     "spec_len", "probe_count", "probe_visibility"}
    for t in report["tests"]:
        assert t["significant"] is False and t["negligible"] is True


def test_run_tests_detects_planted_position_dependence():
    # synthetic rows whose template mix is rigged by label: the same
    # machinery the gate uses must call this significant.
    rows = []
    for i in range(400):
        label = i % 2
        rows.append({"template": "affine" if label == 0 else "rotate",
                     "op_count": 12, "state_size": 12, "spec_len": 40,
                     "probe_count": 13, "probe_visibility": 0.5,
                     "position": label + 1})
    tests = calibrate.run_tests(rows, [r["position"] - 1 for r in rows])
    mix = next(t for t in tests if t["name"] == "template_mix")
    assert mix["significant"] is True and mix["negligible"] is False
