"""Phase A-3 controller: Wilson intervals against known values, the
error signal, and the closed-loop control law driven by synthetic
probe functions -- convergence, all three declared-failure classes,
the max-settings bound, and the spend-cap abort path. Everything
offline."""
import pytest

from teamscale import model_calibrate as mc
from teamscale.ladder import LADDER
from teamscale.stats import wilson_interval


def test_wilson_known_values():
    lo, hi = wilson_interval(0, 100)
    assert lo == 0.0 and hi == pytest.approx(0.036993, abs=1e-6)
    lo, hi = wilson_interval(100, 100)
    assert hi == pytest.approx(1.0, abs=1e-9)
    assert lo == pytest.approx(1 - 0.036993, abs=1e-6)
    lo, hi = wilson_interval(10, 100)
    assert lo == pytest.approx(0.055229, abs=1e-6)
    assert hi == pytest.approx(0.174366, abs=1e-6)
    # design table: 5% band validation at n=500 -> half-width ~1.9%
    lo, hi = wilson_interval(25, 500)
    assert lo == pytest.approx(0.034094, abs=1e-6)
    assert hi == pytest.approx(0.072768, abs=1e-6)
    assert wilson_interval(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        wilson_interval(5, 4)


def test_band_error_signal():
    # CI entirely below the band -> negative distance to band_lo
    assert mc.band_error(0.01, 0.04, 0.07, 0.13) \
        == pytest.approx(-0.03)
    # CI entirely above -> positive distance from band_hi
    assert mc.band_error(0.20, 0.30, 0.07, 0.13) \
        == pytest.approx(0.07)
    # intersecting -> 0
    assert mc.band_error(0.05, 0.20, 0.07, 0.13) == 0.0


def _synthetic(rates, n=200):
    """probe_setting from a per-rung true-rate table; the batch is
    exactly round(rate * n) failures, so CIs are deterministic."""
    def probe(setting):
        idx = LADDER.index(setting)
        failures = int(round(rates[idx] * n))
        ci_lo, ci_hi = wilson_interval(failures, n)
        return {"n": n, "failures": failures, "errors": 0,
                "rate": failures / n, "ci_lo": ci_lo, "ci_hi": ci_hi}
    return probe


def test_controller_converges_and_freezes():
    # failure rate climbs the ladder; rung 5 sits at exactly 10%
    rates = [0.0, 0.01, 0.02, 0.04, 0.06, 0.10, 0.14, 0.20, 0.30, 0.40]
    result = mc.run_controller(target=0.10, tolerance=0.03,
                               probe_setting=_synthetic(rates))
    assert result["status"] == "converged"
    assert result["setting"] == LADDER[result["setting_index"]].name
    assert result["iterations"] == len(result["trace"]) <= 6
    last = result["trace"][-1]
    assert last["setting"] == result["setting"]
    assert "converged" in last["adjustment"]
    for entry in result["trace"]:
        assert {"iteration", "setting", "n", "failures", "rate",
                "ci_lo", "ci_hi", "error", "adjustment"} \
            <= set(entry)


def test_controller_saturated_below_is_declared_failure():
    rates = [0.0] * len(LADDER)   # nothing ever fails: band unreachable
    result = mc.run_controller(target=0.05, tolerance=0.02,
                               probe_setting=_synthetic(rates))
    assert result["status"] == "declared_failure"
    assert result["failure_class"] == "saturated_below"
    assert result["setting"] is None
    # it must have probed the top rung before declaring saturation
    assert max(e["setting_index"] for e in result["trace"]) \
        == len(LADDER) - 1


def test_controller_overshoot_at_floor_is_declared_failure():
    rates = [0.50] * len(LADDER)  # even the floor is far above 5%
    result = mc.run_controller(target=0.05, tolerance=0.02,
                               probe_setting=_synthetic(rates))
    assert result["status"] == "declared_failure"
    assert result["failure_class"] == "overshoot_at_floor"
    assert min(e["setting_index"] for e in result["trace"]) == 0


def test_controller_band_between_rungs_is_no_stable_setting():
    # rung 4 at 2% (CI below the 10% band), rung 5 at 30% (CI above):
    # the band falls between adjacent rungs
    rates = [0.0, 0.0, 0.01, 0.01, 0.02, 0.30, 0.35, 0.40, 0.45, 0.50]
    result = mc.run_controller(target=0.10, tolerance=0.03,
                               probe_setting=_synthetic(rates))
    assert result["status"] == "declared_failure"
    assert result["failure_class"] == "no_stable_setting"


def test_controller_respects_max_settings():
    rates = [0.0, 0.01, 0.02, 0.04, 0.06, 0.10, 0.14, 0.20, 0.30, 0.40]
    result = mc.run_controller(target=0.10, tolerance=0.03,
                               probe_setting=_synthetic(rates),
                               max_settings=2)
    assert result["iterations"] == 2
    assert result["status"] == "declared_failure"


def test_controller_abort_path():
    def probe(setting):
        return {"aborted": True}
    result = mc.run_controller(target=0.10, tolerance=0.03,
                               probe_setting=probe)
    assert result["status"] == "aborted_spend_cap"
    assert result["trace"][-1]["adjustment"].startswith("aborted")


def test_batch_stats_excludes_error_probes():
    records = [
        {"outcome": "correct"}, {"outcome": "incorrect"},
        {"outcome": "incorrect"}, {"outcome": "error"},
    ]
    stats = mc.batch_stats(records)
    assert stats["n"] == 3 and stats["failures"] == 2
    assert stats["errors"] == 1
    assert stats["rate"] == pytest.approx(2 / 3)
