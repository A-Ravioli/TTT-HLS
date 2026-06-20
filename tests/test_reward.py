"""Tests for the config-mode reward (ttt.reward) and its bounded-tier contract."""

from ttt.reward import fits_board, legacy_reward, resource_pct, reward, safe
from ttt.reward_base import (
    TIER_COMPILE_FAIL,
    TIER_NUMERICS_TOP,
    TIER_OVERBUDGET_TOP,
)
from ttt.reward_variants import config_reward, get_variant

BAL = get_variant("v2_balanced")


def _cfg(**kw):
    base = {"compile_success": True, "max_error": 0.01, "target_part": "xc7z020clg400-1"}
    base.update(kw)
    return base


def test_safe_handles_none_and_garbage():
    assert safe(None, default=5.0) == 5.0
    assert safe("abc", default=1.0) == 1.0
    assert safe(float("nan"), default=2.0) == 2.0
    assert safe(3) == 3.0


def test_reward_is_bounded():
    # Every outcome stays within [-1, 1].
    cases = [
        {"compile_success": False},
        _cfg(max_error=0.9),
        _cfg(dsp=10_000_000, lut=10_000_000),
        _cfg(latency_cycles=10, dsp=1, lut=1, ff=1, bram=1),
    ]
    for c in cases:
        r = config_reward(c, BAL)
        assert -1.0 <= r <= 1.0


def test_compile_failure_is_floor():
    assert config_reward({"compile_success": False}, BAL) == TIER_COMPILE_FAIL


def test_tier_ordering_holds_on_large_part():
    """The headline fix: a valid resource-heavy design on a BIG part must beat a
    compile failure AND an accuracy failure (the old raw-count reward inverted
    this)."""
    part = "xcu250-figd2104-2l-e"  # Alveo U250, huge LUT budget
    compile_fail = config_reward({"compile_success": False, "target_part": part}, BAL)
    acc_fail = config_reward(
        {"compile_success": True, "max_error": 0.9, "target_part": part}, BAL
    )
    # Heavy but FITS the U250 (large absolute counts, well under budget).
    heavy_fits = config_reward(
        _cfg(
            target_part=part,
            max_error=0.02,
            latency_cycles=500,
            dsp=4000,
            lut=400_000,
            ff=400_000,
            bram=400,
        ),
        BAL,
    )
    assert heavy_fits > acc_fail > compile_fail
    assert heavy_fits >= 0.0  # feasible band


def test_over_budget_below_feasible_above_accuracy_fail():
    over = config_reward(_cfg(dsp=10_000, max_error=0.02), BAL)  # >> PYNQ 220 DSP
    feasible = config_reward(_cfg(dsp=20, lut=2000, ff=2000, bram=4, latency_cycles=200), BAL)
    acc_fail = config_reward(_cfg(max_error=0.9), BAL)
    assert acc_fail < over < TIER_OVERBUDGET_TOP + 1e-9
    assert over < feasible
    assert feasible >= 0.0


def test_good_config_beats_resource_heavy_config():
    light = _cfg(latency_cycles=100, dsp=20, lut=3000, ff=3000, bram=4)
    heavy = _cfg(latency_cycles=100, dsp=180, lut=45000, ff=45000, bram=100)
    assert config_reward(light, BAL) > config_reward(heavy, BAL)


def test_lower_latency_is_rewarded():
    base = dict(dsp=10, lut=1000, ff=1000, bram=2)
    fast = config_reward(_cfg(latency_cycles=100, **base), BAL)
    slow = config_reward(_cfg(latency_cycles=5000, **base), BAL)
    assert fast > slow


def test_accuracy_drift_is_penalized_into_numerics_band():
    r = config_reward(_cfg(max_error=0.9), BAL)
    assert r < TIER_NUMERICS_TOP


def test_variants_differ_in_emphasis():
    """throughput_first should reward a fast-but-heavy design more than
    efficiency_first does (relative to a slow-but-light design)."""
    fast_heavy = _cfg(latency_cycles=80, dsp=180, lut=40000, ff=40000, bram=100)
    slow_light = _cfg(latency_cycles=3000, dsp=10, lut=1000, ff=1000, bram=2)
    tf = get_variant("throughput_first")
    ef = get_variant("efficiency_first")
    tf_gap = config_reward(fast_heavy, tf) - config_reward(slow_light, tf)
    ef_gap = config_reward(fast_heavy, ef) - config_reward(slow_light, ef)
    assert tf_gap > ef_gap


def test_legacy_variant_preserves_old_numbers():
    assert legacy_reward({"compile_success": False}) == -1000.0
    assert legacy_reward({"compile_success": True, "max_error": 0.5}) < -500.0


def test_missing_resources_do_not_crash():
    r = config_reward({"compile_success": True, "max_error": 0.0}, BAL)
    assert isinstance(r, float)
    assert -1.0 <= r <= 1.0


def test_fits_board():
    assert fits_board({"dsp": 50, "lut": 1000}) is True
    assert fits_board({"dsp": 5000}) is False
    assert fits_board({"dsp": None}) is True


def test_resource_pct():
    pct = resource_pct({"dsp": 110})
    assert abs(pct["dsp_pct"] - 50.0) < 1e-6


def test_default_reward_uses_active_variant(monkeypatch):
    monkeypatch.setenv("BURN_REWARD_VARIANT", "legacy")
    assert reward({"compile_success": False}) == -1000.0
    monkeypatch.setenv("BURN_REWARD_VARIANT", "v2_balanced")
    assert reward({"compile_success": False}) == TIER_COMPILE_FAIL
