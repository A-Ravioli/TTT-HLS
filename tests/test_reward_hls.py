"""Tests for the HLS-mode reward (ttt.reward_hls) and its bounded-tier contract."""

from ttt.reward_base import (
    TIER_ACCURACY_TOP,
    TIER_COMPILE_FAIL,
    TIER_NUMERICS_TOP,
    TIER_OVERBUDGET_TOP,
    TIER_TIMING_TOP,
)
from ttt.reward_hls import legacy_reward_hls, reward_hls
from ttt.reward_variants import get_variant, hls_reward

BAL = get_variant("v2_balanced")


def _ok(**kw):
    base = {
        "hls_compile_success": True,
        "cosim_pass": True,
        "timing_met": True,
        "max_error": 0.001,
        "max_error_threshold": 0.01,
        "part": "xc7z020clg400-1",
        "dsp": 50,
        "lut": 5000,
        "ff": 5000,
        "bram": 10,
        "tokens_per_sec": 5000.0,
        "power_w": 5.0,
    }
    base.update(kw)
    return base


def test_compile_failure_is_floor():
    assert hls_reward({"hls_compile_success": False}, BAL) == TIER_COMPILE_FAIL


def test_tier_ordering():
    compile_fail = hls_reward({"hls_compile_success": False}, BAL)
    cosim_fail = hls_reward({"hls_compile_success": True, "cosim_pass": False, "max_error": 0.5}, BAL)
    timing_fail = hls_reward(
        {"hls_compile_success": True, "cosim_pass": True, "timing_met": False, "wns_violation_ns": 2.0,
         "max_error": 0.001, "max_error_threshold": 0.01}, BAL
    )
    acc_fail = hls_reward(
        {"hls_compile_success": True, "cosim_pass": True, "timing_met": True, "max_error": 0.05,
         "max_error_threshold": 0.01, "tokens_per_sec": 1000.0}, BAL
    )
    over = hls_reward(_ok(dsp=100000), BAL)  # way over PYNQ budget
    feasible = hls_reward(_ok(), BAL)

    assert compile_fail < cosim_fail < timing_fail < acc_fail < over < feasible
    assert cosim_fail < TIER_NUMERICS_TOP
    assert timing_fail < TIER_TIMING_TOP
    assert acc_fail < TIER_ACCURACY_TOP
    assert over < TIER_OVERBUDGET_TOP + 1e-9
    assert feasible >= 0.0


def test_all_outcomes_bounded():
    for c in [
        {"hls_compile_success": False},
        {"hls_compile_success": True, "cosim_pass": False, "max_error": 9.0},
        _ok(tokens_per_sec=1e12),
        _ok(dsp=10**9),
    ]:
        r = hls_reward(c, BAL)
        assert -1.0 <= r <= 1.0


def test_success_reward_scales_with_tps():
    slow = hls_reward(_ok(tokens_per_sec=100.0), BAL)
    fast = hls_reward(_ok(tokens_per_sec=50000.0), BAL)
    assert fast > slow


def test_over_budget_penalized():
    in_budget = hls_reward(_ok(dsp=100), BAL)
    over = hls_reward(_ok(dsp=100000), BAL)
    assert in_budget > over


def test_tile_inflation_cannot_beat_a_fitting_design():
    """A kernel that 'inflates tiles' for huge tps but blows the resource budget
    must NOT outscore a modest fitting kernel (the closed reward hack)."""
    fitting = hls_reward(_ok(tokens_per_sec=4000.0, dsp=100, lut=5000), BAL)
    bloated = hls_reward(_ok(tokens_per_sec=500000.0, dsp=100000, lut=2_000_000), BAL)
    assert fitting > bloated


def test_legacy_variant_preserved():
    assert legacy_reward_hls({"hls_compile_success": False}) == -1000.0


def test_default_dispatch(monkeypatch):
    monkeypatch.setenv("BURN_REWARD_VARIANT", "legacy")
    assert reward_hls({"hls_compile_success": False}) == -1000.0
    monkeypatch.setenv("BURN_REWARD_VARIANT", "v2_balanced")
    assert reward_hls({"hls_compile_success": False}) == TIER_COMPILE_FAIL
