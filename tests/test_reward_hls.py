"""Tests for ttt.reward_hls — HLS reward function gates."""

import pytest

from ttt.reward_hls import reward_hls


def test_compile_failure_returns_negative_1000():
    result = {"hls_compile_success": False}
    assert reward_hls(result) == -1000.0


def test_cosim_failure_penalty():
    result = {
        "hls_compile_success": True,
        "cosim_pass": False,
        "max_error": 0.5,
    }
    r = reward_hls(result)
    assert r == pytest.approx(-800.0 - 100.0 * 0.5)


def test_timing_failure_penalty():
    result = {
        "hls_compile_success": True,
        "cosim_pass": True,
        "timing_met": False,
        "wns_violation_ns": 2.0,
        "max_error": 0.005,
        "max_error_threshold": 0.01,
    }
    r = reward_hls(result)
    assert r == pytest.approx(-600.0 - 10.0 * 2.0)


def test_accuracy_threshold_gate():
    result = {
        "hls_compile_success": True,
        "cosim_pass": True,
        "timing_met": True,
        "max_error": 0.05,
        "max_error_threshold": 0.01,
        "tokens_per_sec": 1000.0,
    }
    r = reward_hls(result)
    assert r < -400  # Should be penalized for exceeding threshold


def test_success_reward_scales_with_tps():
    base = {
        "hls_compile_success": True,
        "cosim_pass": True,
        "timing_met": True,
        "max_error": 0.001,
        "max_error_threshold": 0.01,
        "power_w": 10.0,
    }
    slow = {**base, "tokens_per_sec": 100.0}
    fast = {**base, "tokens_per_sec": 1000.0}
    assert reward_hls(fast) > reward_hls(slow)


def test_success_with_zero_tps():
    result = {
        "hls_compile_success": True,
        "cosim_pass": True,
        "timing_met": True,
        "max_error": 0.001,
        "max_error_threshold": 0.01,
        "tokens_per_sec": 0.0,
        "power_w": 0.0,
    }
    r = reward_hls(result)
    # Should be near 0 (1000 * 0 - small penalties)
    assert r < 0  # penalty from max_error


def test_reward_scale_is_sane_not_astronomical():
    # log-scaled throughput keeps the reward in a sane range even at realistic
    # tokens/sec (used to be ~1e8 with the literal 1000*tps, dwarfing penalties).
    result = {
        "hls_compile_success": True,
        "cosim_pass": True,
        "timing_met": True,
        "max_error": 0.001,
        "max_error_threshold": 0.01,
        "tokens_per_sec": 1e5,
        "power_w": 0.0,
    }
    assert 0 < reward_hls(result) < 1e5


def test_over_budget_beats_marginal_speedup():
    # A faster-but-over-budget kernel must NOT outrank a slightly slower in-budget
    # one. With the old 1000*tps reward the over-budget penalty was negligible.
    over = {
        "hls_compile_success": True,
        "cosim_pass": True,
        "timing_met": True,
        "max_error": 0.001,
        "max_error_threshold": 0.01,
        "tokens_per_sec": 1.0e5,
        "power_w": 0.0,
        "part": "xc7z020clg400-1",  # 220 DSPs
        "dsp": 500,                 # over budget
    }
    in_budget = {**over, "tokens_per_sec": 9.0e4, "dsp": 100}
    assert reward_hls(in_budget) > reward_hls(over)


def test_over_budget_penalty():
    result = {
        "hls_compile_success": True,
        "cosim_pass": True,
        "timing_met": True,
        "max_error": 0.001,
        "max_error_threshold": 0.01,
        "tokens_per_sec": 100.0,
        "power_w": 0.0,
        "part": "xc7z020clg400-1",  # PYNQ-Z2: 220 DSPs
        "dsp": 500,  # Way over budget
    }
    r = reward_hls(result)
    # Should have over-budget penalty
    result_in_budget = {**result, "dsp": 100}
    assert reward_hls(result_in_budget) > r
