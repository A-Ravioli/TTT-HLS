"""Reward function for custom HLS kernels (Phase 4+, plan.md section 6).

Primary metric: **tokens/sec** (or inverse latency per token).
Hard constraints: cosim pass, timing closure, max_error threshold.
Soft penalties: power, BRAM usage.
"""

from __future__ import annotations

from typing import Any

from paths import get_logger
from ttt.reward import BOARD_BUDGETS, get_board_budget, safe

logger = get_logger("burnttt.reward_hls")

# Accuracy threshold for cosim (tighter than config mode)
HLS_MAX_ERROR_THRESHOLD = 0.01


def reward_hls(result: dict[str, Any]) -> float:
    """Compute the scalar reward for an HLS evaluation result.

    Implements the reward function from plan.md section 6:
    - Compile failure: -1000
    - Cosim failure: -800 - 100 * max_error
    - Timing failure: -600 - 10 * WNS violation
    - Success: 1000 * tokens_per_sec - penalties
    """
    # Gate 1: HLS compile
    if not result.get("hls_compile_success"):
        return -1000.0

    # Gate 2: Cosim correctness
    if not result.get("cosim_pass"):
        max_err = safe(result.get("max_error"), default=1.0)
        return -800.0 - 100.0 * max_err

    # Gate 3: Timing closure
    if not result.get("timing_met", True):
        wns = safe(result.get("wns_violation_ns"), default=1.0)
        return -600.0 - 10.0 * wns

    # Gate 4: Accuracy within threshold
    max_err = safe(result.get("max_error"), default=0.0)
    threshold = result.get("max_error_threshold", HLS_MAX_ERROR_THRESHOLD)
    if max_err > threshold:
        return -500.0 - 100.0 * max_err

    # Success: primary reward is throughput
    tps = safe(result.get("tokens_per_sec"), default=0.0)
    power = safe(result.get("power_w"), default=0.0)

    score = 1000.0 * tps - 0.01 * power - 50.0 * max_err

    # Over-budget penalties
    budget = get_board_budget(result.get("part"))
    for field, cap in budget.items():
        usage = safe(result.get(field), default=0.0)
        if cap and usage > cap:
            score -= 500.0  # Hard over-budget penalty for HLS mode

    return score


def reward_hls_comparison(result_hls: dict[str, Any], result_hls4ml: dict[str, Any]) -> dict[str, Any]:
    """Compare custom HLS vs hls4ml config: which is better?

    Returns a summary dict with the comparison metrics.
    """
    hls_tps = safe(result_hls.get("tokens_per_sec"), default=0.0)
    hls4ml_tps = safe(result_hls4ml.get("tokens_per_sec"), default=0.0)

    hls_reward = safe(result_hls.get("reward"), default=-1000.0)
    hls4ml_reward = safe(result_hls4ml.get("reward"), default=-1000.0)

    speedup = hls_tps / hls4ml_tps if hls4ml_tps > 0 else float("inf")

    return {
        "hls_tokens_per_sec": hls_tps,
        "hls4ml_tokens_per_sec": hls4ml_tps,
        "speedup": speedup,
        "hls_reward": hls_reward,
        "hls4ml_reward": hls4ml_reward,
        "hls_wins": hls_reward > hls4ml_reward,
        "hls_cosim_pass": result_hls.get("cosim_pass", False),
        "hls_timing_met": result_hls.get("timing_met", False),
        "hls_max_error": result_hls.get("max_error"),
        "hls4ml_max_error": result_hls4ml.get("max_error"),
    }
