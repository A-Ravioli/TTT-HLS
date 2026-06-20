"""HLS-mode reward for custom Vitis HLS kernels (Phase 4+).

The public :func:`reward_hls` dispatches to the **active reward variant**
(``BURN_REWARD_VARIANT``; see :mod:`ttt.reward_variants`). The default variant is
bounded and lexicographically tiered:

    HLS compile fail  <  cosim fail  <  timing fail  <  accuracy fail
                      <  over-budget  <  feasible

Feasible kernels score in ``[0, 1]`` as a blend of tokens/sec, resource frugality
(as a fraction of the part budget) folded with power, and accuracy. This removes
the old ``1000 * tps`` scale dependence and -- together with the analytical
resource estimates now populated in :mod:`ttt.evaluate_hls` -- closes the
"inflate the tile size for free throughput" reward hack in the no-toolchain path.

The original reward is preserved as the ``legacy`` variant and as
:func:`legacy_reward_hls`.
"""

from __future__ import annotations

from typing import Any

from paths import get_logger
from ttt.reward_base import BOARD_BUDGETS, get_board_budget, safe

logger = get_logger("burnttt.reward_hls")

# Tight accuracy gate for cosim (kept for backward compatibility).
HLS_MAX_ERROR_THRESHOLD = 0.01

__all__ = [
    "HLS_MAX_ERROR_THRESHOLD",
    "BOARD_BUDGETS",
    "get_board_budget",
    "safe",
    "reward_hls",
    "legacy_reward_hls",
    "reward_hls_comparison",
]


def legacy_reward_hls(result: dict[str, Any]) -> float:
    """The original unbounded HLS reward (kept for ablation)."""
    if not result.get("hls_compile_success"):
        return -1000.0
    if not result.get("cosim_pass"):
        max_err = safe(result.get("max_error"), default=1.0)
        return -800.0 - 100.0 * max_err
    if not result.get("timing_met", True):
        wns = safe(result.get("wns_violation_ns"), default=1.0)
        return -600.0 - 10.0 * wns
    max_err = safe(result.get("max_error"), default=0.0)
    threshold = result.get("max_error_threshold", HLS_MAX_ERROR_THRESHOLD)
    if max_err > threshold:
        return -500.0 - 100.0 * max_err
    tps = safe(result.get("tokens_per_sec"), default=0.0)
    power = safe(result.get("power_w"), default=0.0)
    score = 1000.0 * tps - 0.01 * power - 50.0 * max_err
    budget = get_board_budget(result.get("part"))
    for field, cap in budget.items():
        usage = safe(result.get(field), default=0.0)
        if cap and usage > cap:
            score -= 500.0
    return score


def reward_hls(result: dict[str, Any]) -> float:
    """Compute the scalar HLS reward using the active reward variant."""
    from ttt.reward_variants import active_weights, hls_reward

    weights = active_weights()
    if weights.legacy:
        return legacy_reward_hls(result)
    # The HLS cosim gate is tighter than the variant's generic threshold unless
    # the result explicitly carries its own; honor an explicit per-result value.
    if "max_error_threshold" not in result:
        result = {**result, "max_error_threshold": min(weights.max_error_threshold, HLS_MAX_ERROR_THRESHOLD)}
    return hls_reward(result, weights)


def reward_hls_comparison(result_hls: dict[str, Any], result_hls4ml: dict[str, Any]) -> dict[str, Any]:
    """Compare custom HLS vs hls4ml config: which is better?"""
    hls_tps = safe(result_hls.get("tokens_per_sec"), default=0.0)
    hls4ml_tps = safe(result_hls4ml.get("tokens_per_sec"), default=0.0)

    hls_reward_val = safe(result_hls.get("reward"), default=-1.0)
    hls4ml_reward_val = safe(result_hls4ml.get("reward"), default=-1.0)

    speedup = hls_tps / hls4ml_tps if hls4ml_tps > 0 else float("inf")

    return {
        "hls_tokens_per_sec": hls_tps,
        "hls4ml_tokens_per_sec": hls4ml_tps,
        "speedup": speedup,
        "hls_reward": hls_reward_val,
        "hls4ml_reward": hls4ml_reward_val,
        "hls_wins": hls_reward_val > hls4ml_reward_val,
        "hls_cosim_pass": result_hls.get("cosim_pass", False),
        "hls_timing_met": result_hls.get("timing_met", False),
        "hls_max_error": result_hls.get("max_error"),
        "hls4ml_max_error": result_hls4ml.get("max_error"),
    }
