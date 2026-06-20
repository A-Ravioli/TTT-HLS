"""Config-mode reward for ranking hls4ml hardware configs.

The public :func:`reward` dispatches to the **active reward variant** (selected by
``BURN_REWARD_VARIANT``; see :mod:`ttt.reward_variants`). The default variant is a
bounded, lexicographically-tiered, budget-normalized objective:

    compile fail  <  numerics fail  <  over-budget  <  feasible

with feasible designs scored in ``[0, 1]`` as a blend of latency, resource
frugality (as a *fraction of the part's budget*, so it is part-agnostic), and
accuracy. This fixes the old raw-count reward, where a valid but resource-heavy
design on a large part could score *below* a compile failure.

The original unbounded reward is preserved as the ``legacy`` variant and as
:func:`legacy_reward` for ablation.

Numeric helpers (:func:`safe`, :func:`get_board_budget`, :func:`resource_pct`,
:func:`fits_board`) and :data:`BOARD_BUDGETS` are re-exported from
:mod:`ttt.reward_base` for backward compatibility.
"""

from __future__ import annotations

from typing import Any

from ttt.reward_base import (
    BOARD_BUDGET,
    BOARD_BUDGETS,
    DEFAULT_PART,
    MAX_ERROR_THRESHOLD,
    fits_board,
    get_board_budget,
    resource_pct,
    safe,
)

__all__ = [
    "BOARD_BUDGET",
    "BOARD_BUDGETS",
    "DEFAULT_PART",
    "MAX_ERROR_THRESHOLD",
    "fits_board",
    "get_board_budget",
    "resource_pct",
    "safe",
    "reward",
    "legacy_reward",
]


def legacy_reward(result: dict[str, Any]) -> float:
    """The original unbounded raw-count reward (Phases 1-3).

    Kept for ablation against the bounded variants. Note its known failure mode:
    on a large part the raw resource terms can drive a *valid* design below the
    compile-failure floor of -1000.
    """
    if not result.get("compile_success"):
        return -1000.0

    max_error = result.get("max_error")
    if max_error is not None and max_error > MAX_ERROR_THRESHOLD:
        return -500.0 - 100.0 * float(max_error)

    score = (
        1000.0
        - 0.05 * safe(result.get("latency_cycles"), default=10000.0)
        - 1.0 * safe(result.get("dsp"), default=0.0)
        - 0.05 * safe(result.get("lut"), default=0.0)
        - 0.1 * safe(result.get("bram"), default=0.0)
        - 50.0 * safe(result.get("max_error"), default=0.0)
    )

    budget = get_board_budget(result.get("target_part"))
    for field, cap in budget.items():
        usage = safe(result.get(field), default=0.0)
        if cap and usage > 0.8 * cap:
            score -= 300.0
    return score


def reward(result: dict[str, Any]) -> float:
    """Compute the scalar reward using the active reward variant."""
    from ttt.reward_variants import active_weights, config_reward

    weights = active_weights()
    if weights.legacy:
        return legacy_reward(result)
    return config_reward(result, weights)
