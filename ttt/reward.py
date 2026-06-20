"""Reward function for ranking hardware configs (see plan.md sections 6 & 9).

The reward heavily penalizes compile/sim failures and accuracy drift, penalizes
resource usage, and rewards low latency. It is intentionally simple and is the
single scalar the online policy learns to predict.
"""

from __future__ import annotations

from typing import Any

# PYNQ-Z2 (xc7z020) approximate resource budget, used for over-budget penalties.
BOARD_BUDGET = {"dsp": 220, "lut": 53200, "ff": 106400, "bram": 140}

# Accuracy beyond this is treated as a failed config.
MAX_ERROR_THRESHOLD = 0.25


def safe(value: Any, default: float = 0.0) -> float:
    """Return ``float(value)`` or ``default`` if value is None/NaN/non-numeric."""
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if f != f:  # NaN
        return default
    return f


def reward(result: dict[str, Any]) -> float:
    """Compute the scalar reward for an evaluation result dict."""
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

    # Over-budget penalties (plan.md section 9): punish configs that won't fit.
    for field, budget in BOARD_BUDGET.items():
        usage = safe(result.get(field), default=0.0)
        if budget and usage > 0.8 * budget:
            score -= 300.0
    return score


def resource_pct(result: dict[str, Any]) -> dict[str, float]:
    """Percent-of-board utilization for each resource (0 if unknown)."""
    out = {}
    for field, budget in BOARD_BUDGET.items():
        usage = safe(result.get(field), default=0.0)
        out[f"{field}_pct"] = 100.0 * usage / budget if budget else 0.0
    return out


def fits_board(result: dict[str, Any]) -> bool:
    """True if all known resource usages are within the board budget."""
    for field, budget in BOARD_BUDGET.items():
        usage = result.get(field)
        if usage is not None and budget and float(usage) > budget:
            return False
    return True
