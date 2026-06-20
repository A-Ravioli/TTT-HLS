"""Reward function for ranking hardware configs (see plan.md sections 6 & 9).

The reward heavily penalizes compile/sim failures and accuracy drift, penalizes
resource usage, and rewards low latency. It is intentionally simple and is the
single scalar the online policy learns to predict.
"""

from __future__ import annotations

from typing import Any

# Approximate per-part resource budgets, used for over-budget penalties and the
# "fits board?" check. A toy FFN fits a PYNQ-Z2; a Qwen-2B transformer block does
# not, so larger parts are included for the scaled-up tasks.
BOARD_BUDGETS: dict[str, dict[str, int]] = {
    # PYNQ-Z2 (Zynq-7020).
    "xc7z020clg400-1": {"dsp": 220, "lut": 53200, "ff": 106400, "bram": 140},
    # Kria KV260 / ZU7EV-class (Zynq UltraScale+).
    "xczu7ev-ffvc1156-2-e": {"dsp": 1728, "lut": 230400, "ff": 460800, "bram": 312},
    # Alveo U250 (large datacenter FPGA) -- where a Qwen block can realistically land.
    # Alveo U250 (large datacenter FPGA).
    "xcu250-figd2104-2l-e": {"dsp": 12288, "lut": 1728000, "ff": 3456000, "bram": 2688},
    # AWS EC2 F2 — AMD Virtex UltraScale+ HBM VU47P.
    "xcvu47p-fsvh2892-2-e": {"dsp": 9024, "lut": 1303680, "ff": 2607360, "bram": 2016},
}

DEFAULT_PART = "xc7z020clg400-1"

# Backward-compatible default budget (PYNQ-Z2). Existing imports rely on this.
BOARD_BUDGET = BOARD_BUDGETS[DEFAULT_PART]

# Accuracy beyond this is treated as a failed config.
MAX_ERROR_THRESHOLD = 0.25


def get_board_budget(part: str | None) -> dict[str, int]:
    """Return the resource budget for ``part`` (prefix/loose match), else default."""
    if not part:
        return BOARD_BUDGET
    key = str(part).strip().lower()
    if key in BOARD_BUDGETS:
        return BOARD_BUDGETS[key]
    for known, budget in BOARD_BUDGETS.items():
        if key.startswith(known.split("-")[0]) or known.startswith(key.split("-")[0]):
            return budget
    return BOARD_BUDGET


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
    """Compute the scalar reward for an evaluation result dict.

    Uses the budget of ``result['target_part']`` when present (so a Qwen block on
    an Alveo is judged against Alveo capacity), falling back to the PYNQ-Z2
    default for results that don't carry a part.
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

    # Over-budget penalties (plan.md section 9): punish configs that won't fit.
    budget = get_board_budget(result.get("target_part"))
    for field, cap in budget.items():
        usage = safe(result.get(field), default=0.0)
        if cap and usage > 0.8 * cap:
            score -= 300.0
    return score


def resource_pct(result: dict[str, Any]) -> dict[str, float]:
    """Percent-of-board utilization for each resource (0 if unknown)."""
    out = {}
    budget = get_board_budget(result.get("target_part"))
    for field, cap in budget.items():
        usage = safe(result.get(field), default=0.0)
        out[f"{field}_pct"] = 100.0 * usage / cap if cap else 0.0
    return out


def fits_board(result: dict[str, Any]) -> bool:
    """True if all known resource usages are within the board budget."""
    budget = get_board_budget(result.get("target_part"))
    for field, cap in budget.items():
        usage = result.get(field)
        if usage is not None and cap and float(usage) > cap:
            return False
    return True
