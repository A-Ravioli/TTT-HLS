"""Shared primitives for BurnTTT reward functions.

This module holds the board budgets, numeric helpers, and the bounded-tier
machinery shared by the config-mode reward (:mod:`ttt.reward`), the HLS-mode
reward (:mod:`ttt.reward_hls`), and the reward-variant registry
(:mod:`ttt.reward_variants`).

It deliberately imports nothing from the reward modules so the variant registry
and the public reward functions can both depend on it without a cycle.

Design contract for the v2 rewards built on top of this module
--------------------------------------------------------------
Every reward is **bounded** and **lexicographically tiered** so that the
*outcome class* always dominates the within-class shaping:

    compile/HLS fail  <  numerics fail  <  timing fail  <  over-budget  <  feasible

The tiers occupy disjoint, non-overlapping bands, so a perfectly valid but
resource-heavy design on a *large* part can never score below a compile failure
(the ordering-inversion bug the old raw-count reward had on big FPGAs). Within
each band the score varies smoothly via :func:`squash`, which makes the reward a
good GRPO advantage signal *and* a good regression target for the surrogate.
"""

from __future__ import annotations

from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Board budgets (approximate usable resources per part).
# ---------------------------------------------------------------------------
BOARD_BUDGETS: dict[str, dict[str, int]] = {
    # PYNQ-Z2 (Zynq-7020).
    "xc7z020clg400-1": {"dsp": 220, "lut": 53200, "ff": 106400, "bram": 140},
    # Kria KV260 / ZU7EV-class (Zynq UltraScale+).
    "xczu7ev-ffvc1156-2-e": {"dsp": 1728, "lut": 230400, "ff": 460800, "bram": 312},
    # Alveo U250 (large datacenter FPGA) -- where a Qwen block can realistically land.
    "xcu250-figd2104-2l-e": {"dsp": 12288, "lut": 1728000, "ff": 3456000, "bram": 2688},
    # AWS EC2 F2 -- AMD Virtex UltraScale+ HBM VU47P.
    "xcvu47p-fsvh2892-2-e": {"dsp": 9024, "lut": 1303680, "ff": 2607360, "bram": 2016},
}

DEFAULT_PART = "xc7z020clg400-1"

# Backward-compatible default budget (PYNQ-Z2). Existing imports rely on this.
BOARD_BUDGET = BOARD_BUDGETS[DEFAULT_PART]

# Loose accuracy gate for config mode (kept for backward compatibility).
MAX_ERROR_THRESHOLD = 0.25

# Tier band edges (all rewards live in [-1.0, 1.0]).
# Each failure class gets a disjoint band strictly below the feasible band [0, 1].
TIER_COMPILE_FAIL = -1.0
TIER_NUMERICS_TOP = -0.70  # numerics/accuracy failure band: [-1.0, -0.70)
TIER_NUMERICS_BOT = -1.00
TIER_TIMING_TOP = -0.50  # timing failure band: [-0.70, -0.50)
TIER_TIMING_BOT = -0.70
TIER_ACCURACY_TOP = -0.30  # post-cosim accuracy band: [-0.50, -0.30)
TIER_ACCURACY_BOT = -0.50
TIER_OVERBUDGET_TOP = -0.05  # doesn't-fit band: [-0.30, -0.05)
TIER_OVERBUDGET_BOT = -0.30
TIER_FEASIBLE_BOT = 0.0  # feasible band: [0.0, 1.0]


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


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def squash(x: float) -> float:
    """Map ``[0, inf)`` smoothly and monotonically onto ``[0, 1)``."""
    if x <= 0:
        return 0.0
    return x / (1.0 + x)


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


def resource_fracs(
    result: dict[str, Any],
    budget: dict[str, int],
    fields: Iterable[str],
    *,
    missing_is_worst: bool = False,
) -> dict[str, float]:
    """Return ``{field: usage/cap}`` for each known resource field.

    A missing (``None``) field is skipped by default. When ``missing_is_worst``
    is set, a missing-but-expected field is treated as **fully consumed**
    (frac = 1.0) so that silently-dropped resource estimates cannot make a design
    look free -- this is what closes the "no resource data => no penalty"
    reward-hacking hole in the analytical (no-toolchain) path.
    """
    fracs: dict[str, float] = {}
    for f in fields:
        cap = budget.get(f)
        if not cap:
            continue
        v = result.get(f)
        if v is None:
            if missing_is_worst:
                fracs[f] = 1.0
            continue
        fracs[f] = max(0.0, safe(v)) / cap
    return fracs


def band(top: float, bottom: float, quality: float) -> float:
    """Place ``quality`` in ``[0, 1]`` into the band ``[bottom, top]``.

    quality=1 maps to ``top`` (best within the band), quality=0 maps to
    ``bottom`` (worst within the band).
    """
    quality = clamp(quality)
    return bottom + (top - bottom) * quality


def resource_pct(result: dict[str, Any]) -> dict[str, float]:
    """Percent-of-board utilization for each resource (0 if unknown)."""
    out: dict[str, float] = {}
    budget = get_board_budget(result.get("target_part") or result.get("part"))
    for field, cap in budget.items():
        usage = safe(result.get(field), default=0.0)
        out[f"{field}_pct"] = 100.0 * usage / cap if cap else 0.0
    return out


def fits_board(result: dict[str, Any]) -> bool:
    """True if all known resource usages are within the board budget."""
    budget = get_board_budget(result.get("target_part") or result.get("part"))
    for field, cap in budget.items():
        usage = result.get(field)
        if usage is not None and cap and safe(usage) > cap:
            return False
    return True
