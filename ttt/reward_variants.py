"""Reward variants for BurnTTT -- the knob the experiment sweep turns.

Each :class:`RewardWeights` defines a *named* objective: how much the synthesizer
is rewarded for throughput vs. resource frugality vs. accuracy, and how the
shaping references (latency / tokens-per-sec / power) are set. Different Prime
Intellect runs select a different variant via ``BURN_REWARD_VARIANT`` so we can
discover which reward produces the best synthesizer.

All non-legacy variants share the bounded, lexicographically-tiered structure
defined in :mod:`ttt.reward_base`:

* feasible designs live in ``[0, 1]`` (blend of perf / frugality / accuracy),
* every failure class lives in a disjoint band strictly below ``0``,

so the *outcome class* always dominates within-class shaping and a valid design
can never score below a failure regardless of part size.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ttt.reward_base import (
    TIER_ACCURACY_BOT,
    TIER_ACCURACY_TOP,
    TIER_COMPILE_FAIL,
    TIER_NUMERICS_BOT,
    TIER_NUMERICS_TOP,
    TIER_OVERBUDGET_BOT,
    TIER_OVERBUDGET_TOP,
    TIER_TIMING_BOT,
    TIER_TIMING_TOP,
    band,
    clamp,
    get_board_budget,
    resource_fracs,
    safe,
    squash,
)

DEFAULT_VARIANT = "v2_balanced"
ENV_VAR = "BURN_REWARD_VARIANT"


@dataclass(frozen=True)
class RewardWeights:
    """A named reward objective shared by config-mode and HLS-mode rewards."""

    name: str
    description: str
    # Feasible-band blend weights (normalized internally; need not sum to 1).
    w_perf: float = 0.5
    w_resource: float = 0.3
    w_accuracy: float = 0.2
    # Perf shaping half-points: the value at which the perf term reaches ~0.5.
    latency_ref_cycles: float = 800.0
    tps_ref: float = 6000.0
    # Accuracy.
    max_error_threshold: float = 0.05
    # Resource accounting.
    resource_fields: tuple[str, ...] = ("dsp", "lut", "ff", "bram")
    missing_resource_is_worst: bool = False
    power_ref_w: float = 50.0
    # If True, this variant delegates to the original (legacy) raw-count reward.
    legacy: bool = False

    def blend_denom(self) -> float:
        return max(1e-9, self.w_perf + self.w_resource + self.w_accuracy)


# ---------------------------------------------------------------------------
# Variant registry.
# ---------------------------------------------------------------------------
REWARD_VARIANTS: dict[str, RewardWeights] = {
    # Balanced default: care about all three, modest references.
    "v2_balanced": RewardWeights(
        name="v2_balanced",
        description="Bounded tiered reward; balanced throughput/resource/accuracy.",
        w_perf=0.50,
        w_resource=0.30,
        w_accuracy=0.20,
        latency_ref_cycles=800.0,
        tps_ref=6000.0,
        max_error_threshold=0.05,
    ),
    # Chase peak performance; tolerate heavier resource use (still must FIT).
    "throughput_first": RewardWeights(
        name="throughput_first",
        description="Maximize tokens/sec & low latency; light resource penalty.",
        w_perf=0.75,
        w_resource=0.12,
        w_accuracy=0.13,
        latency_ref_cycles=300.0,
        tps_ref=20000.0,
        max_error_threshold=0.05,
        missing_resource_is_worst=True,
    ),
    # Chase small, cheap, board-friendly designs.
    "efficiency_first": RewardWeights(
        name="efficiency_first",
        description="Reward resource frugality and small footprint over raw speed.",
        w_perf=0.22,
        w_resource=0.60,
        w_accuracy=0.18,
        latency_ref_cycles=1500.0,
        tps_ref=3000.0,
        max_error_threshold=0.05,
        missing_resource_is_worst=True,
    ),
    # Tight numerics: prioritize accuracy, useful for low-bit exploration.
    "accuracy_strict": RewardWeights(
        name="accuracy_strict",
        description="Tight accuracy threshold and heavy accuracy weighting.",
        w_perf=0.35,
        w_resource=0.25,
        w_accuracy=0.40,
        latency_ref_cycles=800.0,
        tps_ref=6000.0,
        max_error_threshold=0.02,
    ),
    # The original raw-count reward, kept for ablation/comparison.
    "legacy": RewardWeights(
        name="legacy",
        description="Original unbounded raw-count reward (Phases 1-3).",
        legacy=True,
    ),
}


def get_variant(name: str | None) -> RewardWeights:
    if not name:
        return REWARD_VARIANTS[DEFAULT_VARIANT]
    key = str(name).strip()
    if key in REWARD_VARIANTS:
        return REWARD_VARIANTS[key]
    return REWARD_VARIANTS[DEFAULT_VARIANT]


def active_variant_name() -> str:
    return os.environ.get(ENV_VAR, DEFAULT_VARIANT).strip() or DEFAULT_VARIANT


def active_weights() -> RewardWeights:
    return get_variant(active_variant_name())


def list_variants() -> list[str]:
    return list(REWARD_VARIANTS.keys())


# ---------------------------------------------------------------------------
# Shared shaping helpers.
# ---------------------------------------------------------------------------
def _perf_from_latency(latency_cycles: float | None, ref: float) -> float:
    """Lower latency -> higher perf in (0, 1]. ``ref`` is the ~0.5 half-point."""
    lat = safe(latency_cycles, default=ref)
    if lat <= 0:
        lat = ref
    return clamp(ref / (ref + lat))


def _perf_from_tps(tps: float | None, ref: float) -> float:
    """Higher tokens/sec -> higher perf in [0, 1). ``ref`` is the ~0.5 half-point."""
    t = safe(tps, default=0.0)
    if t <= 0:
        return 0.0
    return clamp(t / (t + ref))


def _resource_summary(
    result: dict[str, Any],
    weights: RewardWeights,
    part_key: str,
) -> tuple[float, float, bool]:
    """Return (binding_frac, frugality, over_budget).

    binding_frac is the max usage/cap across known resources (the constraint that
    actually limits the design). frugality = 1 - binding (clamped) is the feasible
    reward contribution. over_budget is True when any resource exceeds its cap.
    """
    budget = get_board_budget(result.get(part_key) or result.get("part") or result.get("target_part"))
    fracs = resource_fracs(
        result,
        budget,
        weights.resource_fields,
        missing_is_worst=weights.missing_resource_is_worst,
    )
    if not fracs:
        # No resource info at all: neutral, not over budget.
        return 0.5, 0.5, False
    binding = max(fracs.values())
    over = binding > 1.0
    frugality = clamp(1.0 - binding)
    return binding, frugality, over


# ---------------------------------------------------------------------------
# Config-mode reward (hls4ml config author).
# ---------------------------------------------------------------------------
def config_reward(result: dict[str, Any], weights: RewardWeights) -> float:
    """Bounded, tiered reward for a hls4ml ``BurnConfig``/``BlockConfig`` result."""
    if not result.get("compile_success"):
        return TIER_COMPILE_FAIL

    threshold = float(result.get("max_error_threshold", weights.max_error_threshold))
    max_error = result.get("max_error")
    err = safe(max_error, default=0.0)

    # Tier: numerics failure (hard accuracy gate).
    if max_error is not None and err > threshold:
        # Worse (more excess) -> closer to the band bottom.
        excess = (err - threshold) / max(1e-6, threshold)
        quality = 1.0 - squash(excess)
        return band(TIER_NUMERICS_TOP, TIER_NUMERICS_BOT, quality)

    binding, frugality, over = _resource_summary(result, weights, "target_part")

    # Tier: over budget (won't fit the part).
    if over:
        overflow = binding - 1.0
        quality = 1.0 - squash(overflow)
        return band(TIER_OVERBUDGET_TOP, TIER_OVERBUDGET_BOT, quality)

    # Feasible band [0, 1]: blend perf / frugality / accuracy.
    # Latency drives the perf term (more sensitive than the compressed throughput
    # estimate across the reuse grid).
    perf = _perf_from_latency(result.get("latency_cycles"), weights.latency_ref_cycles)
    acc_q = clamp(1.0 - err / max(1e-6, threshold))

    blend = (
        weights.w_perf * perf
        + weights.w_resource * frugality
        + weights.w_accuracy * acc_q
    ) / weights.blend_denom()
    return clamp(blend, 0.0, 1.0)


# ---------------------------------------------------------------------------
# HLS-mode reward (custom Vitis HLS kernel author).
# ---------------------------------------------------------------------------
def hls_reward(result: dict[str, Any], weights: RewardWeights) -> float:
    """Bounded, tiered reward for a custom-HLS ``KernelBundle`` result."""
    # Tier: HLS compile failure.
    if not result.get("hls_compile_success"):
        return TIER_COMPILE_FAIL

    threshold = float(result.get("max_error_threshold", weights.max_error_threshold))

    # Tier: cosim (numerics) failure.
    if not result.get("cosim_pass"):
        err = safe(result.get("max_error"), default=1.0)
        quality = 1.0 - squash(err)
        return band(TIER_NUMERICS_TOP, TIER_NUMERICS_BOT, quality)

    # Tier: timing closure failure.
    if not result.get("timing_met", True):
        wns = safe(result.get("wns_violation_ns"), default=1.0)
        quality = 1.0 - squash(wns)
        return band(TIER_TIMING_TOP, TIER_TIMING_BOT, quality)

    # Tier: post-cosim accuracy gate.
    err = safe(result.get("max_error"), default=0.0)
    if err > threshold:
        excess = (err - threshold) / max(1e-6, threshold)
        quality = 1.0 - squash(excess)
        return band(TIER_ACCURACY_TOP, TIER_ACCURACY_BOT, quality)

    binding, frugality, over = _resource_summary(result, weights, "part")

    # Tier: over budget (won't fit the part).
    if over:
        overflow = binding - 1.0
        quality = 1.0 - squash(overflow)
        return band(TIER_OVERBUDGET_TOP, TIER_OVERBUDGET_BOT, quality)

    # Feasible band [0, 1].
    perf = _perf_from_tps(result.get("tokens_per_sec"), weights.tps_ref)
    acc_q = clamp(1.0 - err / max(1e-6, threshold))
    power = safe(result.get("power_w"), default=0.0)
    power_q = clamp(1.0 - power / max(1e-6, weights.power_ref_w))
    # Fold power into the resource (frugality) term.
    frugality = 0.75 * frugality + 0.25 * power_q

    blend = (
        weights.w_perf * perf
        + weights.w_resource * frugality
        + weights.w_accuracy * acc_q
    ) / weights.blend_denom()
    return clamp(blend, 0.0, 1.0)
