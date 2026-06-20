"""Reward-variant sweep: train the synthesizer under different rewards and rank.

The idea: each :mod:`ttt.reward_variants` variant defines a *different objective*
for the test-time-trained GLM synthesizer. Running a TTT search under each variant
tells us which reward shaping actually steers the generator toward the best
hardware. Because each variant's own reward is on its own scale, we compare runs
on a single **canonical** reward (``v2_balanced``) evaluated on the best design
each run produced -- so "which reward is best" is judged by the *hardware it led
to*, not by its own (incomparable) numbers.

This module holds the pure-Python orchestration (variant selection, canonical
scoring, leaderboard CSV, iteration strategy) so it is unit-testable without a GPU
or toolchain. :mod:`scripts.19_reward_sweep` wires it to the real TTT search.
"""

from __future__ import annotations

import csv
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from paths import RESULTS_DIR, ensure_dirs, get_logger
from ttt.reward_variants import config_reward, get_variant, list_variants

logger = get_logger("burnttt.infra.reward_sweep")

SWEEP_CSV = RESULTS_DIR / "reward_sweep.csv"

# All runs are compared on this single, fixed objective regardless of which reward
# variant drove the search.
CANONICAL_VARIANT = "v2_balanced"

# A row producer: given a variant name (already set in the env) returns the list
# of evaluated result rows for one TTT run.
RunFn = Callable[[str], list[dict[str, Any]]]


def sweep_variants(include_legacy: bool = False) -> list[str]:
    """The default variant set for a sweep (non-legacy first, legacy optional)."""
    variants = [v for v in list_variants() if v != "legacy"]
    if include_legacy:
        variants.append("legacy")
    return variants


def canonical_score(row: dict[str, Any]) -> float:
    """Score a result row on the fixed canonical objective."""
    return config_reward(row, get_variant(CANONICAL_VARIANT))


@contextmanager
def reward_variant_env(variant: str):
    """Temporarily set ``BURN_REWARD_VARIANT`` for the duration of a run."""
    prev = os.environ.get("BURN_REWARD_VARIANT")
    os.environ["BURN_REWARD_VARIANT"] = variant
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("BURN_REWARD_VARIANT", None)
        else:
            os.environ["BURN_REWARD_VARIANT"] = prev


@dataclass
class VariantResult:
    iteration: int
    variant: str
    n_evals: int
    best_reward: float  # under the variant's own reward
    best_canonical: float  # the comparable score we rank on
    mean_reward: float
    best_config: str
    best_max_error: float | None
    best_latency: float | None
    compile_rate: float
    wall_seconds: float
    wandb_url: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _f(row: dict[str, Any], key: str) -> float | None:
    v = row.get(key)
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def summarize_run(iteration: int, variant: str, rows: list[dict[str, Any]], wall_seconds: float) -> VariantResult:
    """Reduce a run's rows to a comparable :class:`VariantResult`."""
    if not rows:
        return VariantResult(
            iteration, variant, 0, float("-inf"), float("-inf"), float("-inf"),
            "", None, None, 0.0, wall_seconds,
        )
    rewards = [(_f(r, "reward") if _f(r, "reward") is not None else float("-inf")) for r in rows]
    best_idx = max(range(len(rows)), key=lambda i: rewards[i])
    best = rows[best_idx]
    # Canonical score over compiled designs (the ones that could ever be deployed).
    compiled = [r for r in rows if r.get("compile_success")]
    canon = max((canonical_score(r) for r in compiled), default=float("-inf"))
    compile_rate = (len(compiled) / len(rows)) if rows else 0.0
    finite = [x for x in rewards if x != float("-inf")]
    mean_reward = sum(finite) / len(finite) if finite else float("-inf")
    return VariantResult(
        iteration=iteration,
        variant=variant,
        n_evals=len(rows),
        best_reward=rewards[best_idx],
        best_canonical=canon,
        mean_reward=mean_reward,
        best_config=str(best.get("config_name") or best.get("kernel_name") or best.get("config") or ""),
        best_max_error=_f(best, "max_error"),
        best_latency=_f(best, "latency_cycles"),
        compile_rate=compile_rate,
        wall_seconds=round(wall_seconds, 1),
    )


def append_sweep_row(result: VariantResult, csv_path: Path = SWEEP_CSV) -> None:
    ensure_dirs()
    row = asdict(result)
    row.pop("extra", None)
    exists = Path(csv_path).exists()
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_variant_sweep(
    run_fn: RunFn,
    variants: list[str],
    iteration: int = 0,
    csv_path: Path = SWEEP_CSV,
    on_result: Callable[[VariantResult], None] | None = None,
) -> list[VariantResult]:
    """Run ``run_fn`` once per variant (env set), summarize, and persist."""
    results: list[VariantResult] = []
    for variant in variants:
        logger.info("=== Sweep iter %d: reward variant %s ===", iteration, variant)
        t0 = time.time()
        with reward_variant_env(variant):
            try:
                rows = run_fn(variant)
            except Exception as exc:  # noqa: BLE001 - one variant must not kill the sweep
                logger.exception("Variant %s failed: %s", variant, exc)
                rows = []
        res = summarize_run(iteration, variant, rows, time.time() - t0)
        results.append(res)
        append_sweep_row(res, csv_path)
        logger.info(
            "Variant %s -> canonical=%.4f best_reward=%.4f compile_rate=%.0f%% (%ds)",
            variant,
            res.best_canonical,
            res.best_reward,
            100 * res.compile_rate,
            res.wall_seconds,
        )
        if on_result is not None:
            on_result(res)
    return results


def rank_variants(results: list[VariantResult]) -> list[VariantResult]:
    """Best-first ranking by the canonical objective."""
    return sorted(results, key=lambda r: r.best_canonical, reverse=True)


def top_variants(results: list[VariantResult], k: int) -> list[str]:
    return [r.variant for r in rank_variants(results)[: max(1, k)]]


def format_leaderboard(results: list[VariantResult]) -> str:
    ranked = rank_variants(results)
    lines = [
        f"{'rank':>4} | {'variant':<18} | {'canonical':>10} | {'own_reward':>10} | "
        f"{'compile%':>8} | {'max_err':>8} | {'config':<24}",
        "-" * 100,
    ]
    for i, r in enumerate(ranked):
        me = f"{r.best_max_error:.4f}" if r.best_max_error is not None else "n/a"
        lines.append(
            f"{i + 1:>4} | {r.variant:<18} | {r.best_canonical:>10.4f} | {r.best_reward:>10.4f} | "
            f"{100 * r.compile_rate:>7.0f}% | {me:>8} | {str(r.best_config)[:24]:<24}"
        )
    return "\n".join(lines)
