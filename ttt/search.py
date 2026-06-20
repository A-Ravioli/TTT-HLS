"""Search orchestration: baseline, random search, and the BurnTTT policy loop.

All three write rows into ``results/runs.csv`` with a ``method`` column so the
dashboard can compare them on equal evaluation budgets.
"""

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tensorflow import keras

from models.export_model import load_golden, load_model
from paths import RUNS_CSV, ensure_dirs, get_logger
from ttt.config_space import BurnConfig, sample_random_configs, seed_configs
from ttt.evaluate_config import evaluate_config
from ttt.online_policy import OnlineTTTPolicy

logger = get_logger("burnttt.search")

# hls4ml-style default config (plan.md baseline): ap_fixed<16,6>, reuse=1, Latency.
DEFAULT_CONFIG = BurnConfig(16, 16, 6, 1, 1, "Latency")

# Preferred leading columns in the CSV for readability.
LEADING_COLUMNS = [
    "method",
    "attempt",
    "round",
    "config_name",
    "weight_bits",
    "activation_bits",
    "int_bits",
    "reuse_dense_1",
    "reuse_dense_2",
    "strategy",
    "compile_success",
    "sim_success",
    "synth_success",
    "max_error",
    "mean_error",
    "latency_cycles",
    "ii",
    "dsp",
    "lut",
    "ff",
    "bram",
    "fits_board",
    "estimated_hw",
    "reward",
]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def reset_results(csv_path: Path = RUNS_CSV) -> None:
    """Delete the results CSV so a run starts fresh."""
    ensure_dirs()
    if csv_path.exists():
        csv_path.unlink()
        logger.info("Cleared %s", csv_path)


def load_results(csv_path: Path = RUNS_CSV) -> pd.DataFrame:
    if Path(csv_path).exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def append_results(rows: list[dict[str, Any]], csv_path: Path = RUNS_CSV) -> pd.DataFrame:
    """Append result rows to the CSV, returning the full updated frame."""
    ensure_dirs()
    new = pd.DataFrame(rows)
    existing = load_results(csv_path)
    combined = pd.concat([existing, new], ignore_index=True) if not existing.empty else new

    # Order columns: known leading columns first, then everything else.
    ordered = [c for c in LEADING_COLUMNS if c in combined.columns]
    rest = [c for c in combined.columns if c not in ordered]
    combined = combined[ordered + rest]
    combined.to_csv(csv_path, index=False)
    return combined


class SearchContext:
    """Shared model + golden vectors so each eval doesn't reload from disk."""

    def __init__(self, run_synth: bool = False, cleanup: bool = True):
        self.model: keras.Model = load_model()
        self.x_test, self.golden = load_golden()
        self.run_synth = run_synth
        self.cleanup = cleanup

    def evaluate(self, config: BurnConfig, cleanup: bool | None = None) -> dict[str, Any]:
        return evaluate_config(
            config,
            model=self.model,
            x_test=self.x_test,
            golden=self.golden,
            run_synth=self.run_synth,
            cleanup=self.cleanup if cleanup is None else cleanup,
        )


def _record(result: dict[str, Any], method: str, attempt: int, round_idx: int | None) -> dict[str, Any]:
    row = dict(result)
    row["method"] = method
    row["attempt"] = attempt
    row["round"] = round_idx
    row["timestamp"] = _now()
    return row


def run_baseline(ctx: SearchContext | None = None) -> dict[str, Any]:
    """Evaluate the single hls4ml default config (keeps its project on disk)."""
    ctx = ctx or SearchContext()
    logger.info("=== Baseline: default hls4ml config %s ===", DEFAULT_CONFIG.short_name())
    result = ctx.evaluate(DEFAULT_CONFIG, cleanup=False)
    return _record(result, "default", attempt=0, round_idx=None)


def run_random_search(n_evals: int, seed: int = 7, ctx: SearchContext | None = None) -> list[dict[str, Any]]:
    """Random search baseline with a fixed evaluation budget."""
    ctx = ctx or SearchContext()
    rng = random.Random(seed)
    configs = sample_random_configs(n_evals, rng)
    rows: list[dict[str, Any]] = []
    logger.info("=== Random search: %d evaluations ===", n_evals)
    for i, cfg in enumerate(configs):
        result = ctx.evaluate(cfg)
        rows.append(_record(result, "random", attempt=i, round_idx=None))
        logger.info("[random %d/%d] %s reward=%.1f", i + 1, n_evals, cfg.short_name(), result["reward"])
    return rows


def run_burnttt_search(
    rounds: int,
    candidates_per_round: int,
    seed: int = 0,
    ctx: SearchContext | None = None,
) -> list[dict[str, Any]]:
    """The online TTT loop: seed -> (fit -> propose -> evaluate) x rounds."""
    ctx = ctx or SearchContext()
    policy = OnlineTTTPolicy(random_state=seed)

    history_configs: list[BurnConfig] = []
    history_rewards: list[float] = []
    tried: set[str] = set()
    rows: list[dict[str, Any]] = []
    attempt = 0

    logger.info("=== BurnTTT online search: %d rounds x %d candidates ===", rounds, candidates_per_round)

    # Round 0: evaluate the hand-picked seed configs.
    for cfg in seed_configs():
        result = ctx.evaluate(cfg)
        history_configs.append(cfg)
        history_rewards.append(result["reward"])
        tried.add(cfg.short_name())
        rows.append(_record(result, "burnttt", attempt=attempt, round_idx=0))
        logger.info("[burnttt seed] %s reward=%.1f", cfg.short_name(), result["reward"])
        attempt += 1

    # Online rounds.
    for r in range(1, rounds + 1):
        policy.fit(history_configs, history_rewards)
        proposals = policy.propose(n=candidates_per_round, exclude=tried)
        logger.info("Round %d proposals: %s", r, [c.short_name() for c in proposals])
        for cfg in proposals:
            result = ctx.evaluate(cfg)
            history_configs.append(cfg)
            history_rewards.append(result["reward"])
            tried.add(cfg.short_name())
            rows.append(_record(result, "burnttt", attempt=attempt, round_idx=r))
            logger.info("[burnttt r%d] %s reward=%.1f", r, cfg.short_name(), result["reward"])
            attempt += 1

    return rows


def run_full_search(
    rounds: int = 4,
    candidates_per_round: int = 3,
    seed: int = 0,
    run_synth: bool = False,
    fresh: bool = True,
) -> pd.DataFrame:
    """Run baseline + BurnTTT + equal-budget random search; write runs.csv."""
    if fresh:
        reset_results()
    ctx = SearchContext(run_synth=run_synth, cleanup=True)

    all_rows: list[dict[str, Any]] = []
    all_rows.append(run_baseline(ctx))

    burnttt_rows = run_burnttt_search(rounds, candidates_per_round, seed=seed, ctx=ctx)
    all_rows.extend(burnttt_rows)

    # Equal evaluation budget for a fair comparison.
    budget = len(burnttt_rows)
    all_rows.extend(run_random_search(budget, seed=seed + 100, ctx=ctx))

    df = append_results(all_rows)
    logger.info("Wrote %d rows to %s", len(all_rows), RUNS_CSV)
    return df
