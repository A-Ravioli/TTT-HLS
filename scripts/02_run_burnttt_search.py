#!/usr/bin/env python
"""Run the BurnTTT search: baseline + online policy + equal-budget random search.

Writes all evaluations to results/runs.csv and prints a comparison of the best
config found by each method.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from compiler.run_hls import hls_tool_available  # noqa: E402
from paths import RUNS_CSV, get_logger, get_target_part  # noqa: E402
from ttt.search import run_full_search  # noqa: E402

logger = get_logger("burnttt.script.search")


def _summarize(df: pd.DataFrame) -> None:
    print("\n=== BurnTTT search summary ===")
    print(f"Target part: {get_target_part()}   total evaluations: {len(df)}")
    for method in ["default", "random", "burnttt"]:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        valid = sub[sub["compile_success"] == True]  # noqa: E712
        best = valid.sort_values("reward", ascending=False).head(1)
        if best.empty:
            print(f"  {method:8s}: no successful configs")
            continue
        row = best.iloc[0]
        print(
            f"  {method:8s}: best_reward={row['reward']:8.1f}  "
            f"cfg={row['config_name']:24s}  max_err={row['max_error']}  "
            f"lat={row['latency_cycles']}  dsp={row['dsp']}  fits={row['fits_board']}"
        )
    print(f"\nResults written to: {RUNS_CSV}")
    print("Next: streamlit run dashboard/app.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BurnTTT autotuner search")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--candidates-per-round", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--synth", action="store_true", help="Run HLS synthesis per config if a toolchain is present")
    parser.add_argument("--append", action="store_true", help="Append to runs.csv instead of starting fresh")
    args = parser.parse_args()

    logger.info("HLS toolchain available: %s (synth=%s)", hls_tool_available(), args.synth)
    df = run_full_search(
        rounds=args.rounds,
        candidates_per_round=args.candidates_per_round,
        seed=args.seed,
        run_synth=args.synth,
        fresh=not args.append,
    )
    _summarize(df)


if __name__ == "__main__":
    main()
