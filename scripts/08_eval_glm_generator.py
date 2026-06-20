#!/usr/bin/env python
"""Head-to-head: default vs random vs random-forest vs GLM vs GLM+test-time-train.

Runs every method on an equal evaluation budget and writes results/runs.csv for
the dashboard. This is the headline experiment of the realigned project: does an
LLM that *authors* hardware configs (and adapts its weights at test time) beat the
random-forest surrogate baseline?

Without a real GLM (BURN_GLM_MODEL unset / no transformers) the heuristic backend
stands in, so this still runs end-to-end.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from compiler.run_hls import hls_tool_available  # noqa: E402
from glm.serving import load_backend  # noqa: E402
from infra.launch import describe_placement  # noqa: E402
from paths import RUNS_CSV, get_logger, get_target_part  # noqa: E402
from ttt.search import run_full_search  # noqa: E402

logger = get_logger("burnttt.script.eval_glm")

METHODS = ["default", "random", "burnttt", "glm", "glm_ttt"]
LABELS = {
    "default": "Default hls4ml",
    "random": "Random search",
    "burnttt": "Random forest (baseline)",
    "glm": "GLM (frozen)",
    "glm_ttt": "GLM (test-time finetuned)",
}


def _summarize(df: pd.DataFrame) -> None:
    print("\n=== GLM generator evaluation summary ===")
    print(f"Target part: {get_target_part()}   total evaluations: {len(df)}")
    for method in METHODS:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        valid = sub[sub["compile_success"] == True]  # noqa: E712
        best = valid.sort_values("reward", ascending=False).head(1)
        if best.empty:
            print(f"  {LABELS[method]:30s}: no successful configs")
            continue
        row = best.iloc[0]
        print(
            f"  {LABELS[method]:30s}: best_reward={row['reward']:8.1f}  "
            f"cfg={row['config_name']:24s}  max_err={row['max_error']}  "
            f"dsp={row['dsp']}  fits={row['fits_board']}"
        )
    print(f"\nResults written to: {RUNS_CSV}")
    print("Next: streamlit run dashboard/app.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the GLM generator vs baselines")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--candidates-per-round", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--synth", action="store_true", help="Run HLS synthesis per config if a toolchain is present")
    parser.add_argument("--append", action="store_true", help="Append to runs.csv instead of starting fresh")
    parser.add_argument("--no-ttt", action="store_true", help="Skip the test-time-finetuned GLM run")
    args = parser.parse_args()

    backend = load_backend()
    logger.info("GLM backend: %s | %s", backend.name, describe_placement())
    logger.info("HLS toolchain available: %s (synth=%s)", hls_tool_available(), args.synth)

    df = run_full_search(
        rounds=args.rounds,
        candidates_per_round=args.candidates_per_round,
        seed=args.seed,
        run_synth=args.synth,
        fresh=not args.append,
        include_glm=True,
        include_glm_ttt=not args.no_ttt,
    )
    _summarize(df)


if __name__ == "__main__":
    main()
