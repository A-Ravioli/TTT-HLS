#!/usr/bin/env python
"""Test-time-finetune the GLM generator and chart reward-vs-step.

This is the core deliverable of the realignment: within a single run, on a single
(model block, FPGA part) task, GLM's weights are adapted (LoRA when a real model +
GPU are present; the heuristic backend otherwise) on its own synthesis/simulation
feedback, so it authors better hardware each round.

Runs the frozen GLM, the test-time-finetuned GLM, and the random-forest baseline
on an equal evaluation budget, writes results/runs.csv for the dashboard, and
prints the best-reward-over-evaluations curve for each.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from glm.finetune import lora as lora_mod  # noqa: E402
from glm.serving import load_backend  # noqa: E402
from infra.launch import describe_placement  # noqa: E402
from infra import wandb_run  # noqa: E402
from paths import RUNS_CSV, get_logger, get_target_part  # noqa: E402
from ttt.search import run_full_search  # noqa: E402

logger = get_logger("burnttt.script.ttt")

CURVE_METHODS = ["burnttt", "glm", "glm_ttt"]
LABELS = {
    "burnttt": "Random forest (baseline)",
    "glm": "GLM (frozen)",
    "glm_ttt": "GLM (test-time finetuned)",
}


def _print_reward_curves(df: pd.DataFrame) -> None:
    curves = {}
    max_len = 0
    for method in CURVE_METHODS:
        sub = df[df["method"] == method].sort_values("attempt")
        if sub.empty:
            continue
        cur = sub["reward"].cummax().reset_index(drop=True)
        curves[method] = cur
        max_len = max(max_len, len(cur))
    if not curves:
        print("No GLM/baseline rows to chart.")
        return

    print("\n=== Best reward over evaluations (reward-vs-step) ===")
    header = "eval | " + " | ".join(f"{LABELS[m]:>28s}" for m in curves)
    print(header)
    print("-" * len(header))
    for i in range(max_len):
        cells = []
        for m in curves:
            cur = curves[m]
            val = cur.iloc[i] if i < len(cur) else cur.iloc[-1]
            cells.append(f"{val:28.1f}")
        print(f"{i:4d} | " + " | ".join(cells))

    print("\nFinal best reward:")
    for m in curves:
        print(f"  {LABELS[m]:30s}: {curves[m].iloc[-1]:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test-time finetune GLM and chart reward-vs-step")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--candidates-per-round", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--synth", action="store_true")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    backend = load_backend()
    print("=== Test-time finetuning the GLM generator ===")
    print(f"FPGA part        : {get_target_part()}")
    print(f"GLM backend      : {backend.name}")
    print(f"Device placement : {describe_placement()}")
    print(f"Real LoRA path   : {backend.is_llm and lora_mod.peft_available()}")

    run = wandb_run.init_run(
        config={
            "rounds": args.rounds,
            "candidates_per_round": args.candidates_per_round,
            "backend": backend.name,
            "part": get_target_part(),
        }
    )
    if run is not None:
        print(f"wandb run: {run.url}")

    df = run_full_search(
        rounds=args.rounds,
        candidates_per_round=args.candidates_per_round,
        seed=args.seed,
        run_synth=args.synth,
        fresh=not args.append,
        include_glm=True,
        include_glm_ttt=True,
    )
    _print_reward_curves(df)
    print(f"\nResults written to: {RUNS_CSV}")
    print("Next: streamlit run dashboard/app.py")
    wandb_run.finish()


if __name__ == "__main__":
    main()
