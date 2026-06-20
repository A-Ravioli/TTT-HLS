#!/usr/bin/env python
"""Collect GLM generation trajectories for a task into data/trajectories/.

Runs the (frozen) GLM generator over the current model + FPGA part, logging every
(config -> feedback -> reward) trajectory. These trajectories are the training
data for test-time finetuning (scripts/07) and can be inspected/replayed offline.

Off-GPU the heuristic backend stands in for GLM; the trajectories are identical in
shape, so the rest of the pipeline is unchanged.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glm.finetune.dataset import to_preference_pairs, to_sft_examples  # noqa: E402
from glm.serving import load_backend  # noqa: E402
from glm.trajectories import TrajectoryStore  # noqa: E402
from paths import get_logger, get_target_part  # noqa: E402
from ttt.search import SearchContext, build_task, run_glm_search  # noqa: E402

logger = get_logger("burnttt.script.collect")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect GLM trajectories for a task")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--candidates-per-round", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--synth", action="store_true")
    args = parser.parse_args()

    logger.info("GLM backend: %s | part: %s", load_backend().name, get_target_part())

    ctx = SearchContext(run_synth=args.synth, cleanup=True)
    task = build_task(ctx.model)
    store = TrajectoryStore(run_name="collect")

    run_glm_search(args.rounds, args.candidates_per_round, seed=args.seed, ctx=ctx, store=store, task=task)

    rows = TrajectoryStore.read(store.path)
    sft = to_sft_examples(task, rows)
    prefs = to_preference_pairs(task, rows)
    print("\n=== Trajectory collection complete ===")
    print(f"Task              : {task.name}")
    print(f"Trajectories saved: {store.path}  ({len(rows)} rows)")
    print(f"SFT examples       : {len(sft)} (high-reward configs)")
    print(f"Preference pairs   : {len(prefs)}")
    print("Next: python scripts/07_ttt_finetune_glm.py")


if __name__ == "__main__":
    main()
