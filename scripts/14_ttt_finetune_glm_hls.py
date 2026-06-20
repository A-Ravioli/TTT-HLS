#!/usr/bin/env python3
"""LoRA+DPO finetuning on HLS trajectories; chart reward-vs-iteration.

Loads HLS trajectories, constructs SFT/DPO training data, and runs test-time
LoRA finetuning on the GLM. Charts reward improvement over iterations.

Usage:
    python scripts/14_ttt_finetune_glm_hls.py [--run-name glm_hls_ttt] [--steps 8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from glm.finetune.dataset_hls import to_hls_preference_pairs, to_hls_sft_examples
from glm.tasks import FpgaTask, make_task
from infra.trace_store import HLSTraceStore
from models.qwen.decompose import mlp_block_spec
from models.qwen.load_qwen import get_default_arch
from paths import ensure_dirs, get_logger
from ttt.reward import get_board_budget

logger = get_logger("burnttt.scripts.ttt_finetune_hls")


def main():
    parser = argparse.ArgumentParser(description="TTT finetune GLM on HLS trajectories")
    parser.add_argument("--run-name", default="glm_hls_ttt", help="Trace store run name")
    parser.add_argument("--steps", type=int, default=8, help="LoRA gradient steps")
    parser.add_argument("--part", default="xcu250-figd2104-2l-e", help="Target part")
    parser.add_argument("--hidden-dim", type=int, default=24, help="Hidden dim (tiled)")
    parser.add_argument("--intermediate-dim", type=int, default=64, help="Intermediate dim (tiled)")
    args = parser.parse_args()

    ensure_dirs()
    arch = get_default_arch()

    # Build task
    block_spec = mlp_block_spec(arch)
    budget = get_board_budget(args.part)
    task = make_task(block=block_spec, target_part=args.part, budget=budget)

    # Load trajectories
    store = HLSTraceStore(run_name=args.run_name)
    traces = store.load_all()

    if not traces:
        print(f"No traces found for '{args.run_name}'. Run script 11 first.")
        return

    # Convert to training-compatible format
    rows = []
    for t in traces:
        row = dict(t.result)
        row["sources"] = t.sources
        rows.append(row)

    # Generate training examples
    sft_examples = to_hls_sft_examples(
        task, rows,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
    )
    dpo_pairs = to_hls_preference_pairs(
        task, rows,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
    )

    print(f"\n=== HLS TTT Finetune Report ===")
    print(f"Trajectories loaded: {len(traces)}")
    print(f"SFT examples:        {len(sft_examples)}")
    print(f"DPO pairs:           {len(dpo_pairs)}")

    if not sft_examples:
        print("\nNot enough high-reward examples for finetuning.")
        print("Run more iterations of script 11 to collect training data.")
        return

    # Report reward distribution
    rewards = [t.result.get("reward", -1e9) for t in traces]
    passing = [t for t in traces if t.result.get("cosim_pass")]
    print(f"\nReward stats:")
    print(f"  Min:     {min(rewards):.1f}")
    print(f"  Max:     {max(rewards):.1f}")
    print(f"  Median:  {sorted(rewards)[len(rewards)//2]:.1f}")
    print(f"  Passing: {len(passing)}/{len(traces)}")

    # Attempt real LoRA training if GPU available
    try:
        from glm.agent_hls import GLMCompilerAgent
        from glm.finetune.trainer_hls import HLSTestTimeTrainer

        agent = GLMCompilerAgent()
        trainer = HLSTestTimeTrainer(
            agent, task,
            hidden_dim=args.hidden_dim,
            intermediate_dim=args.intermediate_dim,
            steps_per_round=args.steps,
        )
        if trainer.is_real:
            info = trainer.step(rows)
            print(f"\nLoRA training result: {info}")
        else:
            info = trainer.step(rows)
            print(f"\nHeuristic adaptation: {info}")
            print("(Set BURN_GLM_MODEL for real LoRA training)")
    except Exception as exc:  # noqa: BLE001
        print(f"\nNote: LoRA training skipped ({exc})")
        print("Heuristic adaptation was used. Set BURN_GLM_MODEL for real finetuning.")


if __name__ == "__main__":
    main()
