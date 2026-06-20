#!/usr/bin/env python3
"""GLM propose/repair loop on custom HLS (single block) with LoRA TTT.

This is the main Phase 4 script: the GLM compiler agent iteratively authors
and refines a SwiGLU MLP HLS kernel, with test-time LoRA finetuning between
rounds on high-reward trajectories.

Usage:
    python scripts/11_glm_author_hls.py [--rounds 6] [--tile-div 64] [--run-vivado]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compiler.golden import generate_golden_from_keras
from compiler.kernel_lib.swiglu_mlp import SwiGLUConfig, generate_full_bundle
from glm.agent_hls import GLMCompilerAgent, result_to_hls_history_row
from glm.finetune.trainer_hls import HLSTestTimeTrainer
from glm.tasks import FpgaTask, make_task
from infra import wandb_run
from infra.trace_store import HLSTraceStore
from models.qwen.blocks import build_mlp_keras, tile_dims
from models.qwen.decompose import mlp_block_spec
from models.qwen.load_qwen import get_default_arch
from paths import ensure_dirs, get_logger
from ttt.config_space import KernelBundle
from ttt.evaluate_hls import evaluate_hls
from ttt.reward import get_board_budget

logger = get_logger("burnttt.scripts.glm_author_hls")


def main():
    parser = argparse.ArgumentParser(description="GLM HLS author loop with TTT")
    parser.add_argument("--rounds", type=int, default=6, help="Number of propose/repair rounds")
    parser.add_argument("--tile-div", type=int, default=64, help="Tile divisor for Qwen dims")
    parser.add_argument("--part", default="xcu250-figd2104-2l-e", help="Target FPGA part")
    parser.add_argument("--max-error", type=float, default=0.01, help="Max cosim error threshold")
    parser.add_argument("--run-vivado", action="store_true", help="Run Vivado P&R (requires toolchain)")
    parser.add_argument("--no-ttt", action="store_true", help="Disable test-time training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    ensure_dirs()
    wandb_run.init_run(
        config={
            "rounds": args.rounds,
            "part": args.part,
            "max_error": args.max_error,
            "tile_div": args.tile_div,
        }
    )
    arch = get_default_arch()
    dims = tile_dims(arch, args.tile_div)

    logger.info("=== GLM HLS Author Loop ===")
    logger.info("Target: Qwen SwiGLU MLP (hidden=%d, inter=%d) on %s", dims.hidden, dims.intermediate, args.part)

    # Build task
    block_spec = mlp_block_spec(arch)
    budget = get_board_budget(args.part)
    task = make_task(block=block_spec, target_part=args.part, budget=budget, max_error_threshold=args.max_error)

    # Build golden reference (using the Keras tiled model)
    model, _ = build_mlp_keras(arch, args.tile_div)
    golden = generate_golden_from_keras(model, n_samples=64)

    # Initialize agent and trainer
    agent = GLMCompilerAgent(seed=args.seed)
    method = "glm_hls_ttt" if not args.no_ttt else "glm_hls"
    trainer = None
    if not args.no_ttt:
        trainer = HLSTestTimeTrainer(
            agent, task,
            hidden_dim=dims.hidden,
            intermediate_dim=dims.intermediate,
        )

    store = HLSTraceStore(run_name=method)

    # Seed bundle
    seed_cfg = SwiGLUConfig(hidden_dim=dims.hidden, intermediate_dim=dims.intermediate)
    seed_sources = generate_full_bundle(seed_cfg)
    seed_bundle = KernelBundle(
        sources=seed_sources,
        hidden_dim=dims.hidden,
        intermediate_dim=dims.intermediate,
        part=args.part,
    )

    # Main loop
    history: list[dict] = []
    best_result: dict | None = None
    best_bundle: KernelBundle | None = None

    for r in range(args.rounds):
        logger.info("--- Round %d/%d ---", r + 1, args.rounds)

        # Propose
        bundle = agent.propose(
            task, history,
            hidden_dim=dims.hidden,
            intermediate_dim=dims.intermediate,
            seed_bundle=seed_bundle if r == 0 else best_bundle,
        )
        logger.info("Proposed: %s", bundle.short_name())

        # Evaluate
        result = evaluate_hls(
            bundle, golden,
            run_vivado=args.run_vivado,
            max_error_threshold=args.max_error,
            cleanup=False,
        )
        history.append(result_to_hls_history_row(result))
        store.append(
            task_name=task.name,
            kernel_name=bundle.short_name(),
            sources=bundle.sources,
            result=result,
            round_idx=r,
            method=method,
        )
        wandb_run.log_eval(len(history), method, result, round_idx=r)

        # Update best
        if best_result is None or result.get("reward", -1e9) > best_result.get("reward", -1e9):
            best_result = result
            best_bundle = bundle

        # Repair loop (if needed)
        if not result.get("cosim_pass"):
            for repair_attempt in range(agent.max_repair_attempts):
                logger.info("Repair attempt %d (error_type: %s)", repair_attempt + 1,
                            "compile" if not result.get("hls_compile_success") else "cosim")
                repaired = agent.repair(task, bundle, result)
                if repaired is None:
                    break
                result = evaluate_hls(
                    repaired, golden,
                    run_vivado=args.run_vivado,
                    max_error_threshold=args.max_error,
                    cleanup=False,
                )
                history.append(result_to_hls_history_row(result))
                store.append(
                    task_name=task.name,
                    kernel_name=repaired.short_name(),
                    sources=repaired.sources,
                    result=result,
                    round_idx=r,
                    method=f"{method}_repair",
                )
                if result.get("reward", -1e9) > best_result.get("reward", -1e9):
                    best_result = result
                    best_bundle = repaired
                if result.get("cosim_pass"):
                    bundle = repaired
                    break

        # Iterate on passing kernel
        if result.get("cosim_pass") and best_bundle:
            improved = agent.iterate(task, best_bundle, result, best_result)
            if improved.short_name() != best_bundle.short_name():
                iter_result = evaluate_hls(
                    improved, golden,
                    run_vivado=args.run_vivado,
                    max_error_threshold=args.max_error,
                    cleanup=False,
                )
                history.append(result_to_hls_history_row(iter_result))
                store.append(
                    task_name=task.name,
                    kernel_name=improved.short_name(),
                    sources=improved.sources,
                    result=iter_result,
                    round_idx=r,
                    method=f"{method}_iterate",
                )
                if iter_result.get("reward", -1e9) > best_result.get("reward", -1e9):
                    best_result = iter_result
                    best_bundle = improved

        # TTT step
        if trainer is not None and len(history) >= 2:
            ttt_info = trainer.step(history, round_idx=r)
            wandb_run.log_ttt_step(len(history), ttt_info)
            logger.info("TTT step: %s", ttt_info)

    if trainer is not None:
        trainer.save_adapter()

    # Summary
    logger.info("=== GLM HLS Loop Complete ===")
    logger.info("Rounds: %d, Total evaluations: %d", args.rounds, len(history))
    if best_result:
        logger.info("Best kernel: %s", best_bundle.short_name() if best_bundle else "none")
        logger.info("  cosim_pass: %s", best_result.get("cosim_pass"))
        logger.info("  max_error:  %s", best_result.get("max_error"))
        logger.info("  reward:     %.1f", best_result.get("reward", -1e9))
        logger.info("  tps:        %s", best_result.get("tokens_per_sec"))
    logger.info("Traces stored in %s", store._path)
    wandb_run.finish()


if __name__ == "__main__":
    main()
