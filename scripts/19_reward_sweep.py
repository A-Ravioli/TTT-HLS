#!/usr/bin/env python
"""Train the synthesizer under different reward variants and iterate on the best.

This is the "spin up training runs with different rewards, constantly iterating"
driver. For each reward variant (see :mod:`ttt.reward_variants`) it runs a
test-time-finetuned GLM search, compares every run on a single canonical
objective, and -- in ``--loop`` mode -- concentrates the next iteration's compute
on the best-performing rewards with escalated round budgets.

Backends (auto-detected via .env / BURN_GLM_BACKEND):
  * prime     -> GLM-5.2 via Prime Intellect Inference API (cheap; generation +
                 heuristic-style adaptation). Real frontier synthesizer, no LoRA.
  * hf        -> local HuggingFace weights on a GPU pod (real LoRA test-time train).
  * heuristic -> offline deterministic stand-in (free; for plumbing/CI).

Examples:
  # One sweep over all variants with the configured backend:
  python scripts/19_reward_sweep.py --rounds 6 --candidates-per-round 3

  # Continuously iterate, keeping the best 2 rewards and growing the budget:
  python scripts/19_reward_sweep.py --loop 0 --top-k 2 --escalate-rounds 2
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paths  # noqa: F401,E402 -- load .env
from paths import RESULTS_DIR, get_logger, get_target_part  # noqa: E402

logger = get_logger("burnttt.script.reward_sweep")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reward-variant TTT sweep with iterative refinement")
    parser.add_argument("--variants", default="", help="Comma list (default: all non-legacy variants)")
    parser.add_argument("--include-legacy", action="store_true", help="Also run the legacy raw-count reward")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--candidates-per-round", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--synth", action="store_true", help="Run real HLS synthesis if toolchain present")
    parser.add_argument("--loop", type=int, default=1, help="Iterations (0 = run forever)")
    parser.add_argument("--top-k", type=int, default=2, help="Keep this many best variants after iter 0")
    parser.add_argument("--escalate-rounds", type=int, default=1, help="Extra rounds added each iteration")
    parser.add_argument("--max-rounds", type=int, default=24, help="Cap on escalated rounds")
    args = parser.parse_args()

    # Heavy imports happen here so the orchestration module stays import-light.
    from glm.serving import load_backend
    from infra import reward_sweep as sweep
    from infra import wandb_run
    from ttt.search import SearchContext, build_task, run_glm_ttt_search

    backend = load_backend()
    part = get_target_part()
    variants = (
        [v.strip() for v in args.variants.split(",") if v.strip()]
        if args.variants
        else sweep.sweep_variants(include_legacy=args.include_legacy)
    )
    wandb_base = os.environ.get("WANDB_RUN_NAME", "reward-sweep")

    print("=== Reward-variant TTT sweep ===")
    print(f"FPGA part   : {part}")
    print(f"GLM backend : {backend.name}  (real LoRA: {backend.is_llm and backend.name == 'hf'})")
    print(f"Variants    : {', '.join(variants)}")
    print(f"Canonical   : {sweep.CANONICAL_VARIANT} (all runs ranked on this objective)")
    print(f"Loop        : {'forever' if args.loop == 0 else args.loop} iteration(s), top-k={args.top_k}")

    # Shared model + golden vectors; the reward variant (read at eval time from the
    # env) is what changes between runs, not the task.
    ctx = SearchContext(run_synth=args.synth, cleanup=True)
    task = build_task(ctx.model, part=part)

    history: list[sweep.VariantResult] = []
    iteration = 0
    active_variants = list(variants)

    while True:
        rounds = min(args.max_rounds, args.rounds + iteration * args.escalate_rounds)

        def run_fn(variant: str, _rounds: int = rounds) -> list[dict]:
            # Per-variant wandb run so curves are separable in the dashboard.
            wandb_run.finish()
            os.environ["WANDB_RUN_NAME"] = f"{wandb_base}-{variant}-it{iteration}"
            wandb_run.init_run(
                config={
                    "reward_variant": variant,
                    "iteration": iteration,
                    "rounds": _rounds,
                    "candidates_per_round": args.candidates_per_round,
                    "backend": backend.name,
                    "part": part,
                }
            )
            rows = run_glm_ttt_search(
                rounds=_rounds,
                candidates_per_round=args.candidates_per_round,
                seed=args.seed + iteration,
                ctx=ctx,
                task=task,
            )
            wandb_run.finish()
            return rows

        results = sweep.run_variant_sweep(run_fn, active_variants, iteration=iteration)
        history.extend(results)

        print(f"\n=== Iteration {iteration} leaderboard (rounds={rounds}) ===")
        print(sweep.format_leaderboard(results))
        print(f"\nSweep log: {sweep.SWEEP_CSV}")

        ranked = sweep.rank_variants(results)
        if ranked:
            best = ranked[0]
            print(f"Best reward variant so far: {best.variant} (canonical={best.best_canonical:.4f})")

        iteration += 1
        if args.loop != 0 and iteration >= args.loop:
            break
        # Concentrate the next iteration's compute on the best rewards.
        active_variants = sweep.top_variants(results, args.top_k)
        logger.info("Next iteration focuses on: %s", active_variants)

    print("\n=== Final cross-iteration ranking ===")
    print(sweep.format_leaderboard(history))


if __name__ == "__main__":
    main()
