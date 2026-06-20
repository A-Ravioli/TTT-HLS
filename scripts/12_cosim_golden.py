#!/usr/bin/env python3
"""Batch cosim all kernels in a run vs Qwen golden reference.

Reads HLS trajectories from the trace store and re-runs cosim against fresh
golden vectors. Useful for validating that stored kernels still pass after
any golden-model changes.

Usage:
    python scripts/12_cosim_golden.py [--run-name glm_hls_ttt] [--max-error 0.01]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compiler.golden import generate_golden_from_keras
from infra.trace_store import HLSTraceStore
from models.qwen.blocks import build_mlp_keras, tile_dims
from models.qwen.load_qwen import get_default_arch
from paths import BUILD_DIR, ensure_dirs, get_logger
from ttt.config_space import KernelBundle
from ttt.evaluate_hls import evaluate_hls

logger = get_logger("burnttt.scripts.cosim_golden")


def main():
    parser = argparse.ArgumentParser(description="Batch cosim vs golden")
    parser.add_argument("--run-name", default="glm_hls_ttt", help="Trace store run name")
    parser.add_argument("--tile-div", type=int, default=64, help="Tile divisor")
    parser.add_argument("--max-error", type=float, default=0.01, help="Cosim threshold")
    parser.add_argument("--top-k", type=int, default=10, help="Only check top-k by reward")
    args = parser.parse_args()

    ensure_dirs()
    arch = get_default_arch()

    # Build golden
    model, _ = build_mlp_keras(arch, args.tile_div)
    golden = generate_golden_from_keras(model, n_samples=64)

    # Load traces
    store = HLSTraceStore(run_name=args.run_name)
    traces = store.load_top_k(args.top_k)

    if not traces:
        logger.info("No traces found for run '%s'", args.run_name)
        print(f"No traces found. Run scripts/11_glm_author_hls.py first.")
        return

    logger.info("Re-running cosim on top %d kernels from '%s'", len(traces), args.run_name)

    results = []
    for trace in traces:
        bundle = KernelBundle.from_dict({
            "sources": trace.sources,
            **trace.result,
        })
        result = evaluate_hls(bundle, golden, max_error_threshold=args.max_error, cleanup=True)
        results.append({
            "kernel": trace.kernel_name,
            "original_reward": trace.result.get("reward"),
            "re_cosim_pass": result.get("cosim_pass"),
            "re_max_error": result.get("max_error"),
            "re_reward": result.get("reward"),
        })

    # Summary
    print(f"\nCosim validation results ({len(results)} kernels):")
    print("-" * 70)
    passed = sum(1 for r in results if r["re_cosim_pass"])
    for r in results:
        status = "PASS" if r["re_cosim_pass"] else "FAIL"
        print(f"  [{status}] {r['kernel']}: max_error={r['re_max_error']:.6f}" if r["re_max_error"] else
              f"  [{status}] {r['kernel']}: error=N/A")
    print(f"\n{passed}/{len(results)} kernels pass cosim (threshold={args.max_error})")


if __name__ == "__main__":
    main()
