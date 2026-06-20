#!/usr/bin/env python3
"""Parallel Vivado P&R for cosim-passing kernels.

Loads top kernels from the trace store and runs Vivado synthesis + P&R in
parallel (or sequentially if Vivado unavailable). Reports timing, resources,
and tokens/sec.

Usage:
    python scripts/13_synth_farm.py [--run-name glm_hls_ttt] [--workers 4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compiler.golden import generate_golden_from_keras
from infra.synth_farm import SynthFarm
from infra.trace_store import HLSTraceStore
from models.qwen.blocks import build_mlp_keras, tile_dims
from models.qwen.load_qwen import get_default_arch
from paths import ensure_dirs, get_logger
from ttt.config_space import KernelBundle
from ttt.evaluate_hls import evaluate_hls

logger = get_logger("burnttt.scripts.synth_farm")


def main():
    parser = argparse.ArgumentParser(description="Parallel synthesis farm")
    parser.add_argument("--run-name", default="glm_hls_ttt", help="Trace store run name")
    parser.add_argument("--workers", type=int, default=4, help="Max parallel workers")
    parser.add_argument("--top-k", type=int, default=5, help="Synthesize top-k by reward")
    parser.add_argument("--tile-div", type=int, default=64, help="Tile divisor")
    args = parser.parse_args()

    ensure_dirs()
    arch = get_default_arch()

    # Build golden for evaluation
    model, _ = build_mlp_keras(arch, args.tile_div)
    golden = generate_golden_from_keras(model, n_samples=64)

    # Load top cosim-passing traces
    store = HLSTraceStore(run_name=args.run_name)
    traces = [t for t in store.load_top_k(args.top_k * 2) if t.result.get("cosim_pass")]
    traces = traces[: args.top_k]

    if not traces:
        logger.info("No cosim-passing traces found for '%s'", args.run_name)
        print("No cosim-passing kernels to synthesize. Run script 11 first.")
        return

    logger.info("Submitting %d kernels to synth farm (workers=%d)", len(traces), args.workers)

    # Set up farm
    farm = SynthFarm(max_workers=args.workers)

    for trace in traces:
        farm.submit(
            bundle_dict={
                "sources": trace.sources,
                "part": trace.result.get("part", "xcu250-figd2104-2l-e"),
                **trace.result,
            },
            kernel_name=trace.kernel_name,
            priority=trace.result.get("reward", 0),
        )

    # Run synthesis (with Vivado flag)
    def eval_fn(bundle_dict):
        bundle = KernelBundle.from_dict(bundle_dict)
        return evaluate_hls(bundle, golden, run_vivado=True, cleanup=True)

    farm.run_all(eval_fn, parallel=False)  # Sequential for safety

    # Report
    summary = farm.summary()
    print(f"\n=== Synth Farm Results ===")
    print(f"Total jobs: {summary['total_jobs']}")
    print(f"Completed:  {summary['completed']}")
    print(f"Failed:     {summary['failed']}")
    print(f"Best:       {summary['best_kernel']} (reward={summary['best_reward']})")

    print(f"\nDetailed results:")
    for job in farm.get_results():
        r = job.result or {}
        print(
            f"  {job.kernel_name}: timing={r.get('timing_met')} "
            f"tps={r.get('tokens_per_sec', 'n/a')} "
            f"dsp={r.get('dsp')} lut={r.get('lut')} "
            f"reward={r.get('reward', -1e9):.1f}"
        )


if __name__ == "__main__":
    main()
