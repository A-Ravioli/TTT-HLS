#!/usr/bin/env python3
"""Comparison table: custom HLS vs hls4ml config TTT vs baselines → runs.csv.

Evaluates the best custom HLS kernel against the best hls4ml config-TTT result
and generates a comparison table. This is the Phase 4 "done" criterion:
custom HLS must beat best hls4ml+TTT on throughput at equal accuracy.

Usage:
    python scripts/15_eval_hls_vs_hls4ml.py [--run-name glm_hls_ttt] [--tile-div 64]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from infra.trace_store import HLSTraceStore
from paths import RESULTS_DIR, ensure_dirs, get_logger
from ttt.reward_hls import reward_hls_comparison

logger = get_logger("burnttt.scripts.eval_hls_vs_hls4ml")

HLS_RUNS_CSV = RESULTS_DIR / "hls_comparison.csv"


def main():
    parser = argparse.ArgumentParser(description="Compare custom HLS vs hls4ml")
    parser.add_argument("--hls-run", default="glm_hls_ttt", help="HLS trace store run")
    parser.add_argument("--config-csv", default=None, help="Path to config-mode runs.csv")
    args = parser.parse_args()

    ensure_dirs()

    # Load best HLS result
    hls_store = HLSTraceStore(run_name=args.hls_run)
    hls_traces = hls_store.load_top_k(1)

    if not hls_traces:
        print(f"No HLS traces found for '{args.hls_run}'. Run script 11 first.")
        return

    best_hls = hls_traces[0].result

    # Load best hls4ml config result
    config_csv = Path(args.config_csv) if args.config_csv else RESULTS_DIR / "runs.csv"
    best_hls4ml = None
    if config_csv.exists():
        df = pd.read_csv(config_csv)
        # Get best GLM+TTT result
        ttt_rows = df[df["method"] == "glm_ttt"] if "method" in df.columns else df
        if not ttt_rows.empty:
            best_idx = ttt_rows["reward"].idxmax()
            best_hls4ml = ttt_rows.loc[best_idx].to_dict()

    # Comparison
    print("\n" + "=" * 70)
    print(" Phase 4 Evaluation: Custom HLS vs hls4ml Config TTT")
    print("=" * 70)

    print(f"\n--- Best Custom HLS (from '{args.hls_run}') ---")
    _print_result(best_hls, "HLS")

    if best_hls4ml:
        print(f"\n--- Best hls4ml Config TTT ---")
        _print_result(best_hls4ml, "hls4ml")

        comparison = reward_hls_comparison(best_hls, best_hls4ml)
        print(f"\n--- Comparison ---")
        print(f"  HLS tokens/sec:    {comparison['hls_tokens_per_sec']:.1f}")
        print(f"  hls4ml tokens/sec: {comparison['hls4ml_tokens_per_sec']:.1f}")
        print(f"  Speedup:           {comparison['speedup']:.2f}x")
        print(f"  HLS wins:          {'YES' if comparison['hls_wins'] else 'NO'}")

        # Write comparison CSV
        rows = [
            {"method": "custom_hls_ttt", "kernel": hls_traces[0].kernel_name, **best_hls},
            {"method": "hls4ml_config_ttt", **best_hls4ml},
        ]
        df_out = pd.DataFrame(rows)
        df_out.to_csv(HLS_RUNS_CSV, index=False)
        print(f"\n  Results written to {HLS_RUNS_CSV}")

        # Phase 4 done check
        if comparison["hls_wins"] and comparison.get("hls_cosim_pass"):
            print(f"\n  *** PHASE 4 CRITERION MET: Custom HLS beats hls4ml+TTT ***")
        else:
            print(f"\n  Phase 4 criterion NOT yet met. Continue iterating.")
    else:
        print(f"\n  No hls4ml config results found at {config_csv}")
        print("  Run scripts/08_eval_glm_generator.py first for comparison baseline.")

        # Still write HLS results
        rows = [{"method": "custom_hls_ttt", "kernel": hls_traces[0].kernel_name, **best_hls}]
        df_out = pd.DataFrame(rows)
        df_out.to_csv(HLS_RUNS_CSV, index=False)
        print(f"\n  HLS results written to {HLS_RUNS_CSV}")


def _print_result(result: dict, label: str):
    keys = [
        ("cosim_pass", "Cosim pass"),
        ("timing_met", "Timing met"),
        ("max_error", "Max error"),
        ("tokens_per_sec", "Tokens/sec"),
        ("latency_cycles", "Latency (cycles)"),
        ("dsp", "DSPs"),
        ("lut", "LUTs"),
        ("bram", "BRAMs"),
        ("reward", "Reward"),
    ]
    for key, label_str in keys:
        val = result.get(key)
        if val is not None:
            if isinstance(val, float):
                print(f"  {label_str:20s}: {val:.4g}")
            else:
                print(f"  {label_str:20s}: {val}")


if __name__ == "__main__":
    main()
