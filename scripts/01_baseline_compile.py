#!/usr/bin/env python
"""Compile the model once with the hls4ml default config (the baseline).

Generates the HLS project under build/baseline, runs bit-accurate prediction
against the golden outputs, parses/synthesizes if a toolchain is available, and
prints a one-line summary. This is the "software-level burned model" proof.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compiler.run_hls import hls_tool_available  # noqa: E402
from paths import BUILD_DIR, get_logger, get_target_part  # noqa: E402
from ttt.evaluate_config import evaluate_config  # noqa: E402
from ttt.search import DEFAULT_CONFIG  # noqa: E402

logger = get_logger("burnttt.script.baseline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline hls4ml compile")
    parser.add_argument("--synth", action="store_true", help="Run HLS synthesis if a toolchain is present")
    args = parser.parse_args()

    out_dir = BUILD_DIR / "baseline"
    logger.info("Target part: %s", get_target_part())
    logger.info("HLS toolchain available: %s", hls_tool_available())
    logger.info("Baseline config: %s", DEFAULT_CONFIG.short_name())

    result = evaluate_config(
        DEFAULT_CONFIG,
        output_dir=out_dir,
        run_synth=args.synth,
        cleanup=False,
    )

    logger.info("Baseline result:\n%s", json.dumps(result, indent=2, default=str))
    print("\n=== Baseline summary ===")
    print(f"  compile_success : {result['compile_success']}")
    print(f"  max_error       : {result['max_error']}")
    print(f"  mean_error      : {result['mean_error']}")
    print(f"  latency_cycles  : {result['latency_cycles']}  (estimated={result['estimated_hw']})")
    print(f"  dsp/lut/ff/bram : {result['dsp']}/{result['lut']}/{result['ff']}/{result['bram']}")
    print(f"  reward          : {result['reward']:.1f}")
    print(f"  project dir     : {result['output_dir']}")
    print("\nNext: python scripts/02_run_burnttt_search.py --rounds 3 --candidates-per-round 3")


if __name__ == "__main__":
    main()
