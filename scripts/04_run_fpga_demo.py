#!/usr/bin/env python
"""Run the live FPGA demo: stream golden inputs through the board, compare output.

On a real PYNQ board this loads the generated overlay and prints a PASS/FAIL
against the Python golden outputs. Off-board it explains what is missing while
still demonstrating the *software* equivalence (hls4ml bit-accurate prediction)
so there is always something to show.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from compiler.build_hls4ml_project import build_project, compile_project  # noqa: E402
from compiler.deploy_pynq import deploy_and_run, find_bitstream, pynq_available  # noqa: E402
from models.export_model import load_golden, load_model  # noqa: E402
from paths import ARTIFACTS_DIR, BUILD_DIR, get_logger  # noqa: E402
from ttt.config_space import BurnConfig  # noqa: E402

logger = get_logger("burnttt.script.fpga_demo")

BEST_CONFIG_PATH = ARTIFACTS_DIR / "best_config.json"


def _print_vectors(py_out: np.ndarray, hw_out: np.ndarray | None, label: str) -> None:
    np.set_printoptions(precision=4, suppress=True)
    print(f"\nInput[0]        : {load_golden()[0][0]}")
    print(f"Python output[0]: {py_out[0]}")
    if hw_out is not None:
        print(f"{label}[0]: {hw_out[0]}")
        max_err = float(np.max(np.abs(hw_out - py_out)))
        print(f"Max error       : {max_err}")
        print("PASS" if max_err < 0.25 else "FAIL")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FPGA (or software) demo")
    parser.parse_args()

    x, golden = load_golden()
    out_dir = BUILD_DIR / "best"

    bit = find_bitstream(out_dir)
    if pynq_available() and bit is not None:
        logger.info("PYNQ board + bitstream detected; running on hardware.")
        status = deploy_and_run(out_dir, x, golden)
        if status.get("deployed"):
            _print_vectors(golden, status.get("y_fpga"), "FPGA output")
            return
        logger.warning("On-board run did not complete: %s", status.get("reason"))

    # Off-board fallback: demonstrate software (bit-accurate) equivalence.
    print("\n=== No live FPGA available — running software equivalence demo ===")
    print("(pynq_available=%s, bitstream=%s)" % (pynq_available(), bit))

    if BEST_CONFIG_PATH.exists():
        config = BurnConfig.from_dict(json.loads(BEST_CONFIG_PATH.read_text()))
        logger.info("Using best config from %s: %s", BEST_CONFIG_PATH, config.short_name())
    else:
        from ttt.search import DEFAULT_CONFIG

        config = DEFAULT_CONFIG
        logger.info("No best_config.json; using default %s", config.short_name())

    model = load_model()
    hls_model = build_project(model, config, output_dir=BUILD_DIR / "demo")
    if compile_project(hls_model):
        y_hls = hls_model.predict(np.ascontiguousarray(x, dtype=np.float32)).reshape(golden.shape)
        _print_vectors(golden, y_hls, "hls4ml (C-sim) output")
    else:
        print("hls4ml compile failed; cannot run software demo.")

    print("\nTo run on real hardware:")
    print("  1. Build a bitstream on a Vivado machine: python scripts/03_build_best_bitstream.py")
    print("  2. Copy build/best/*.bit and *.hwh to the PYNQ board")
    print("  3. On the board: python scripts/04_run_fpga_demo.py")


if __name__ == "__main__":
    main()
