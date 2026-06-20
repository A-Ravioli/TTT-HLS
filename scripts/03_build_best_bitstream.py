#!/usr/bin/env python
"""Pick the best config from results/runs.csv and build its HLS project.

If a Vivado/Vitis toolchain (and the VivadoAccelerator backend) is available it
attempts a full bitstream build for the target board. Otherwise it still emits
the complete, synthesizable HLS project under build/best and explains exactly
what is needed to produce a .bit on a machine with the toolchain.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from compiler.build_hls4ml_project import build_project  # noqa: E402
from compiler.deploy_pynq import find_bitstream  # noqa: E402
from compiler.make_hls_config import make_hls_config  # noqa: E402
from compiler.run_hls import detect_hls_tool, hls_tool_available  # noqa: E402
from models.export_model import load_model  # noqa: E402
from paths import ARTIFACTS_DIR, BUILD_DIR, RUNS_CSV, ensure_dirs, get_logger, get_target_part  # noqa: E402
from ttt.config_space import BurnConfig  # noqa: E402

logger = get_logger("burnttt.script.bitstream")

BEST_CONFIG_PATH = ARTIFACTS_DIR / "best_config.json"


def pick_best_config(df: pd.DataFrame) -> tuple[BurnConfig, dict]:
    """Return the best fitting, compiling config (highest reward)."""
    valid = df[df["compile_success"] == True]  # noqa: E712
    if "fits_board" in valid.columns:
        fitting = valid[valid["fits_board"] == True]  # noqa: E712
        if not fitting.empty:
            valid = fitting
    if valid.empty:
        raise RuntimeError("No successful configs in runs.csv. Run scripts/02 first.")
    best = valid.sort_values("reward", ascending=False).iloc[0]
    config = BurnConfig.from_dict(best.to_dict())
    return config, best.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build best config bitstream/project")
    parser.add_argument("--board", default=os.environ.get("BURN_BOARD", "pynq-z2"))
    args = parser.parse_args()

    if not RUNS_CSV.exists():
        logger.error("No results at %s. Run scripts/02_run_burnttt_search.py first.", RUNS_CSV)
        sys.exit(1)

    df = pd.read_csv(RUNS_CSV)
    config, best_row = pick_best_config(df)
    ensure_dirs()
    logger.info("Best config: %s (reward=%.1f)", config.short_name(), best_row.get("reward", float("nan")))

    BEST_CONFIG_PATH.write_text(json.dumps(config.to_dict(), indent=2))
    logger.info("Saved best config -> %s", BEST_CONFIG_PATH)

    model = load_model()
    out_dir = BUILD_DIR / "best"

    if hls_tool_available():
        logger.info("HLS toolchain detected at %s; attempting bitstream build for board=%s", detect_hls_tool(), args.board)
        try:
            import hls4ml  # noqa: E402

            hls_config = make_hls_config(model, config)
            hls_model = hls4ml.converters.convert_from_keras_model(
                model,
                hls_config=hls_config,
                output_dir=str(out_dir),
                backend="VivadoAccelerator",
                board=args.board,
                io_type="io_stream",
            )
            hls_model.write()
            hls_model.build(csim=False, synth=True, export=True, bitfile=True)
            bit = find_bitstream(out_dir)
            if bit:
                print(f"\nBitstream generated: {bit}")
            else:
                print("\nBuild finished but no .bit found; check the Vivado logs under", out_dir)
        except Exception as exc:  # noqa: BLE001
            logger.error("Bitstream build failed: %s", exc)
            print("\nBitstream build failed. The HLS project is still available at", out_dir)
    else:
        # No toolchain: emit the synthesizable HLS project anyway.
        build_project(model, config, output_dir=out_dir, part=get_target_part())
        print("\n=== No Vivado/Vitis toolchain found ===")
        print(f"Generated synthesizable HLS project at: {out_dir}")
        print("To produce a .bit bitstream, on a machine with Vivado installed run:")
        print("    source /tools/Xilinx/Vivado/<version>/settings64.sh")
        print(f"    BURN_BOARD={args.board} python scripts/03_build_best_bitstream.py")
        print("Then deploy with: python scripts/04_run_fpga_demo.py")

    print(f"\nBest config saved to: {BEST_CONFIG_PATH}")


if __name__ == "__main__":
    main()
