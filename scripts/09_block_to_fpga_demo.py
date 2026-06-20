#!/usr/bin/env python
"""Block-to-FPGA demo + full-model orchestration plan (north-star).

For a chosen block (the toy FFN, or a tiled Qwen MLP exported by scripts/05) this:
  1. prints the full-model orchestration plan (research stub) for Qwen, and
  2. runs the software bit-accurate equivalence demo for the single block, with a
     PASS/FAIL against golden, plus the on-board deployment steps.

A live PYNQ board path is reused from scripts/04 when present; otherwise the
software-equivalence demo always gives something to show.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from paths import ARTIFACTS_DIR, BUILD_DIR, get_logger, get_target_part  # noqa: E402
from ttt.config_space import BurnConfig  # noqa: E402

logger = get_logger("burnttt.script.block_demo")

BEST_CONFIG_PATH = ARTIFACTS_DIR / "best_config.json"
QWEN_MLP_MODEL = ARTIFACTS_DIR / "qwen_mlp_tile.keras"
QWEN_MLP_INPUTS = ARTIFACTS_DIR / "qwen_mlp_inputs.npy"
QWEN_MLP_GOLDEN = ARTIFACTS_DIR / "qwen_mlp_golden.npy"


def _load_block(block: str):
    """Return (model, x, golden) for the requested block."""
    from tensorflow import keras

    if block == "qwen_mlp":
        if not QWEN_MLP_MODEL.exists():
            raise FileNotFoundError(
                f"{QWEN_MLP_MODEL} not found. Run: python scripts/05_ingest_qwen.py --export"
            )
        model = keras.models.load_model(QWEN_MLP_MODEL)
        return model, np.load(QWEN_MLP_INPUTS), np.load(QWEN_MLP_GOLDEN)

    from models.export_model import load_golden, load_model

    model = load_model()
    x, golden = load_golden()
    return model, x, golden


def _best_config() -> BurnConfig:
    if BEST_CONFIG_PATH.exists():
        cfg = BurnConfig.from_dict(json.loads(BEST_CONFIG_PATH.read_text()))
        logger.info("Using best config %s", cfg.short_name())
        return cfg
    cfg = BurnConfig(12, 12, 4, 8, 8, "Resource")
    logger.info("No best_config.json; using %s", cfg.short_name())
    return cfg


def _print_orchestration(model_id: str, parts: list[str], reuse: int) -> None:
    from models.qwen.load_qwen import load_qwen_arch
    from models.qwen.orchestrate import describe_plan

    arch = load_qwen_arch(model_id)
    print("\n" + describe_plan(arch, parts, reuse=reuse))


def _software_demo(block: str) -> None:
    from compiler.build_hls4ml_project import build_project, compile_project

    model, x, golden = _load_block(block)
    config = _best_config()
    print(f"\n=== Software bit-accurate equivalence demo: {block} on {get_target_part()} ===")
    hls_model = build_project(model, config, output_dir=BUILD_DIR / f"demo_{block}")
    if not compile_project(hls_model):
        print("hls4ml compile failed; cannot run software demo for this config.")
        return
    y_hls = hls_model.predict(np.ascontiguousarray(x, dtype=np.float32)).reshape(golden.shape)
    max_err = float(np.max(np.abs(y_hls - golden)))
    np.set_printoptions(precision=4, suppress=True)
    print(f"Input[0]        : {x[0][:8]} ...")
    print(f"Float output[0] : {golden[0][:8]} ...")
    print(f"hls4ml output[0]: {y_hls[0][:8]} ...")
    print(f"Max error       : {max_err}")
    print("PASS" if max_err < 0.25 else "FAIL")


def main() -> None:
    parser = argparse.ArgumentParser(description="Block-to-FPGA demo + orchestration plan")
    parser.add_argument("--block", choices=["tiny_ffn", "qwen_mlp"], default="tiny_ffn")
    parser.add_argument("--model-id", default="Qwen/Qwen2-1.5B")
    parser.add_argument("--parts", default="xcu250-figd2104-2l-e", help="Comma-separated FPGA parts for the plan")
    parser.add_argument("--reuse", type=int, default=64)
    parser.add_argument("--no-demo", action="store_true", help="Only print the orchestration plan")
    args = parser.parse_args()

    if args.block == "qwen_mlp" or args.block == "tiny_ffn":
        _print_orchestration(args.model_id, [p.strip() for p in args.parts.split(",") if p.strip()], args.reuse)

    if not args.no_demo:
        try:
            _software_demo(args.block)
        except Exception as exc:  # noqa: BLE001
            print(f"\nSoftware demo unavailable ({exc}).")

    print("\nTo run on real hardware:")
    print("  1. Build a bitstream: python scripts/03_build_best_bitstream.py")
    print("  2. Copy build/best/*.bit and *.hwh to the PYNQ board")
    print("  3. On the board: python scripts/04_run_fpga_demo.py")
    print("\nFull Qwen-2B (all blocks) is the north star; see the orchestration plan above for remaining work.")


if __name__ == "__main__":
    main()
