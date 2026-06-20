#!/usr/bin/env python
"""Ingest Qwen-2B: report its architecture, decompose it, and (optionally) export
a tiled, hls4ml-compilable MLP sub-block + golden vectors.

Architecture inspection and decomposition work offline (no weights download). The
``--export`` path builds the tiled SwiGLU MLP Keras model and golden I/O so the
GLM generator + feedback engine can be pointed at a real Qwen sub-block instead of
the toy FFN.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from glm.tasks import make_task  # noqa: E402
from models.qwen.decompose import decompose_layer, feasibility_report, mlp_block_spec  # noqa: E402
from models.qwen.load_qwen import DEFAULT_MODEL_ID, load_qwen_arch  # noqa: E402
from paths import ARTIFACTS_DIR, ensure_dirs, get_logger, get_target_part  # noqa: E402
from ttt.reward import get_board_budget  # noqa: E402

logger = get_logger("burnttt.script.ingest_qwen")

QWEN_MLP_MODEL = ARTIFACTS_DIR / "qwen_mlp_tile.keras"
QWEN_MLP_INPUTS = ARTIFACTS_DIR / "qwen_mlp_inputs.npy"
QWEN_MLP_GOLDEN = ARTIFACTS_DIR / "qwen_mlp_golden.npy"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest + decompose Qwen for FPGA mapping")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--tile-div", type=int, default=64, help="Shrink MLP dims by this factor for compilation")
    parser.add_argument("--export", action="store_true", help="Build + save the tiled MLP Keras block + golden I/O")
    args = parser.parse_args()

    arch = load_qwen_arch(args.model_id)
    part = get_target_part()
    print("\n" + feasibility_report(arch))

    print("\nDecoder-layer sub-blocks:")
    for sb in decompose_layer(arch):
        flag = "hls4ml-ready" if sb.hls4ml_ready else "RESEARCH STUB"
        print(f"  [{flag}] {sb.spec.name}  ({sb.spec.total_macs():,} MACs)  -- {sb.note}")

    task = make_task(mlp_block_spec(arch), part, get_board_budget(part))
    print("\nFPGA task for the MLP block:")
    print(task.describe())

    if args.export:
        from models.qwen.blocks import build_mlp_keras, export_golden

        ensure_dirs()
        model, dims = build_mlp_keras(arch, tile_div=args.tile_div)
        x, y = export_golden(model)
        model.save(QWEN_MLP_MODEL)
        np.save(QWEN_MLP_INPUTS, x)
        np.save(QWEN_MLP_GOLDEN, y)
        print(
            f"\nExported tiled MLP (hidden={dims.hidden}, intermediate={dims.intermediate}):"
            f"\n  model : {QWEN_MLP_MODEL}\n  inputs: {QWEN_MLP_INPUTS}\n  golden: {QWEN_MLP_GOLDEN}"
        )
        print("Next: point the search at this block (see README) or run scripts/08.")
    else:
        print("\n(Use --export to build the tiled MLP Keras block + golden vectors.)")


if __name__ == "__main__":
    main()
