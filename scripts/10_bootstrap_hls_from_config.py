#!/usr/bin/env python3
"""Warm-start custom HLS from the best Phase-3 BlockConfig / hls4ml export.

Takes the best config found by ``08_eval_glm_generator.py`` and translates it
into an initial KernelBundle (SwiGLU HLS template with matching precision/tiling).
This is the bridge from config-author mode to compiler-author mode.

Usage:
    python scripts/10_bootstrap_hls_from_config.py [--tile-div 64] [--part xcu250-figd2104-2l-e]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compiler.kernel_lib.swiglu_mlp import SwiGLUConfig, generate_full_bundle
from models.qwen.blocks import build_mlp_keras, tile_dims
from models.qwen.load_qwen import get_default_arch
from paths import BUILD_DIR, ensure_dirs, get_logger
from ttt.config_space import BlockConfig, BurnConfig, KernelBundle, LayerKnobs

logger = get_logger("burnttt.scripts.bootstrap_hls")


def main():
    parser = argparse.ArgumentParser(description="Bootstrap HLS from best config")
    parser.add_argument("--tile-div", type=int, default=64, help="Tile divisor for Qwen dims")
    parser.add_argument("--part", default="xcu250-figd2104-2l-e", help="Target FPGA part")
    parser.add_argument("--weight-bits", type=int, default=16, help="Weight precision")
    parser.add_argument("--act-bits", type=int, default=16, help="Activation precision")
    parser.add_argument("--int-bits", type=int, default=6, help="Integer bits")
    args = parser.parse_args()

    ensure_dirs()
    arch = get_default_arch()
    dims = tile_dims(arch, args.tile_div)

    logger.info(
        "Bootstrapping HLS from config: hidden=%d intermediate=%d part=%s",
        dims.hidden, dims.intermediate, args.part,
    )

    # Build a KernelBundle from the specified precision
    cfg = SwiGLUConfig(
        hidden_dim=dims.hidden,
        intermediate_dim=dims.intermediate,
        weight_bits=args.weight_bits,
        weight_int_bits=args.int_bits,
        act_bits=args.act_bits,
        act_int_bits=args.int_bits,
    )
    sources = generate_full_bundle(cfg)

    bundle = KernelBundle(
        sources=sources,
        hidden_dim=dims.hidden,
        intermediate_dim=dims.intermediate,
        part=args.part,
        weight_bits=args.weight_bits,
        weight_int_bits=args.int_bits,
        act_bits=args.act_bits,
        act_int_bits=args.int_bits,
    )

    # Write the HLS project
    output_dir = BUILD_DIR / "bootstrap_hls" / bundle.short_name()
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in sources.items():
        (output_dir / filename).write_text(content)

    logger.info("Bootstrap HLS written to %s", output_dir)
    logger.info("KernelBundle: %s", bundle.short_name())
    logger.info("Files: %s", list(sources.keys()))

    print(f"\nBootstrap complete: {output_dir}")
    print(f"  Hidden dim:      {dims.hidden}")
    print(f"  Intermediate:    {dims.intermediate}")
    print(f"  Precision:       w{args.weight_bits}a{args.act_bits}i{args.int_bits}")
    print(f"  Part:            {args.part}")
    print(f"\nNext: run scripts/11_glm_author_hls.py to start the GLM HLS loop.")


if __name__ == "__main__":
    main()
