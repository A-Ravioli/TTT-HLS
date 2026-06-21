#!/usr/bin/env python
"""Synthesise the NanoCoder MLP block into a PYNQ-Z2 bitstream (Linux + Vivado).

This is the step that crosses the bitstream wall. It MUST run on Linux with Vivado
on PATH (no macOS build exists) -- see docs/BITSTREAM_RECIPE.md. It uses hls4ml's
``VivadoAccelerator`` backend with ``board='pynq-z2'`` to produce a ``.bit`` + a
``.hwh`` overlay and an axi_stream Python driver that scripts/04 / compiler.deploy_pynq
load on the board.

Flow:
  Keras MLP (trained NanoCoder block) -> hls4ml VivadoAccelerator(pynq-z2, axi_stream)
    -> Vivado HLS csynth -> Vivado IPI + bitstream -> design_1_wrapper.bit + .hwh

Run on the Linux+Vivado host (after `source <Vivado>/settings64.sh`):
    conda activate burnttt
    python scripts/25_synth_nanocoder.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from models.nanocoder.harden import build_mlp_keras  # noqa: E402
from models.nanocoder.model import DEFAULT_ARCH  # noqa: E402
from paths import ARTIFACTS_DIR, BUILD_DIR, get_logger  # noqa: E402

logger = get_logger("burnttt.script.synth_nanocoder")

MLP_NPZ = ARTIFACTS_DIR / "nanocoder" / "mlp_layer0.npz"
OUT_DIR = BUILD_DIR / "nanocoder_pynq"


def _have_vivado() -> bool:
    return any(shutil.which(t) for t in ("vivado", "vivado_hls", "vitis_hls"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesise NanoCoder MLP -> PYNQ-Z2 bitstream")
    parser.add_argument("--reuse", type=int, default=DEFAULT_ARCH.reuse_to_fit(220))
    parser.add_argument("--precision", default="ap_fixed<16,6>")
    parser.add_argument("--clock", type=int, default=10, help="Target clock period (ns)")
    parser.add_argument("--no-bitfile", action="store_true", help="csynth + export only, skip the long bitstream P&R")
    args = parser.parse_args()

    import hls4ml

    if not _have_vivado():
        print(
            "\nNo Vivado on PATH. This script only runs on a Linux host with Vivado.\n"
            "  1. Install Vivado ML 2020.1 (free WebPACK covers xc7z020).\n"
            "  2. source /tools/Xilinx/Vivado/2020.1/settings64.sh\n"
            "  3. conda activate burnttt && python scripts/25_synth_nanocoder.py\n"
            "See docs/BITSTREAM_RECIPE.md (incl. the Modal cloud path)."
        )
        sys.exit(2)

    # Trained NanoCoder layer-0 MLP weights (extracted, torch-free).
    weights = dict(np.load(MLP_NPZ)) if MLP_NPZ.exists() else None
    if weights is None:
        logger.warning("No trained weights at %s; synthesising a random-init block.", MLP_NPZ)
    model = build_mlp_keras(DEFAULT_ARCH, 0, weights=weights)

    cfg = hls4ml.utils.config_from_keras_model(model, granularity="name", default_precision=args.precision)
    cfg["Model"]["ReuseFactor"] = args.reuse

    logger.info("Converting with VivadoAccelerator backend (board=pynq-z2, axi_stream)...")
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=cfg,
        backend="VivadoAccelerator",
        board="pynq-z2",
        interface="axi_stream",
        io_type="io_stream",
        clock_period=args.clock,
        output_dir=str(OUT_DIR),
    )
    hls_model.compile()  # bit-accurate csim sanity check before the long build
    logger.info("csim OK. Running Vivado HLS synth + bitstream (this takes ~20-40 min)...")

    hls_model.build(csim=False, synth=True, export=True, bitfile=not args.no_bitfile)

    bits = sorted(OUT_DIR.rglob("*.bit"))
    hwh = sorted(OUT_DIR.rglob("*.hwh"))
    print("\n=== SYNTHESIS COMPLETE ===")
    print(f"  bitstream : {bits[0] if bits else '(not produced — check Vivado log)'}")
    print(f"  overlay   : {hwh[0] if hwh else '(missing .hwh)'}")
    print(f"  driver    : {OUT_DIR}/axi_stream_driver.py")
    print("\nDeploy to the PYNQ-Z2 (booted into the PYNQ Linux image):")
    print(f"  scp {bits[0] if bits else 'design_1_wrapper.bit'} {hwh[0] if hwh else 'design_1.hwh'} xilinx@<board-ip>:~/nanocoder/")
    print("  then on the board:  python scripts/04_run_fpga_demo.py  (uses compiler.deploy_pynq)")


if __name__ == "__main__":
    main()
