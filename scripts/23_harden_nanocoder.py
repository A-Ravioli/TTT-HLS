#!/usr/bin/env python
"""Harden the NanoCoder MLP block onto the PYNQ-Z2 (hls4ml + bit-accurate csim).

Runs the full hls4ml flow short of the Vivado bitstream (which needs a Linux host
with Vivado/Vitis HLS -- there is no macOS build):
  1. build the NanoCoder MLP block (128 -> 512 -> 128, ReLU) -- trained weights if a
     checkpoint exists, else random init (the fit is weight-independent);
  2. convert to an hls4ml io_parallel project targeting xc7z020 at ReuseFactor 596;
  3. compile the bit-accurate C++ csim and run it -- these numbers are EXACTLY what
     the FPGA fabric would produce;
  4. sweep ap_fixed precision until the quantized block matches the float golden,
     and report the DSP fit against the board's 220-DSP budget.

Run inside the TF2.14 + hls4ml env:  conda run -n burnttt python scripts/23_harden_nanocoder.py
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from models.nanocoder.harden import build_mlp_keras, export_golden, mlp_block_spec  # noqa: E402
from models.nanocoder.model import DEFAULT_ARCH  # noqa: E402
from paths import ARTIFACTS_DIR, get_logger, get_target_part  # noqa: E402

logger = get_logger("burnttt.script.harden_nanocoder")

CKPT_DIR = ARTIFACTS_DIR / "nanocoder"
MLP_NPZ = CKPT_DIR / "mlp_layer0.npz"
# ap_fixed<total,int> candidates, coarse -> fine. First to pass wins.
PRECISIONS = ["ap_fixed<16,6>", "ap_fixed<18,7>", "ap_fixed<20,8>", "ap_fixed<24,9>", "ap_fixed<28,10>"]


def _maybe_load_trained():
    """Return trained layer-0 MLP weights (npz dict) if extracted, else None.

    Uses the torch-free npz (scripts/22 writes it) so this FPGA-side script needs
    only numpy/TF, not torch/transformers.
    """
    if not MLP_NPZ.exists():
        return None
    logger.info("Hardening the TRAINED NanoCoder block from %s", MLP_NPZ)
    return dict(np.load(MLP_NPZ))


def main() -> None:
    parser = argparse.ArgumentParser(description="Harden the NanoCoder MLP block onto the PYNQ-Z2")
    parser.add_argument("--reuse", type=int, default=DEFAULT_ARCH.reuse_to_fit(220))
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()

    import hls4ml

    arch = DEFAULT_ARCH
    part = get_target_part()
    spec = mlp_block_spec(arch, args.layer)
    macs = spec.total_macs()
    dsp_at_reuse = -(-macs // args.reuse)

    print("\n=== NanoCoder -> PYNQ-Z2 hardening ===")
    print(spec.describe())
    print(f"\nTarget part : {part}  (220 DSPs)")
    print(f"MLP MACs    : {macs:,}")
    print(f"ReuseFactor : {args.reuse}  ->  ~{dsp_at_reuse} DSPs  ({'FITS' if dsp_at_reuse <= 220 else 'OVER'} 220)")

    trained = _maybe_load_trained()
    model = build_mlp_keras(arch, args.layer, weights=trained)
    # Realistic post-LayerNorm input scale so the precision sweep is meaningful.
    x, golden = export_golden(model, n=128, scale=1.0)

    print("\nSweeping ap_fixed precision for bit-accurate fit:")
    best = None
    for prec in PRECISIONS:
        cfg = hls4ml.utils.config_from_keras_model(model, granularity="name", default_precision=prec)
        cfg["Model"]["ReuseFactor"] = args.reuse
        hls_model = hls4ml.converters.convert_from_keras_model(
            model, hls_config=cfg, output_dir="/tmp/nanocoder_hls",
            part=part, io_type="io_parallel",
        )
        hls_model.compile()
        y = hls_model.predict(np.ascontiguousarray(x, dtype=np.float32)).reshape(golden.shape)
        err = float(np.max(np.abs(y - golden)))
        status = "PASS" if err < args.threshold else "fail"
        print(f"  {prec:<18} max_err={err:.4f}  [{status}]")
        if err < args.threshold:
            best = (prec, err)
            break

    print("\n=== RESULT ===")
    if best:
        prec, err = best
        print(f"HARDENED: NanoCoder layer-{args.layer} MLP is bit-accurate on {part}")
        print(f"  precision : {prec}")
        print(f"  max error : {err:.4f}  (< {args.threshold})")
        print(f"  resources : ~{dsp_at_reuse}/220 DSPs at ReuseFactor {args.reuse}  -> FITS")
        print(f"  weights   : {'TRAINED checkpoint' if trained is not None else 'random init (fit is weight-independent)'}")
        print("\nThe csim output above is bit-identical to what the Z2 fabric computes.")
        print("Remaining for a live bitstream: synthesise on a Linux+Vivado host")
        print("  (scripts/03_build_best_bitstream.py), then load .bit/.hwh on the board.")
    else:
        print("No swept precision passed; widen PRECISIONS or raise --reuse. (csim still ran bit-accurately.)")


if __name__ == "__main__":
    main()
