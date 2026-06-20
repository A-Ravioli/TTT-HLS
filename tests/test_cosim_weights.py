"""Regression tests for baked-weight cosim (the zero-weight cosim bug).

Before the fix, the SwiGLU kernel used zero-initialized weight arrays, so software
cosim compared the (real, non-zero) golden against an all-zeros kernel output and
could never pass. ``generate_full_bundle(cfg, weights=...)`` now bakes the golden's
projection matrices into a ``weights.h`` the kernel uses.

The end-to-end test needs g++ + Xilinx ap_types headers (via hls4ml); it is skipped
when those are unavailable so CI without the toolchain still runs the suite.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from compiler.cosim import ap_types_include_dirs
from compiler.kernel_lib.swiglu_mlp import SwiGLUConfig, generate_full_bundle, generate_weights_header


def test_weights_header_shape_validation():
    cfg = SwiGLUConfig(hidden_dim=4, intermediate_dim=6)
    good = {
        "gate_w": np.zeros((4, 6)),
        "up_w": np.zeros((4, 6)),
        "down_w": np.zeros((6, 4)),
    }
    header = generate_weights_header(cfg, good)
    assert "static const weight_t gate_w[HIDDEN_DIM][INTER_DIM]" in header
    assert "static const weight_t down_w[INTER_DIM][HIDDEN_DIM]" in header

    bad = {**good, "gate_w": np.zeros((3, 6))}  # wrong hidden dim
    with pytest.raises(ValueError):
        generate_weights_header(cfg, bad)


def test_bundle_bakes_weights_and_uses_accurate_silu():
    cfg = SwiGLUConfig(hidden_dim=4, intermediate_dim=6)
    w = {"gate_w": np.zeros((4, 6)), "up_w": np.zeros((4, 6)), "down_w": np.zeros((6, 4))}

    with_w = generate_full_bundle(cfg, weights=w)
    assert "weights.h" in with_w
    assert '#include "weights.h"' in with_w["kernel_top.cpp"]
    # No zero-initialized local weight storage when weights are baked.
    assert "static weight_t gate_w[HIDDEN_DIM][INTER_DIM];" not in with_w["kernel_top.cpp"]
    # Accurate sigmoid, not the old piecewise-linear 0.197 approximation.
    assert "expf(" in with_w["kernel_top.cpp"]
    assert "0.197" not in with_w["kernel_top.cpp"]

    without_w = generate_full_bundle(cfg)
    assert "weights.h" not in without_w
    assert "static weight_t gate_w[HIDDEN_DIM][INTER_DIM];" in without_w["kernel_top.cpp"]


@pytest.mark.skipif(
    shutil.which("g++") is None or not ap_types_include_dirs(),
    reason="software cosim needs g++ and Xilinx ap_types headers (hls4ml)",
)
def test_software_cosim_passes_with_weights_fails_without():
    pytest.importorskip("tensorflow")
    from compiler.golden import generate_golden_from_keras
    from models.qwen.blocks import build_mlp_keras
    from models.qwen.load_qwen import QwenArch
    from ttt.config_space import KernelBundle
    from ttt.evaluate_hls import evaluate_hls

    arch = QwenArch(
        model_id="test", hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
    )
    model, dims = build_mlp_keras(arch, tile_div=2)  # hidden=8, inter=16
    golden = generate_golden_from_keras(model, n_samples=8)
    golden_mag = float(np.abs(next(iter(golden.outputs.values()))).max())
    assert golden_mag > 0.1  # golden is non-trivial

    weights = {
        "gate_w": model.get_layer("gate_proj").get_weights()[0],
        "up_w": model.get_layer("up_proj").get_weights()[0],
        "down_w": model.get_layer("down_proj").get_weights()[0],
    }
    cfg = SwiGLUConfig(hidden_dim=dims.hidden, intermediate_dim=dims.intermediate)

    baked = KernelBundle(
        sources=generate_full_bundle(cfg, weights=weights),
        hidden_dim=dims.hidden, intermediate_dim=dims.intermediate, part="xcu250-figd2104-2l-e",
    )
    res = evaluate_hls(baked, golden, max_error_threshold=0.05, cleanup=True)
    assert res["cosim_pass"], f"baked-weight cosim should pass, got max_error={res['max_error']}"
    assert res["max_error"] < 0.05

    zero = KernelBundle(
        sources=generate_full_bundle(cfg),  # zero-init weights (old behavior)
        hidden_dim=dims.hidden, intermediate_dim=dims.intermediate, part="xcu250-figd2104-2l-e",
    )
    res0 = evaluate_hls(zero, golden, max_error_threshold=0.05, cleanup=True)
    assert not res0["cosim_pass"]
    # Output was ~all zeros, so the error equals the golden's own magnitude.
    assert res0["max_error"] == pytest.approx(golden_mag, rel=0.2)
