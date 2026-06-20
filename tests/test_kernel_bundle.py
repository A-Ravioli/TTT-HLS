"""Tests for KernelBundle dataclass and seed template generation."""

import pytest

from compiler.kernel_lib.swiglu_mlp import SwiGLUConfig, generate_full_bundle
from ttt.config_space import BlockConfig, KernelBundle, LayerKnobs


def test_kernel_bundle_roundtrip():
    sources = {"kernel_top.h": "header", "kernel_top.cpp": "source"}
    bundle = KernelBundle(
        sources=sources,
        hidden_dim=24,
        intermediate_dim=64,
        weight_bits=12,
        act_bits=10,
    )
    d = bundle.to_dict()
    restored = KernelBundle.from_dict(d)
    assert restored.sources == sources
    assert restored.hidden_dim == 24
    assert restored.intermediate_dim == 64
    assert restored.weight_bits == 12
    assert restored.act_bits == 10


def test_kernel_bundle_short_name():
    bundle = KernelBundle(
        sources={},
        hidden_dim=24,
        intermediate_dim=64,
        weight_bits=16,
        act_bits=16,
        tile_hidden=8,
        tile_inter=16,
    )
    name = bundle.short_name()
    assert "hls_w16a16" in name
    assert "h24i64" in name
    assert "t8-16" in name


def test_kernel_bundle_from_block_config():
    block = BlockConfig(
        layers={
            "gate_proj": LayerKnobs(12, 12, 4, 4),
            "up_proj": LayerKnobs(12, 12, 4, 4),
            "down_proj": LayerKnobs(12, 12, 4, 4),
        },
        strategy="Resource",
    )
    bundle = KernelBundle.from_block_config(block, hidden_dim=24, intermediate_dim=64)
    assert bundle.hidden_dim == 24
    assert bundle.intermediate_dim == 64
    assert bundle.weight_bits == 12
    assert "kernel_top.h" in bundle.sources
    assert "kernel_top.cpp" in bundle.sources


def test_swiglu_config_generates_valid_sources():
    cfg = SwiGLUConfig(hidden_dim=16, intermediate_dim=32)
    sources = generate_full_bundle(cfg)
    assert "kernel_top.h" in sources
    assert "kernel_top.cpp" in sources
    # Check header has required content
    assert "HIDDEN_DIM" in sources["kernel_top.h"]
    assert "kernel_top" in sources["kernel_top.h"]
    # Check source has pragmas
    assert "#pragma HLS" in sources["kernel_top.cpp"]
    assert "kernel_top" in sources["kernel_top.cpp"]


def test_swiglu_config_respects_dimensions():
    cfg = SwiGLUConfig(hidden_dim=32, intermediate_dim=128)
    sources = generate_full_bundle(cfg)
    assert "HIDDEN_DIM = 32" in sources["kernel_top.h"]
    assert "INTER_DIM = 128" in sources["kernel_top.h"]


def test_swiglu_config_respects_precision():
    cfg = SwiGLUConfig(weight_bits=10, weight_int_bits=4, act_bits=12, act_int_bits=5)
    sources = generate_full_bundle(cfg)
    assert "ap_fixed<10,4>" in sources["kernel_top.h"]
    assert "ap_fixed<12,5>" in sources["kernel_top.h"]
