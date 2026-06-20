"""Tests for compiler.ir — block IR from Qwen decomposition."""

import pytest

from compiler.ir import BlockIR, OpNode, TensorDesc, block_spec_to_ir
from glm.tasks import BlockSpec, LayerSpec


def test_swiglu_ir_structure():
    spec = BlockSpec(
        name="test_mlp",
        layers=(
            LayerSpec("gate_proj", "mlp_gate", 24, 64),
            LayerSpec("up_proj", "mlp_up", 24, 64),
            LayerSpec("down_proj", "mlp_down", 64, 24),
        ),
        notes="SwiGLU: down(silu(gate(x)) * up(x))",
    )
    ir = block_spec_to_ir(spec)
    assert ir.name == "test_mlp"
    assert len(ir.input_names) == 1
    assert ir.input_names[0] == "x"
    assert len(ir.output_names) == 1
    assert ir.output_names[0] == "y"
    # Should have gate_proj, silu, up_proj, gate_mul, down_proj
    assert len(ir.ops) == 5
    op_names = [op.name for op in ir.ops]
    assert "gate_proj" in op_names
    assert "silu" in op_names
    assert "up_proj" in op_names
    assert "gate_mul" in op_names
    assert "down_proj" in op_names


def test_swiglu_ir_total_macs():
    spec = BlockSpec(
        name="mlp",
        layers=(
            LayerSpec("gate_proj", "mlp_gate", 24, 64),
            LayerSpec("up_proj", "mlp_up", 24, 64),
            LayerSpec("down_proj", "mlp_down", 64, 24),
        ),
    )
    ir = block_spec_to_ir(spec)
    # 24*64 + 24*64 + 64*24 = 4608
    assert ir.total_macs() == 24 * 64 + 24 * 64 + 64 * 24


def test_generic_chain_ir():
    spec = BlockSpec(
        name="simple",
        layers=(
            LayerSpec("dense_1", "dense", 16, 64),
            LayerSpec("dense_2", "dense", 64, 8),
        ),
    )
    ir = block_spec_to_ir(spec)
    assert len(ir.ops) == 2
    assert ir.input_names == ["x"]
    assert ir.output_names == ["y"]


def test_tensor_desc_numel():
    t = TensorDesc("x", (3, 4, 5))
    assert t.numel == 60
