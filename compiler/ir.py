"""Block IR from Qwen decomposition: shapes, dtypes, fusion boundaries.

Bridges ``models/qwen/decompose.py`` block specs to the custom HLS toolchain.
The IR captures enough information to emit golden testbench vectors and to
generate the HLS project scaffold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from glm.tasks import BlockSpec, LayerSpec


@dataclass(frozen=True)
class TensorDesc:
    """A named tensor flowing between layers."""

    name: str
    shape: tuple[int, ...]
    dtype: str = "float32"

    @property
    def numel(self) -> int:
        r = 1
        for s in self.shape:
            r *= s
        return r


@dataclass(frozen=True)
class OpNode:
    """One compute operation in the block IR graph."""

    name: str
    kind: str  # "matmul", "elementwise_mul", "activation", "add"
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class BlockIR:
    """Flat dataflow IR for a single compilable block.

    Enough to drive golden-vector generation, HLS scaffold emission, and
    cosim testbench construction.
    """

    name: str
    tensors: dict[str, TensorDesc] = field(default_factory=dict)
    ops: list[OpNode] = field(default_factory=list)
    input_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)

    def input_tensors(self) -> list[TensorDesc]:
        return [self.tensors[n] for n in self.input_names]

    def output_tensors(self) -> list[TensorDesc]:
        return [self.tensors[n] for n in self.output_names]

    def total_macs(self) -> int:
        total = 0
        for op in self.ops:
            if op.kind == "matmul":
                total += op.attrs.get("n_in", 0) * op.attrs.get("n_out", 0)
        return total


def block_spec_to_ir(spec: BlockSpec) -> BlockIR:
    """Convert a :class:`BlockSpec` (from Qwen decomposition) to :class:`BlockIR`."""
    ir = BlockIR(name=spec.name)

    # For an MLP block: x -> gate_proj -> silu -> * up_proj -> down_proj -> out
    # For generic blocks: linear chain of matmuls
    if _is_swiglu_mlp(spec):
        return _swiglu_ir(spec)
    return _generic_linear_chain(spec)


def _is_swiglu_mlp(spec: BlockSpec) -> bool:
    names = {l.name for l in spec.layers}
    return {"gate_proj", "up_proj", "down_proj"}.issubset(names)


def _swiglu_ir(spec: BlockSpec) -> BlockIR:
    layers = {l.name: l for l in spec.layers}
    gate = layers["gate_proj"]
    up = layers["up_proj"]
    down = layers["down_proj"]

    ir = BlockIR(name=spec.name)
    # Tensors
    ir.tensors["x"] = TensorDesc("x", (gate.n_in,))
    ir.tensors["gate_out"] = TensorDesc("gate_out", (gate.n_out,))
    ir.tensors["silu_out"] = TensorDesc("silu_out", (gate.n_out,))
    ir.tensors["up_out"] = TensorDesc("up_out", (up.n_out,))
    ir.tensors["mul_out"] = TensorDesc("mul_out", (gate.n_out,))
    ir.tensors["y"] = TensorDesc("y", (down.n_out,))

    ir.input_names = ["x"]
    ir.output_names = ["y"]

    ir.ops = [
        OpNode("gate_proj", "matmul", ("x",), ("gate_out",),
               {"n_in": gate.n_in, "n_out": gate.n_out}),
        OpNode("silu", "activation", ("gate_out",), ("silu_out",),
               {"function": "silu"}),
        OpNode("up_proj", "matmul", ("x",), ("up_out",),
               {"n_in": up.n_in, "n_out": up.n_out}),
        OpNode("gate_mul", "elementwise_mul", ("silu_out", "up_out"), ("mul_out",), {}),
        OpNode("down_proj", "matmul", ("mul_out",), ("y",),
               {"n_in": down.n_in, "n_out": down.n_out}),
    ]
    return ir


def _generic_linear_chain(spec: BlockSpec) -> BlockIR:
    ir = BlockIR(name=spec.name)
    prev_name = "x"
    ir.tensors[prev_name] = TensorDesc(prev_name, (spec.layers[0].n_in,))
    ir.input_names = [prev_name]

    for i, layer in enumerate(spec.layers):
        out_name = f"{layer.name}_out" if i < len(spec.layers) - 1 else "y"
        ir.tensors[out_name] = TensorDesc(out_name, (layer.n_out,))
        ir.ops.append(
            OpNode(layer.name, "matmul", (prev_name,), (out_name,),
                   {"n_in": layer.n_in, "n_out": layer.n_out})
        )
        prev_name = out_name

    ir.output_names = [prev_name]
    return ir
