"""An :class:`FpgaTask` is the unit of test-time adaptation.

The whole point of the realignment is that the GLM generator treats each
``(model-block, FPGA-part)`` pair as a *fresh task* and adapts to it. This module
defines what a task is and how to describe it to the LLM, without importing any
heavy ML dependency (so it can be constructed and inspected anywhere).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LayerSpec:
    """A single matmul-bearing layer in a block (the part hls4ml maps to DSPs)."""

    name: str
    kind: str  # "dense", "attention_qkv", "attention_out", "mlp_gate", ...
    n_in: int
    n_out: int

    def n_mult(self) -> int:
        return self.n_in * self.n_out

    def describe(self) -> str:
        return f"{self.name} ({self.kind}): {self.n_in} -> {self.n_out}  [{self.n_mult():,} MACs]"


@dataclass(frozen=True)
class BlockSpec:
    """A model block to be compiled onto fabric (e.g. a transformer FFN block)."""

    name: str
    layers: tuple[LayerSpec, ...]
    notes: str = ""

    def total_macs(self) -> int:
        return sum(l.n_mult() for l in self.layers)

    def describe(self) -> str:
        lines = [f"Block: {self.name}  (total {self.total_macs():,} MACs)"]
        for l in self.layers:
            lines.append(f"  - {l.describe()}")
        if self.notes:
            lines.append(f"  notes: {self.notes}")
        return "\n".join(lines)


@dataclass(frozen=True)
class FpgaTask:
    """A concrete adaptation task: this block, on this part, under this budget."""

    name: str
    block: BlockSpec
    target_part: str
    budget: dict[str, int]
    max_error_threshold: float = 0.25
    extra: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        budget_str = ", ".join(f"{k.upper()}={v}" for k, v in self.budget.items())
        return (
            f"TASK: {self.name}\n"
            f"FPGA part: {self.target_part}\n"
            f"Resource budget (must fit): {budget_str}\n"
            f"Max acceptable output error vs float golden: {self.max_error_threshold}\n"
            f"{self.block.describe()}"
        )


# -- builders ---------------------------------------------------------------

def tiny_ffn_block() -> BlockSpec:
    """The existing toy model as a :class:`BlockSpec` (16 -> 64 -> 8 FFN)."""
    return BlockSpec(
        name="TinyFFNBlock",
        layers=(
            LayerSpec("dense_1", "dense", 16, 64),
            LayerSpec("dense_2", "dense", 64, 8),
        ),
        notes="ReLU between the two dense layers; transformer-FFN-shaped toy model.",
    )


def block_from_keras_model(model: Any, name: str | None = None) -> BlockSpec:
    """Extract a :class:`BlockSpec` from a Keras model (lazy; keras not imported here).

    We duck-type the layer interface so this works without importing TensorFlow at
    module load time.
    """
    layers: list[LayerSpec] = []
    for layer in getattr(model, "layers", []):
        units = getattr(layer, "units", None)
        input_shape = getattr(layer, "input_shape", None)
        if units is not None and input_shape is not None:
            n_in = int(input_shape[-1])
            layers.append(LayerSpec(getattr(layer, "name", f"dense_{len(layers)+1}"), "dense", n_in, int(units)))
    return BlockSpec(name=name or getattr(model, "name", "KerasBlock"), layers=tuple(layers))


def make_task(
    block: BlockSpec,
    target_part: str,
    budget: dict[str, int],
    max_error_threshold: float = 0.25,
    name: str | None = None,
) -> FpgaTask:
    return FpgaTask(
        name=name or f"{block.name}@{target_part}",
        block=block,
        target_part=target_part,
        budget=dict(budget),
        max_error_threshold=max_error_threshold,
    )
