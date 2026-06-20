"""Generate golden reference I/O for cosimulation from PyTorch/NumPy Qwen blocks.

The golden vectors are the ground-truth outputs that HLS cosim must match within
ε. They come from the float PyTorch model (or the Keras tiled block) and are
stored as ``.npy`` files alongside the HLS project.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from compiler.ir import BlockIR
from paths import get_logger

logger = get_logger("burnttt.compiler.golden")


@dataclass
class GoldenIO:
    """Input/output reference tensors for a block."""

    inputs: dict[str, np.ndarray]
    outputs: dict[str, np.ndarray]
    metadata: dict[str, Any]

    @property
    def n_samples(self) -> int:
        first = next(iter(self.inputs.values()))
        return int(first.shape[0])

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for name, arr in self.inputs.items():
            np.save(directory / f"input_{name}.npy", arr)
        for name, arr in self.outputs.items():
            np.save(directory / f"output_{name}.npy", arr)

    @classmethod
    def load(cls, directory: Path) -> "GoldenIO":
        inputs = {}
        outputs = {}
        for p in sorted(directory.glob("input_*.npy")):
            name = p.stem.removeprefix("input_")
            inputs[name] = np.load(p)
        for p in sorted(directory.glob("output_*.npy")):
            name = p.stem.removeprefix("output_")
            outputs[name] = np.load(p)
        return cls(inputs=inputs, outputs=outputs, metadata={})


def generate_golden_from_keras(model: Any, n_samples: int = 128, seed: int = 42) -> GoldenIO:
    """Run the Keras model on random inputs to get golden outputs."""
    rng = np.random.default_rng(seed)
    in_dim = int(model.input_shape[-1])
    x = rng.standard_normal((n_samples, in_dim)).astype("float32")
    y = model.predict(x, verbose=0).astype("float32")
    return GoldenIO(
        inputs={"x": x},
        outputs={"y": y},
        metadata={"source": "keras", "n_samples": n_samples, "seed": seed},
    )


def generate_golden_from_weights(
    ir: BlockIR,
    weights: dict[str, np.ndarray],
    n_samples: int = 128,
    seed: int = 42,
) -> GoldenIO:
    """Execute the IR with NumPy using provided weight matrices (no framework needed).

    Supports matmul, silu activation, and elementwise_mul — enough for SwiGLU MLP.
    """
    rng = np.random.default_rng(seed)
    in_tensor = ir.input_tensors()[0]
    x = rng.standard_normal((n_samples, *in_tensor.shape)).astype("float32")

    tensors: dict[str, np.ndarray] = {"x": x}
    for op in ir.ops:
        if op.kind == "matmul":
            inp = tensors[op.inputs[0]]
            w = weights[f"{op.name}.weight"]
            b = weights.get(f"{op.name}.bias")
            out = inp @ w
            if b is not None:
                out = out + b
            tensors[op.outputs[0]] = out
        elif op.kind == "activation":
            inp = tensors[op.inputs[0]]
            if op.attrs.get("function") == "silu":
                tensors[op.outputs[0]] = inp * _sigmoid(inp)
            else:
                tensors[op.outputs[0]] = np.maximum(0, inp)
        elif op.kind == "elementwise_mul":
            a = tensors[op.inputs[0]]
            b = tensors[op.inputs[1]]
            tensors[op.outputs[0]] = a * b
        else:
            raise ValueError(f"Unsupported op kind: {op.kind}")

    outputs = {name: tensors[name] for name in ir.output_names}
    return GoldenIO(
        inputs={"x": x},
        outputs=outputs,
        metadata={"source": "numpy_ir", "n_samples": n_samples, "seed": seed},
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
