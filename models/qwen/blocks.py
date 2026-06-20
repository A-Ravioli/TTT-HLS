"""Build a compilable Keras model for a Qwen sub-block + golden I/O export.

The full Qwen MLP (e.g. 1536 -> 8960) is far too large for ``io_parallel`` on any
single FPGA, so for actual compilation experiments we build a *tiled* version
(hidden/intermediate divided down) that exercises the exact same SwiGLU structure
and quantization tradeoffs at a tractable size. The :class:`BlockSpec` still
reports the true full-size MAC count so the task description is honest.

TensorFlow/Keras is imported lazily so this module is importable without it.
"""

from __future__ import annotations

from dataclasses import dataclass

from models.qwen.load_qwen import QwenArch
from paths import get_logger

logger = get_logger("burnttt.qwen.blocks")


@dataclass
class TiledDims:
    hidden: int
    intermediate: int
    tile_div: int


def tile_dims(arch: QwenArch, tile_div: int = 64, min_dim: int = 8) -> TiledDims:
    """Shrink the MLP dims by ``tile_div`` to a tractable, hls4ml-compilable size."""
    h = max(min_dim, arch.hidden_size // tile_div)
    inter = max(min_dim, arch.intermediate_size // tile_div)
    return TiledDims(hidden=h, intermediate=inter, tile_div=tile_div)


def build_mlp_keras(arch: QwenArch, tile_div: int = 64, seed: int = 0):
    """Build a (tiled) SwiGLU MLP block as a named Keras functional model.

        x -> gate_proj -> SiLU --\\
        x -> up_proj   -----------> (*) -> down_proj -> out

    Layer names (``gate_proj``/``up_proj``/``down_proj``) match the BlockSpec so
    per-layer hls4ml configs line up.
    """
    import tensorflow as tf  # noqa: F401
    from tensorflow import keras

    dims = tile_dims(arch, tile_div)
    logger.info(
        "Building tiled Qwen MLP: hidden=%d intermediate=%d (tile_div=%d) from %s",
        dims.hidden, dims.intermediate, tile_div, arch.model_id,
    )
    init = keras.initializers.GlorotUniform(seed=seed)
    inputs = keras.Input(shape=(dims.hidden,), name="mlp_input")
    # Qwen MLP projections are bias-free; keeping them so here also lets the custom
    # HLS kernel (which has no bias term) match this golden during cosim.
    gate = keras.layers.Dense(dims.intermediate, name="gate_proj", use_bias=False, kernel_initializer=init)(inputs)
    gate_act = keras.layers.Activation("swish", name="silu")(gate)  # keras 'swish' == SiLU
    up = keras.layers.Dense(dims.intermediate, name="up_proj", use_bias=False, kernel_initializer=init)(inputs)
    gated = keras.layers.Multiply(name="gate_mul")([gate_act, up])
    out = keras.layers.Dense(dims.hidden, name="down_proj", use_bias=False, kernel_initializer=init)(gated)
    model = keras.Model(inputs=inputs, outputs=out, name=f"QwenMLPTile_d{tile_div}")
    return model, dims


def export_golden(model, n: int = 128, seed: int = 1234):
    """Random inputs + the block's float outputs (the function we burn)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    in_dim = int(model.input_shape[-1])
    x = rng.standard_normal((n, in_dim)).astype("float32")
    y = model.predict(x, verbose=0).astype("float32")
    return x, y


def build_attention_keras(arch: QwenArch, tile_div: int = 64):
    """RESEARCH STUB: attention is not hls4ml-native.

    The q/k/v/o projections are plain Dense layers and DO compile, but softmax,
    scaled dot-product, and RoPE are not expressible in hls4ml's Keras frontend.
    A real attention block needs FINN or hand-written HLS (Phase 4). We raise here
    rather than silently emitting a wrong block.
    """
    raise NotImplementedError(
        "Attention block compilation is a Phase 4 research stub: hls4ml cannot "
        "express softmax/RoPE. Use models.qwen.decompose.attention_block_spec to "
        "study the projection-only resource cost, or route attention through FINN."
    )
