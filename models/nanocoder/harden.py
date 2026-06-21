"""NanoCoder MLP sub-block as a compilable Keras model + BlockSpec.

This is the piece that actually hardens onto the PYNQ-Z2. It mirrors
``models.tinystories.blocks`` but (a) uses ReLU (hls4ml-native) and (b) can lift
the *trained* weights out of a NanoCoder ``GPTNeoForCausalLM`` so we harden the
real block, not a random one.

TensorFlow/Keras and torch are imported lazily.
"""

from __future__ import annotations

from glm.tasks import BlockSpec, LayerSpec
from models.nanocoder.model import DEFAULT_ARCH, NanoCoderArch
from paths import get_logger

logger = get_logger("burnttt.nanocoder.harden")


def mlp_block_spec(arch: NanoCoderArch = DEFAULT_ARCH, layer_idx: int = 0) -> BlockSpec:
    """The standard GPT-Neo MLP sub-block (c_fc -> ReLU -> c_proj)."""
    h, inter = arch.hidden_size, arch.intermediate_size
    return BlockSpec(
        name=f"NanoCoder.layer{layer_idx}.mlp",
        layers=(
            LayerSpec("c_fc", "mlp_fc", h, inter),
            LayerSpec("c_proj", "mlp_proj", inter, h),
        ),
        notes="Standard MLP: c_proj(relu(c_fc(x))). ReLU chosen for hls4ml hardening.",
    )


def build_mlp_keras(arch: NanoCoderArch = DEFAULT_ARCH, layer_idx: int = 0,
                    torch_model=None, weights=None, seed: int = 0):
    """Build the NanoCoder MLP block (h -> inter -> h, ReLU) as a Keras model.

    Weight source, in priority order:
      * ``weights`` -- a dict/NpzFile with keys ``c_fc_kernel``/``c_fc_bias``/
        ``c_proj_kernel``/``c_proj_bias`` already in Keras layout (in, out). This is
        the torch-free path (extracted via scripts/22) so the FPGA env needs only
        numpy, not torch.
      * ``torch_model`` -- a trained ``GPTNeoForCausalLM`` to lift weights from.
      * neither -- deterministic random init (the hardware fit is weight-independent).
    """
    from tensorflow import keras

    h, inter = arch.hidden_size, arch.intermediate_size
    init = keras.initializers.GlorotUniform(seed=seed)
    inputs = keras.Input(shape=(h,), name="mlp_input")
    fc = keras.layers.Dense(inter, name="c_fc", kernel_initializer=init)(inputs)
    act = keras.layers.Activation("relu", name="relu")(fc)
    out = keras.layers.Dense(h, name="c_proj", kernel_initializer=init)(act)
    model = keras.Model(inputs=inputs, outputs=out, name=f"NanoCoderMLP_L{layer_idx}")

    if weights is not None:
        model.get_layer("c_fc").set_weights([weights["c_fc_kernel"], weights["c_fc_bias"]])
        model.get_layer("c_proj").set_weights([weights["c_proj_kernel"], weights["c_proj_bias"]])
        logger.info("Loaded trained NanoCoder layer-%d MLP weights (npz) into the Keras block.", layer_idx)
    elif torch_model is not None:
        mlp = torch_model.transformer.h[layer_idx].mlp
        # torch Linear weight is (out, in); keras Dense kernel is (in, out).
        model.get_layer("c_fc").set_weights(
            [mlp.c_fc.weight.detach().cpu().numpy().T, mlp.c_fc.bias.detach().cpu().numpy()]
        )
        model.get_layer("c_proj").set_weights(
            [mlp.c_proj.weight.detach().cpu().numpy().T, mlp.c_proj.bias.detach().cpu().numpy()]
        )
        logger.info("Loaded trained NanoCoder layer-%d MLP weights (torch) into the Keras block.", layer_idx)

    return model


def export_golden(model, n: int = 128, seed: int = 1234, scale: float = 1.0):
    """Random inputs (scaled to a realistic post-LayerNorm range) + float outputs."""
    import numpy as np

    rng = np.random.default_rng(seed)
    in_dim = int(model.input_shape[-1])
    x = (scale * rng.standard_normal((n, in_dim))).astype("float32")
    y = model.predict(x, verbose=0).astype("float32")
    return x, y
