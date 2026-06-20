"""Translate a :class:`BurnConfig` into an hls4ml config dictionary.

We start from ``config_from_keras_model(granularity='name')`` and then patch
the per-layer precision and reuse factors so the autotuner controls exactly the
knobs in the search space.
"""

from __future__ import annotations

from typing import Any

import hls4ml
from tensorflow import keras

from models.tiny_ffn import DENSE_1_NAME, DENSE_2_NAME
from paths import get_logger
from ttt.config_space import BlockConfig, BurnConfig

logger = get_logger("burnttt.hls_config")


def _set_layer_precision(layer_cfg: dict[str, Any], weight_prec: str, act_prec: str) -> None:
    prec = layer_cfg.setdefault("Precision", {})
    if isinstance(prec, dict):
        prec["weight"] = weight_prec
        prec["bias"] = weight_prec
        prec["result"] = act_prec
    else:
        # Some granularities give a single precision string per layer.
        layer_cfg["Precision"] = act_prec


def make_hls_config(model: keras.Model, config: BurnConfig) -> dict[str, Any]:
    """Build a complete hls4ml config dict for ``model`` under ``config``."""
    weight_prec = config.weight_precision()
    act_prec = config.activation_precision()

    hls_config = hls4ml.utils.config_from_keras_model(
        model,
        granularity="name",
        default_precision=act_prec,
        default_reuse_factor=config.reuse_dense_1,
    )

    # Model-level defaults.
    model_cfg = hls_config.setdefault("Model", {})
    model_cfg["Precision"] = act_prec
    model_cfg["ReuseFactor"] = config.reuse_dense_1
    model_cfg["Strategy"] = config.strategy

    # Map our two tunable dense layers to their reuse factors.
    reuse_by_layer = {
        DENSE_1_NAME: config.reuse_dense_1,
        DENSE_2_NAME: config.reuse_dense_2,
    }

    layer_names = hls_config.get("LayerName", {})
    for name, layer_cfg in layer_names.items():
        _set_layer_precision(layer_cfg, weight_prec, act_prec)
        layer_cfg["Strategy"] = config.strategy
        if name in reuse_by_layer:
            layer_cfg["ReuseFactor"] = reuse_by_layer[name]

    logger.debug("hls_config for %s: %s", config.short_name(), hls_config)
    return hls_config


def make_hls_config_from_block(model: keras.Model, block: BlockConfig) -> dict[str, Any]:
    """Build an hls4ml config from a per-layer :class:`BlockConfig`.

    Each named layer in ``block.layers`` gets its own precision and reuse factor;
    layers not named in the block fall back to the first layer's knobs. This is the
    richer artifact the GLM authors for multi-layer blocks (e.g. Qwen's SwiGLU MLP).
    """
    if not block.layers:
        raise ValueError("BlockConfig has no layers")
    first = next(iter(block.layers.values()))

    hls_config = hls4ml.utils.config_from_keras_model(
        model,
        granularity="name",
        default_precision=first.activation_precision(),
        default_reuse_factor=first.reuse,
    )
    model_cfg = hls_config.setdefault("Model", {})
    model_cfg["Precision"] = first.activation_precision()
    model_cfg["ReuseFactor"] = first.reuse
    model_cfg["Strategy"] = block.strategy

    layer_names = hls_config.get("LayerName", {})
    for name, layer_cfg in layer_names.items():
        knobs = block.layers.get(name, first)
        _set_layer_precision(layer_cfg, knobs.weight_precision(), knobs.activation_precision())
        layer_cfg["Strategy"] = block.strategy
        layer_cfg["ReuseFactor"] = knobs.reuse

    logger.debug("block hls_config (%d layers): %s", len(block.layers), hls_config)
    return hls_config
