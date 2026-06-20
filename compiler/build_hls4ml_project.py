"""Generate (and optionally C-compile) an hls4ml HLS project from a config."""

from __future__ import annotations

from pathlib import Path

import hls4ml
from tensorflow import keras

from compiler.make_hls_config import make_hls_config
from paths import get_logger, get_target_part
from ttt.config_space import BurnConfig

logger = get_logger("burnttt.build")


def build_project(
    model: keras.Model,
    config: BurnConfig,
    output_dir: str | Path,
    part: str | None = None,
    backend: str = "Vivado",
    io_type: str = "io_parallel",
):
    """Convert a Keras model into an hls4ml project on disk.

    Returns the (uncompiled) ``hls_model`` object. Call :func:`compile_project`
    to build the C++ bridge used for bit-accurate prediction.
    """
    part = part or get_target_part()
    output_dir = str(output_dir)
    hls_config = make_hls_config(model, config)

    logger.info(
        "Converting %s -> %s (part=%s, backend=%s, strategy=%s)",
        config.short_name(),
        output_dir,
        part,
        backend,
        config.strategy,
    )

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=hls_config,
        output_dir=output_dir,
        part=part,
        backend=backend,
        io_type=io_type,
    )
    hls_model.write()
    return hls_model


def compile_project(hls_model) -> bool:
    """C-compile the generated project so ``hls_model.predict`` is bit-accurate.

    Returns ``True`` on success. Requires a working C++ compiler (g++); failures
    are logged and reported as ``False`` rather than raised.
    """
    try:
        hls_model.compile()
        return True
    except Exception as exc:  # noqa: BLE001 - tool failures must not crash search
        logger.warning("hls_model.compile() failed: %s", exc)
        return False
