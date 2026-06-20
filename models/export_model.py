"""Persist the trained model and golden test vectors to ``artifacts/``."""

from __future__ import annotations

import numpy as np
from tensorflow import keras

from models.train_toy_model import TrainedArtifacts
from paths import (
    GOLDEN_OUTPUTS_PATH,
    MODEL_PATH,
    TEST_INPUTS_PATH,
    ensure_dirs,
    get_logger,
)

logger = get_logger("burnttt.export")


def export(artifacts: TrainedArtifacts) -> None:
    """Save model (.keras) and golden vectors (.npy) to canonical paths."""
    ensure_dirs()
    artifacts.model.save(MODEL_PATH)
    np.save(TEST_INPUTS_PATH, artifacts.x_test)
    np.save(GOLDEN_OUTPUTS_PATH, artifacts.y_golden)
    logger.info("Saved model       -> %s", MODEL_PATH)
    logger.info("Saved test inputs -> %s  shape=%s", TEST_INPUTS_PATH, artifacts.x_test.shape)
    logger.info("Saved golden out  -> %s  shape=%s", GOLDEN_OUTPUTS_PATH, artifacts.y_golden.shape)


def load_model() -> keras.Model:
    """Load the exported Keras model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run scripts/00_train_model.py first."
        )
    return keras.models.load_model(MODEL_PATH)


def load_golden() -> tuple[np.ndarray, np.ndarray]:
    """Load exported test inputs and golden outputs."""
    if not TEST_INPUTS_PATH.exists() or not GOLDEN_OUTPUTS_PATH.exists():
        raise FileNotFoundError(
            "Golden vectors not found. Run scripts/00_train_model.py first."
        )
    x = np.load(TEST_INPUTS_PATH)
    y = np.load(GOLDEN_OUTPUTS_PATH)
    return x, y
