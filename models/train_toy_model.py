"""Train TinyFFNBlock on synthetic teacher-generated data.

No real dataset is needed: the proof of this project is hardware equivalence,
not benchmark accuracy. We sample random inputs, label them with a fixed random
teacher network, and fit the student TinyFFNBlock with MSE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow import keras

from models.tiny_ffn import INPUT_DIM, OUTPUT_DIM, make_model, make_teacher_model
from paths import get_logger

logger = get_logger("burnttt.train")


@dataclass
class TrainedArtifacts:
    model: keras.Model
    x_test: np.ndarray
    y_golden: np.ndarray


def _make_dataset(
    teacher: keras.Model,
    n_samples: int,
    input_dim: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    x = rng.standard_normal((n_samples, input_dim)).astype(np.float32)
    y = teacher.predict(x, verbose=0).astype(np.float32)
    return x, y


def train(
    epochs: int = 30,
    n_train: int = 4096,
    n_test: int = 256,
    seed: int = 1234,
) -> TrainedArtifacts:
    """Train the student model and return it with golden test vectors."""
    tf.random.set_seed(seed)
    rng = np.random.default_rng(seed)

    teacher = make_teacher_model(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
    x_train, y_train = _make_dataset(teacher, n_train, INPUT_DIM, rng)
    x_test, _ = _make_dataset(teacher, n_test, INPUT_DIM, rng)

    model = make_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    logger.info("Training TinyFFNBlock: %d params", model.count_params())
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=64,
        validation_split=0.1,
        verbose=0,
    )

    train_loss = model.evaluate(x_train, y_train, verbose=0)
    logger.info("Final train loss/mae: %s", train_loss)

    # Golden outputs come from the trained student (this is the function we burn).
    y_golden = model.predict(x_test, verbose=0).astype(np.float32)
    return TrainedArtifacts(model=model, x_test=x_test, y_golden=y_golden)
