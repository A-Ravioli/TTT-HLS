"""TinyFFNBlock: a tiny transformer-style feed-forward block.

This is intentionally minimal so that hls4ml conversion and (optional) HLS
synthesis are fast. It mirrors the feed-forward sub-block of a transformer:

    x -> Dense(hidden) -> ReLU -> Dense(out)

Default dims (see plan.md section 1):
    input_dim = 16, hidden_dim = 64, output_dim = 8
"""

from __future__ import annotations

from tensorflow import keras

INPUT_DIM = 16
HIDDEN_DIM = 64
OUTPUT_DIM = 8

# Stable, descriptive layer names so per-layer hls4ml configs are easy to target.
DENSE_1_NAME = "dense_1"
DENSE_2_NAME = "dense_2"
RELU_NAME = "relu_1"


def make_model(
    input_dim: int = INPUT_DIM,
    hidden_dim: int = HIDDEN_DIM,
    output_dim: int = OUTPUT_DIM,
) -> keras.Model:
    """Build the TinyFFNBlock as a named Keras functional model.

    A functional model with explicit layer names keeps the hls4ml graph stable,
    which matters because the compiler indexes per-layer precision/reuse configs
    by name (``dense_1``, ``dense_2``).
    """
    inputs = keras.Input(shape=(input_dim,), name="ffn_input")
    x = keras.layers.Dense(hidden_dim, name=DENSE_1_NAME)(inputs)
    x = keras.layers.Activation("relu", name=RELU_NAME)(x)
    outputs = keras.layers.Dense(output_dim, name=DENSE_2_NAME)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="TinyFFNBlock")
    return model


def make_teacher_model(
    input_dim: int = INPUT_DIM,
    output_dim: int = OUTPUT_DIM,
) -> keras.Model:
    """A fixed random 'teacher' that defines the synthetic regression target.

    Using a teacher network (rather than pure noise) gives the student a smooth,
    learnable function, so the trained model has meaningful structure to quantize.
    """
    teacher = keras.Sequential(
        [
            keras.Input(shape=(input_dim,)),
            keras.layers.Dense(48, activation="tanh"),
            keras.layers.Dense(output_dim, activation="linear"),
        ],
        name="teacher",
    )
    return teacher
