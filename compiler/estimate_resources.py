"""Analytical latency / resource estimator for hls4ml Dense networks.

When no Vivado/Vitis toolchain is installed (the common case), we cannot run
real synthesis. Rather than leaving every config tied on latency/resources, we
estimate them from first principles using how hls4ml maps Dense layers to
``io_parallel`` hardware:

* A Dense(n_in, n_out) layer has ``n_in * n_out`` multiplies.
* With ReuseFactor ``R``, ~``ceil(n_mult / R)`` multipliers run in parallel,
  each costing roughly one DSP; the layer's initiation interval is ~``R``.
* Lower R  -> more parallelism -> lower latency, more DSP/LUT.
* Higher R -> serialized       -> higher latency, fewer DSP, more BRAM.

These estimates are clearly marked ``estimated=True`` and are overridden by real
parsed reports whenever synthesis actually runs. They exist so the search has a
meaningful, monotonic objective even off-toolchain.
"""

from __future__ import annotations

import math
from typing import Any

from tensorflow import keras

from ttt.config_space import BurnConfig


def _dense_layers(model: keras.Model) -> list[tuple[int, int]]:
    """Return (n_in, n_out) for each Dense layer in declaration order."""
    dims: list[tuple[int, int]] = []
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense):
            n_in = int(layer.input_shape[-1])
            n_out = int(layer.units)
            dims.append((n_in, n_out))
    return dims


def estimate_hardware(model: keras.Model, config: BurnConfig) -> dict[str, Any]:
    """Estimate latency_cycles, ii, dsp, lut, ff, bram for ``config``."""
    dims = _dense_layers(model)
    reuse = [config.reuse_dense_1, config.reuse_dense_2]
    # Pad reuse list if the model has more/less than 2 dense layers.
    while len(reuse) < len(dims):
        reuse.append(config.reuse_dense_2)

    bits = max(config.weight_bits, config.activation_bits)
    dsp_per_mult = 1 if bits <= 18 else 2
    strat_factor = 1.0 if config.strategy == "Latency" else 1.6  # Resource serializes

    total_dsp = 0
    total_lut = 0.0
    total_ff = 0.0
    total_latency = 10.0  # fixed I/O + glue overhead
    total_weight_bits = 0
    max_ii = 1

    for (n_in, n_out), r in zip(dims, reuse):
        n_mult = n_in * n_out
        r = max(1, r)
        parallel_mults = math.ceil(n_mult / r)

        # DSP: Resource strategy time-multiplexes harder, so fewer DSPs.
        layer_dsp = parallel_mults * dsp_per_mult
        if config.strategy == "Resource":
            layer_dsp = math.ceil(layer_dsp * 0.6)
        total_dsp += layer_dsp

        # Latency: ~reuse cycles to stream multiplies + adder-tree depth.
        accum_depth = math.ceil(math.log2(max(2, n_in))) + 3
        layer_latency = strat_factor * (r + accum_depth)
        total_latency += layer_latency
        max_ii = max(max_ii, int(round(r * strat_factor)))

        # LUT / FF scale with parallelism and bitwidth.
        total_lut += 45.0 * parallel_mults * (bits / 16.0) + 200.0
        total_ff += 70.0 * parallel_mults * (bits / 16.0) + 200.0

        total_weight_bits += (n_mult + n_out) * config.weight_bits

    # BRAM: Resource strategy stores weights in block RAM (18 Kbit each).
    if config.strategy == "Resource":
        bram = math.ceil(total_weight_bits / (18 * 1024))
    else:
        bram = max(0, math.ceil(total_weight_bits / (18 * 1024 * 4)))

    return {
        "latency_cycles": int(round(total_latency)),
        "ii": int(max_ii),
        "dsp": int(total_dsp),
        "lut": int(round(total_lut)),
        "ff": int(round(total_ff)),
        "bram": int(bram),
        "estimated": True,
    }
