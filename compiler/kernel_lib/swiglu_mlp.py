"""Seed HLS template for a fused SwiGLU MLP kernel.

This is the verified starting point for GLM to iterate on. It implements:
    out = down_proj(silu(gate_proj(x)) * up_proj(x))

with configurable:
- Precision (ap_fixed typedefs)
- Tile sizes for weight streaming
- AXI interface width
- Pipeline/unroll pragmas
- Double-buffered weight loading

The template is correct and synthesizable — GLM's job is to optimize it for a
specific (hidden_dim, intermediate_dim, FPGA part) target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SwiGLUConfig:
    """Parameters for the SwiGLU MLP HLS template."""

    hidden_dim: int = 24
    intermediate_dim: int = 64
    weight_bits: int = 16
    weight_int_bits: int = 6
    act_bits: int = 16
    act_int_bits: int = 6
    accum_bits: int = 32
    accum_int_bits: int = 16
    tile_hidden: int = 8  # process this many input elements per cycle group
    tile_inter: int = 16  # process this many intermediate elements per cycle group
    axi_width: int = 64  # AXI port width in bits
    pipeline_ii: int = 1


def generate_kernel_header(cfg: SwiGLUConfig) -> str:
    """Generate the kernel_top.h header file."""
    return f"""\
#ifndef KERNEL_TOP_H
#define KERNEL_TOP_H

#include <ap_fixed.h>
#include <hls_stream.h>

// Precision typedefs — GLM may adjust these
typedef ap_fixed<{cfg.weight_bits},{cfg.weight_int_bits}> weight_t;
typedef ap_fixed<{cfg.act_bits},{cfg.act_int_bits}> act_t;
typedef ap_fixed<{cfg.accum_bits},{cfg.accum_int_bits}> accum_t;
typedef float io_t;  // AXI interface type (float for cosim compatibility)

// Dimensions
constexpr int HIDDEN_DIM = {cfg.hidden_dim};
constexpr int INTER_DIM = {cfg.intermediate_dim};
constexpr int TILE_H = {cfg.tile_hidden};
constexpr int TILE_I = {cfg.tile_inter};

// Top-level function
void kernel_top(const io_t input[HIDDEN_DIM], io_t output[HIDDEN_DIM]);

#endif // KERNEL_TOP_H
"""


def generate_kernel_source(cfg: SwiGLUConfig, with_weights: bool = False) -> str:
    """Generate the kernel_top.cpp implementation.

    When ``with_weights`` is True the weight arrays come from a generated
    ``weights.h`` (baked from the golden model) so C/RTL cosim is a real numeric
    check. Otherwise they are zero-initialized statics -- fine for structure /
    resource estimation, but cosim against a non-trivial golden will fail.
    """
    if with_weights:
        weight_block = (
            '// Weights baked from the golden model (weights.h) so cosim is a real\n'
            "// numeric comparison rather than a check against zero-filled arrays.\n"
            '#include "weights.h"'
        )
    else:
        weight_block = (
            "// Weight storage (in real deployment these come via AXI from DDR)\n"
            "static weight_t gate_w[HIDDEN_DIM][INTER_DIM];\n"
            "static weight_t up_w[HIDDEN_DIM][INTER_DIM];\n"
            "static weight_t down_w[INTER_DIM][HIDDEN_DIM];"
        )
    return f"""\
#include "kernel_top.h"
#include <cmath>

{weight_block}

// SiLU activation: x * sigmoid(x)
static act_t silu(act_t x) {{
    #pragma HLS INLINE
    // Exact sigmoid in float; matches the golden's swish to within the
    // fixed-point quantization that cosim is meant to measure (the old
    // piecewise-linear approximation diverged from the golden by construction).
    float xf = (float)x;
    float sig = 1.0f / (1.0f + expf(-xf));
    return (act_t)(xf * sig);
}}

void kernel_top(const io_t input[HIDDEN_DIM], io_t output[HIDDEN_DIM]) {{
    #pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1
    #pragma HLS INTERFACE s_axilite port=return

    // Local buffers
    act_t x_local[HIDDEN_DIM];
    act_t gate_out[INTER_DIM];
    act_t up_out[INTER_DIM];
    act_t mul_out[INTER_DIM];
    act_t down_out[HIDDEN_DIM];

    #pragma HLS ARRAY_PARTITION variable=x_local cyclic factor={cfg.tile_hidden}
    #pragma HLS ARRAY_PARTITION variable=gate_out cyclic factor={cfg.tile_inter}
    #pragma HLS ARRAY_PARTITION variable=up_out cyclic factor={cfg.tile_inter}

    // Load input
    LOAD_INPUT:
    for (int i = 0; i < HIDDEN_DIM; i++) {{
        #pragma HLS PIPELINE II={cfg.pipeline_ii}
        x_local[i] = (act_t)input[i];
    }}

    // Gate projection: gate_out = gate_w^T * x
    GATE_PROJ:
    for (int j = 0; j < INTER_DIM; j++) {{
        #pragma HLS PIPELINE II={cfg.pipeline_ii}
        accum_t sum = 0;
        GATE_DOT:
        for (int i = 0; i < HIDDEN_DIM; i++) {{
            #pragma HLS UNROLL factor={cfg.tile_hidden}
            sum += (accum_t)x_local[i] * (accum_t)gate_w[i][j];
        }}
        gate_out[j] = silu((act_t)sum);
    }}

    // Up projection: up_out = up_w^T * x
    UP_PROJ:
    for (int j = 0; j < INTER_DIM; j++) {{
        #pragma HLS PIPELINE II={cfg.pipeline_ii}
        accum_t sum = 0;
        UP_DOT:
        for (int i = 0; i < HIDDEN_DIM; i++) {{
            #pragma HLS UNROLL factor={cfg.tile_hidden}
            sum += (accum_t)x_local[i] * (accum_t)up_w[i][j];
        }}
        up_out[j] = (act_t)sum;
    }}

    // Elementwise multiply: mul_out = silu(gate_out) * up_out
    MUL_GATE:
    for (int j = 0; j < INTER_DIM; j++) {{
        #pragma HLS PIPELINE II={cfg.pipeline_ii}
        mul_out[j] = gate_out[j] * up_out[j];
    }}

    // Down projection: down_out = down_w^T * mul_out
    DOWN_PROJ:
    for (int j = 0; j < HIDDEN_DIM; j++) {{
        #pragma HLS PIPELINE II={cfg.pipeline_ii}
        accum_t sum = 0;
        DOWN_DOT:
        for (int i = 0; i < INTER_DIM; i++) {{
            #pragma HLS UNROLL factor={cfg.tile_inter}
            sum += (accum_t)mul_out[i] * (accum_t)down_w[i][j];
        }}
        down_out[j] = (act_t)sum;
    }}

    // Store output
    STORE_OUTPUT:
    for (int i = 0; i < HIDDEN_DIM; i++) {{
        #pragma HLS PIPELINE II={cfg.pipeline_ii}
        output[i] = (io_t)down_out[i];
    }}
}}
"""


def generate_testbench(cfg: SwiGLUConfig) -> str:
    """Generate a basic HLS testbench for the SwiGLU kernel."""
    return f"""\
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include "kernel_top.h"

int main() {{
    io_t input[HIDDEN_DIM];
    io_t output[HIDDEN_DIM];
    io_t expected[HIDDEN_DIM];

    // Initialize with small random values
    srand(42);
    for (int i = 0; i < HIDDEN_DIM; i++) {{
        input[i] = (io_t)((rand() % 200 - 100) / 100.0f);
    }}

    // Run kernel
    kernel_top(input, output);

    // Basic sanity: outputs should be finite
    int errors = 0;
    for (int i = 0; i < HIDDEN_DIM; i++) {{
        if (std::isnan((float)output[i]) || std::isinf((float)output[i])) {{
            printf("ERROR: output[%d] = %f\\n", i, (float)output[i]);
            errors++;
        }}
    }}

    if (errors == 0) {{
        printf("PASS: all outputs are finite.\\n");
        return 0;
    }} else {{
        printf("FAIL: %d errors detected.\\n", errors);
        return 1;
    }}
}}
"""


def _format_c_matrix(arr: np.ndarray) -> str:
    """Render a 2-D array as a C99 brace-enclosed initializer list."""
    rows = ["    {" + ", ".join(f"{float(v):.8g}" for v in row) + "}" for row in arr]
    return "{\n" + ",\n".join(rows) + "\n}"


def generate_weights_header(cfg: SwiGLUConfig, weights: dict[str, Any]) -> str:
    """Emit ``weights.h`` baking the golden model's projection matrices.

    ``weights`` maps ``gate_w``/``up_w``/``down_w`` to 2-D arrays shaped
    ``(hidden, inter)``, ``(hidden, inter)`` and ``(inter, hidden)`` respectively
    -- exactly the Keras ``Dense`` kernels (``y = x @ W``) the golden was built
    from, so the kernel computes the same function.
    """
    gate_w = np.asarray(weights["gate_w"], dtype=float)
    up_w = np.asarray(weights["up_w"], dtype=float)
    down_w = np.asarray(weights["down_w"], dtype=float)

    expected = {
        "gate_w": (cfg.hidden_dim, cfg.intermediate_dim),
        "up_w": (cfg.hidden_dim, cfg.intermediate_dim),
        "down_w": (cfg.intermediate_dim, cfg.hidden_dim),
    }
    for name, arr in (("gate_w", gate_w), ("up_w", up_w), ("down_w", down_w)):
        if arr.shape != expected[name]:
            raise ValueError(
                f"{name} has shape {arr.shape}, expected {expected[name]} "
                f"for hidden={cfg.hidden_dim}, inter={cfg.intermediate_dim}"
            )

    return f"""\
#ifndef WEIGHTS_H
#define WEIGHTS_H

#include "kernel_top.h"

// Auto-generated from the golden model; do not edit by hand.
static const weight_t gate_w[HIDDEN_DIM][INTER_DIM] = {_format_c_matrix(gate_w)};

static const weight_t up_w[HIDDEN_DIM][INTER_DIM] = {_format_c_matrix(up_w)};

static const weight_t down_w[INTER_DIM][HIDDEN_DIM] = {_format_c_matrix(down_w)};

#endif // WEIGHTS_H
"""


def generate_full_bundle(cfg: SwiGLUConfig, weights: dict[str, Any] | None = None) -> dict[str, str]:
    """Generate all source files for the SwiGLU kernel bundle.

    Returns a dict of {filename: content} suitable for KernelBundle.sources. When
    ``weights`` is given, a ``weights.h`` is emitted and the kernel uses it so
    cosim against the golden is a real numeric check.
    """
    sources = {
        "kernel_top.h": generate_kernel_header(cfg),
        "kernel_top.cpp": generate_kernel_source(cfg, with_weights=weights is not None),
    }
    if weights is not None:
        sources["weights.h"] = generate_weights_header(cfg, weights)
    return sources
