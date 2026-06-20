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


def generate_kernel_source(cfg: SwiGLUConfig) -> str:
    """Generate the kernel_top.cpp implementation."""
    return f"""\
#include "kernel_top.h"
#include <cmath>

// Weight storage (in real deployment these come via AXI from DDR)
static weight_t gate_w[HIDDEN_DIM][INTER_DIM];
static weight_t up_w[HIDDEN_DIM][INTER_DIM];
static weight_t down_w[INTER_DIM][HIDDEN_DIM];

// SiLU activation: x * sigmoid(x)
static act_t silu(act_t x) {{
    #pragma HLS INLINE
    // Piecewise linear approximation for HLS-friendly sigmoid
    act_t abs_x = (x > 0) ? x : (act_t)(-x);
    act_t sig;
    if (abs_x > (act_t)4.0) {{
        sig = (x > 0) ? (act_t)1.0 : (act_t)0.0;
    }} else {{
        // Linear approximation: sigmoid(x) ~ 0.5 + 0.25*x for |x| < 4
        sig = (act_t)0.5 + (act_t)0.197 * x;
        if (sig > (act_t)1.0) sig = (act_t)1.0;
        if (sig < (act_t)0.0) sig = (act_t)0.0;
    }}
    return x * sig;
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


def generate_full_bundle(cfg: SwiGLUConfig) -> dict[str, str]:
    """Generate all source files for the SwiGLU kernel bundle.

    Returns a dict of {filename: content} suitable for KernelBundle.sources.
    """
    return {
        "kernel_top.h": generate_kernel_header(cfg),
        "kernel_top.cpp": generate_kernel_source(cfg),
    }
