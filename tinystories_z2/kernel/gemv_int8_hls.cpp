// gemv_int8_hls.cpp -- Vitis HLS W8A8 GEMV kernel for the PYNQ Z2 (Zynq-7020).
//
// Authored in HLS C++ and exported as an RTL IP (see hdk/run_hls.tcl), then
// dropped into a Vivado block design alongside the Zynq7 PS (hdk/build_bd.tcl):
//   * the m_axi masters become AXI-HP ports into the PS-side 512 MB DDR3,
//   * s_axilite becomes the control slave on M_AXI_GP0,
//   * the host (PYNQ MMIO) writes buffer physical addresses + sizes and pulses
//     ap_start; the kernel DMAs weights/activation from DDR and writes y back.
//
// Differences from the F2 INT4 kernel (qwen_fpga/kernel/gemv_int4_hls.cpp):
//   * INT8 weights, byte-aligned (no nibble unpack), one scale per row (no group
//     loop) -- a much smaller datapath for the limited Z7020 fabric;
//   * the activation cache binds to BRAM (the Z7020 has NO URAM);
//   * GEMV_MAX_N is sized to TinyStories' largest contraction dim to save BRAM.
//
// Numeric contract is identical to gemv_int8_compute() in gemv_int8.hpp and to
// tinystories_z2/quant.py. Plain-C++-compilable for csim (#pragma ignored by g++).
#include <cstdint>

#ifndef GEMV_MAX_N
#define GEMV_MAX_N 1024          // TinyStories-1M..33M intermediate <= 1024; bump if needed
#endif

// fp16 -> float in fabric (LUT logic; small and pipelineable).
static float half_to_float_hw(uint16_t h) {
#pragma HLS INLINE
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = -1; uint32_t m = mant;
            normloop: do { m <<= 1; e++; } while ((m & 0x400u) == 0);
            m &= 0x3FFu;
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    union { uint32_t u; float f; } cvt; cvt.u = bits; return cvt.f;
}

extern "C" {

// One GEMV: y = (w_scale[:,None] * dequant(W)) @ x, with per-call x_scale.
void gemv_int8(
        const int8_t*   gmem_w,       // [M * N]  INT8 weights, row-major
        const uint16_t* gmem_scale,   // [M]      fp16 per-row scales
        const int8_t*   gmem_x,       // [N]      INT8 activation
        float*          gmem_y,       // [M]      fp32 output
        float           x_scale,
        int             M,
        int             N) {
#pragma HLS INTERFACE m_axi     port=gmem_w     offset=slave bundle=gmem_w     max_read_burst_length=256
#pragma HLS INTERFACE m_axi     port=gmem_scale offset=slave bundle=gmem_scale
#pragma HLS INTERFACE m_axi     port=gmem_x     offset=slave bundle=gmem_x
#pragma HLS INTERFACE m_axi     port=gmem_y     offset=slave bundle=gmem_y
#pragma HLS INTERFACE s_axilite port=gmem_w     bundle=control
#pragma HLS INTERFACE s_axilite port=gmem_scale bundle=control
#pragma HLS INTERFACE s_axilite port=gmem_x     bundle=control
#pragma HLS INTERFACE s_axilite port=gmem_y     bundle=control
#pragma HLS INTERFACE s_axilite port=x_scale    bundle=control
#pragma HLS INTERFACE s_axilite port=M          bundle=control
#pragma HLS INTERFACE s_axilite port=N          bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control

    // Cache the activation on-chip once; reused across all M rows. BRAM (no URAM
    // on Z7020). Partitioned so the inner MAC can read several lanes per cycle.
    int8_t xbuf[GEMV_MAX_N];
#pragma HLS BIND_STORAGE variable=xbuf type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=xbuf cyclic factor=16 dim=1
    load_x: for (int n = 0; n < N; ++n) {
#pragma HLS PIPELINE II=1
        xbuf[n] = gmem_x[n];
    }

    rows: for (int m = 0; m < M; ++m) {
        const int8_t* wrow = gmem_w + (long)m * N;
        int32_t acc = 0;
        inner: for (int n = 0; n < N; ++n) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=16
            acc += (int)wrow[n] * (int)xbuf[n];
        }
        gmem_y[m] = (float)acc * half_to_float_hw(gmem_scale[m]) * x_scale;
    }
}

}  // extern "C"
