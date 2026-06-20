// gemv_int4_hls.cpp -- Vitis HLS kernel for the INT4 HBM-streaming GEMV.
//
// This is the FPGA datapath for the AWS F2 accelerator. It is authored in HLS C++
// and exported as an RTL IP core (see hdk/run_hls.tcl); that IP is then
// instantiated inside the AWS HDK custom logic (derived from the CL_MEM_PERF HBM
// example) and built into a DCP -> AFI via aws_build_dcp_from_cl.py. We use HLS
// only to generate the RTL core -- the deployable AFI is produced by the HDK /
// Vivado flow, which is the supported path on F2 (Vitis *AFI generation* is not).
//
// Interfaces:
//   m_axi gmem_w     : packed INT4 weights in HBM   [M, N/2] uint8
//   m_axi gmem_scale : fp16 group scales in HBM     [M, num_groups] uint16
//   m_axi gmem_x     : INT8 activation vector       [N]
//   m_axi gmem_y     : fp32 output vector           [M]
//   s_axilite        : control + scalar args (maps to the register map in
//                      gemv_int4.hpp / hdk/register_map.md)
//
// Numeric contract is identical to gemv_int4_compute() in gemv_int4.hpp and to
// qwen_fpga/export/quant.py. This file is plain-C++-compilable for csim; the
// #pragma HLS lines are ignored by g++.
#include <cstdint>

#ifndef GEMV_MAX_N
#define GEMV_MAX_N 8192          // covers Qwen2.5-1.5B intermediate; bump for larger
#endif
#ifndef GEMV_MAX_GROUPS
#define GEMV_MAX_GROUPS (GEMV_MAX_N / 64 + 1)
#endif

// fp16 -> float for HLS. Vitis maps this to LUT logic; small and pipelineable.
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

static inline int sx4(uint8_t nib) {
#pragma HLS INLINE
    int v = nib & 0xF;
    return (v >= 8) ? (v - 16) : v;
}

extern "C" {

// One GEMV: y = dequant(W) @ x, with per-call activation scale x_scale.
void gemv_int4(
        const uint8_t*  gmem_w,       // [M * (N/2)]
        const uint16_t* gmem_scale,   // [M * num_groups]
        const int8_t*   gmem_x,       // [N]
        float*          gmem_y,       // [M]
        float           x_scale,
        int             M,
        int             N,
        int             group_size) {
#pragma HLS INTERFACE m_axi     port=gmem_w     offset=slave bundle=gmem_w     max_read_burst_length=64
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
#pragma HLS INTERFACE s_axilite port=group_size bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control

    const int half_n = (N + 1) >> 1;
    const int num_groups = (N + group_size - 1) / group_size;

    // Cache the activation vector on-chip once (reused across all M rows).
    int8_t xbuf[GEMV_MAX_N];
#pragma HLS BIND_STORAGE variable=xbuf type=ram_2p impl=uram
    load_x: for (int n = 0; n < N; ++n) {
#pragma HLS PIPELINE II=1
        xbuf[n] = gmem_x[n];
    }

    rows: for (int m = 0; m < M; ++m) {
        const uint8_t*  wrow = gmem_w     + (long)m * half_n;
        const uint16_t* srow = gmem_scale + (long)m * num_groups;
        float acc = 0.0f;
        groups: for (int g = 0; g < num_groups; ++g) {
            int32_t gacc = 0;
            const int n0 = g * group_size;
            const int n1 = (n0 + group_size < N) ? (n0 + group_size) : N;
            inner: for (int n = n0; n < n1; n += 2) {
#pragma HLS PIPELINE II=1
                uint8_t byte = wrow[n >> 1];
                gacc += sx4(byte & 0xF) * (int)xbuf[n];
                if (n + 1 < n1)
                    gacc += sx4(byte >> 4) * (int)xbuf[n + 1];
            }
            acc += (float)gacc * half_to_float_hw(srow[g]);
        }
        gmem_y[m] = acc * x_scale;
    }
}

}  // extern "C"
