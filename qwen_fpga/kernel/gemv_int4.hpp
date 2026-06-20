// gemv_int4.hpp -- shared INT4 groupwise GEMV datapath + AXI-Lite register map.
//
// The *same* inner math is used by three callers:
//   * gemv_int4_ref.cpp  -- portable C++ functional reference (host + CI test)
//   * gemv_int4_hls.cpp  -- Vitis HLS kernel for the AWS F2 custom logic
//   * the C++ host runtime software-fallback path
//
// Numeric contract (must match qwen_fpga/export/quant.py exactly):
//   y[m] = x_scale * sum_g  w_scale[m,g] * ( sum_{n in group g} wq[m,n] * xq[n] )
//   wq : signed INT4 in [-8,7], packed 2-per-byte (col n -> low nibble if n even)
//   xq : signed INT8, single per-call scale
//   group accumulation in INT32; output accumulation in float.
#ifndef QWEN_FPGA_GEMV_INT4_HPP
#define QWEN_FPGA_GEMV_INT4_HPP

#include <cstdint>
#include <cstddef>
#include <cstring>

// ---------------------------------------------------------------------------
// AXI-Lite control/status register map (byte offsets). Matches hdk/register_map.md.
// ---------------------------------------------------------------------------
#define GEMV_REG_CONTROL        0x000  // bit0 start, bit1 reset
#define GEMV_REG_STATUS         0x004  // bit0 done, bit1 busy, bit2 error
#define GEMV_REG_W_BASE_LO      0x010
#define GEMV_REG_W_BASE_HI      0x014
#define GEMV_REG_X_BASE_LO      0x018
#define GEMV_REG_X_BASE_HI      0x01c
#define GEMV_REG_Y_BASE_LO      0x020
#define GEMV_REG_Y_BASE_HI      0x024
#define GEMV_REG_SCALE_BASE_LO  0x028
#define GEMV_REG_SCALE_BASE_HI  0x02c
#define GEMV_REG_M              0x030
#define GEMV_REG_N              0x034
#define GEMV_REG_GROUP_SIZE     0x038
#define GEMV_REG_FLAGS          0x03c  // bit0 x_scale present in XSCALE reg
#define GEMV_REG_XSCALE_BITS    0x040  // float bits of x_scale (host writes as u32)

#define GEMV_CTRL_START         0x1u
#define GEMV_CTRL_RESET         0x2u
#define GEMV_STAT_DONE          0x1u
#define GEMV_STAT_BUSY          0x2u
#define GEMV_STAT_ERROR         0x4u

// IEEE-754 half (binary16) -> float. Branchy but exact; fine for host/CI use.
static inline float gemv_half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;  // +/- 0
        } else {
            // subnormal: normalize
            int e = -1;
            uint32_t m = mant;
            do { m <<= 1; e++; } while ((m & 0x400u) == 0);
            m &= 0x3FFu;
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);  // inf / nan
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    __builtin_memcpy(&f, &bits, sizeof(f));
    return f;
}

// Sign-extend a 4-bit value held in the low nibble of `nib`.
static inline int gemv_sx4(uint8_t nib) {
    int v = nib & 0xF;
    return (v >= 8) ? (v - 16) : v;
}

// Core datapath. W is packed [M, N/2] uint8; scales are fp16 [M, num_groups];
// x is INT8 [N]; y is float [M]. group_size must divide the padded N tiling
// (last group may be partial -- handled by the n < N guard).
//
// Kept header-only and pointer-based so the HLS kernel can wrap it with
// interface pragmas while host/CI compile the identical arithmetic.
static inline void gemv_int4_compute(
        const uint8_t* W, const uint16_t* scales, const int8_t* x,
        float x_scale, float* y,
        int M, int N, int group_size) {
    const int half_n = (N + 1) / 2;
    const int num_groups = (N + group_size - 1) / group_size;
    for (int m = 0; m < M; ++m) {
        const uint8_t* wrow = W + (size_t)m * half_n;
        const uint16_t* srow = scales + (size_t)m * num_groups;
        float acc = 0.0f;
        for (int g = 0; g < num_groups; ++g) {
            int32_t gacc = 0;
            const int n0 = g * group_size;
            const int n1 = (n0 + group_size < N) ? (n0 + group_size) : N;
            for (int n = n0; n < n1; ++n) {
                const uint8_t byte = wrow[n >> 1];
                const uint8_t nib = (n & 1) ? (byte >> 4) : (byte & 0xF);
                gacc += gemv_sx4(nib) * (int)x[n];
            }
            acc += (float)gacc * gemv_half_to_float(srow[g]);
        }
        y[m] = acc * x_scale;
    }
}

#endif  // QWEN_FPGA_GEMV_INT4_HPP
