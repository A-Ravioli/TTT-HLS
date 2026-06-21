// gemv_int8.hpp -- shared W8A8 GEMV datapath + AXI-Lite register map (Zynq-7020).
//
// The *same* inner math is used by three callers:
//   * gemv_int8_capi.cpp -- C ABI shared lib for the host runtime / CI golden test
//   * gemv_int8_hls.cpp  -- Vitis HLS kernel for the PYNQ Z2 (Zynq-7020) PL
//   * tinystories_z2/quant.py::gemv_int8_quantized (the Python reference)
//
// Numeric contract (must match tinystories_z2/quant.py exactly):
//   y[m] = x_scale * w_scale[m] * sum_n ( wq[m,n] * xq[n] )
//   wq : signed INT8 [M,N] row-major
//   xq : signed INT8 [N], single per-call x_scale
//   w_scale : fp16 per output row [M]
//   inner accumulation in INT32; output accumulation in float.
//
// This is the INT4 kernel's simpler cousin: no nibble packing, no group loop --
// byte-aligned so the Z7020's small fabric synthesizes a tight, fast datapath.
#ifndef TS_Z2_GEMV_INT8_HPP
#define TS_Z2_GEMV_INT8_HPP

#include <cstdint>
#include <cstddef>
#include <cstring>

// ---------------------------------------------------------------------------
// AXI-Lite control/status register map (byte offsets). Matches hdk/register_map.md.
// On the Z2 the *_BASE registers hold PS-DDR physical addresses (from
// pynq.allocate(...).device_address), not HBM offsets.
// ---------------------------------------------------------------------------
#define GEMV_REG_CONTROL        0x000  // bit0 start, bit1 reset (ap_ctrl)
#define GEMV_REG_STATUS         0x004  // bit0 done, bit1 idle, bit2 ready
#define GEMV_REG_W_BASE_LO      0x010
#define GEMV_REG_W_BASE_HI      0x014
#define GEMV_REG_SCALE_BASE_LO  0x018
#define GEMV_REG_SCALE_BASE_HI  0x01c
#define GEMV_REG_X_BASE_LO      0x020
#define GEMV_REG_X_BASE_HI      0x024
#define GEMV_REG_Y_BASE_LO      0x028
#define GEMV_REG_Y_BASE_HI      0x02c
#define GEMV_REG_XSCALE_BITS    0x030  // IEEE-754 fp32 bits of x_scale (host writes u32)
#define GEMV_REG_M              0x034
#define GEMV_REG_N              0x038

#define GEMV_CTRL_START         0x1u
#define GEMV_STAT_DONE          0x2u

// IEEE-754 half (binary16) -> float. Branchy but exact; fine for host/CI use and
// small enough to map to LUT logic in HLS.
static inline float gemv_half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;  // +/- 0
        } else {
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

// Core datapath. W is INT8 [M, N] row-major; scales fp16 [M]; x INT8 [N];
// y float [M]. Header-only and pointer-based so the HLS kernel can wrap it with
// interface pragmas while host/CI compile the identical arithmetic.
static inline void gemv_int8_compute(
        const int8_t* W, const uint16_t* scales, const int8_t* x,
        float x_scale, float* y, int M, int N) {
    for (int m = 0; m < M; ++m) {
        const int8_t* wrow = W + (size_t)m * N;
        int32_t acc = 0;
        for (int n = 0; n < N; ++n) {
            acc += (int)wrow[n] * (int)x[n];
        }
        y[m] = (float)acc * gemv_half_to_float(scales[m]) * x_scale;
    }
}

#endif  // TS_Z2_GEMV_INT8_HPP
