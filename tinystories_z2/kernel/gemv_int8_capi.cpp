// gemv_int8_capi.cpp -- C ABI wrapper around the shared W8A8 GEMV datapath.
//
// Compiled to libgemv_int8.{so,dylib} so the Python host runtime can drive the
// *exact same* C++ arithmetic the HLS kernel runs, via ctypes. This is the
// software-fallback / bring-up backend: identical math to the FPGA, on the CPU.
// Build: see tinystories_z2/Makefile.
#include "gemv_int8.hpp"

extern "C" void gemv_int8_capi(
        const int8_t* W, const uint16_t* scales, const int8_t* x,
        float x_scale, float* y, int M, int N) {
    gemv_int8_compute(W, scales, x, x_scale, y, M, N);
}
