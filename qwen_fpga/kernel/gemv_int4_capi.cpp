// gemv_int4_capi.cpp -- C ABI wrapper around the shared GEMV datapath.
//
// Compiled to a shared library (libgemv_int4.so / .dylib) so the Python host
// runtime can drive the *exact same* C++ arithmetic that the HLS kernel runs,
// via ctypes. This is the software-fallback / bring-up backend: identical math
// to the FPGA, just executed on the CPU. Build: see qwen_fpga/Makefile.
#include "gemv_int4.hpp"

extern "C" void gemv_int4_capi(
        const uint8_t* W, const uint16_t* scales, const int8_t* x,
        float x_scale, float* y, int M, int N, int group_size) {
    gemv_int4_compute(W, scales, x, x_scale, y, M, N, group_size);
}
