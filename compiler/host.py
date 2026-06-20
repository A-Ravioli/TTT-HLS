"""XRT/DMA host stub: weight streaming, KV-cache read/write, kernel launch.

Minimal host-side interface for FPGA kernel execution. In Phase 4 this is a
stub that defines the interface; full implementation in Phase 6 when
multi-FPGA deployment is built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from paths import get_logger

logger = get_logger("burnttt.compiler.host")


@dataclass
class HostConfig:
    """Configuration for XRT host-side kernel execution."""

    bitstream_path: Path | None = None
    device_index: int = 0
    kernel_name: str = "kernel_top"
    axi_width_bits: int = 512
    dma_buffer_size_mb: int = 64
    weight_streaming: bool = False
    double_buffer: bool = True


@dataclass
class HostResult:
    """Result from an on-board kernel execution."""

    success: bool
    tokens_per_sec: float | None = None
    latency_us: float | None = None
    throughput_gbps: float | None = None
    error_msg: str = ""


def generate_host_code(
    config: HostConfig,
    input_size: int,
    output_size: int,
    weight_size: int = 0,
) -> str:
    """Generate XRT host code for kernel execution.

    This generates a C++ host program that uses XRT to:
    1. Load the bitstream (xclbin)
    2. Allocate device buffers
    3. Transfer input data + weights
    4. Launch the kernel
    5. Read back outputs
    6. Measure latency
    """
    return f"""\
// Auto-generated XRT host code for {config.kernel_name}
// Phase 4 stub — full weight streaming in Phase 6

#include <iostream>
#include <chrono>
#include <vector>
#include <cstring>

// XRT includes (uncomment when XRT is available)
// #include "xrt/xrt_bo.h"
// #include "xrt/xrt_device.h"
// #include "xrt/xrt_kernel.h"

constexpr int INPUT_SIZE = {input_size};
constexpr int OUTPUT_SIZE = {output_size};
constexpr int WEIGHT_SIZE = {weight_size};
constexpr int AXI_WIDTH_BYTES = {config.axi_width_bits // 8};

struct BenchmarkResult {{
    double latency_us;
    double throughput_gbps;
    double tokens_per_sec;
}};

#ifdef HAS_XRT
BenchmarkResult run_kernel(const char* xclbin_path, const float* input,
                           float* output, const float* weights, int n_tokens) {{
    auto device = xrt::device(0);
    auto uuid = device.load_xclbin(xclbin_path);
    auto kernel = xrt::kernel(device, uuid, "{config.kernel_name}");

    auto bo_in = xrt::bo(device, INPUT_SIZE * sizeof(float), kernel.group_id(0));
    auto bo_out = xrt::bo(device, OUTPUT_SIZE * sizeof(float), kernel.group_id(1));
    {"auto bo_w = xrt::bo(device, WEIGHT_SIZE * sizeof(float), kernel.group_id(2));" if weight_size > 0 else ""}

    // Transfer
    bo_in.write(input);
    bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    {"bo_w.write(weights); bo_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);" if weight_size > 0 else ""}

    // Execute and time
    auto start = std::chrono::high_resolution_clock::now();
    auto run = kernel(bo_in, bo_out{", bo_w" if weight_size > 0 else ""});
    run.wait();
    auto end = std::chrono::high_resolution_clock::now();

    // Read back
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_out.read(output);

    double us = std::chrono::duration<double, std::micro>(end - start).count();
    double gbps = (INPUT_SIZE + OUTPUT_SIZE) * sizeof(float) / us / 1000.0;
    double tps = n_tokens / (us / 1e6);
    return {{us, gbps, tps}};
}}
#else
BenchmarkResult run_kernel(const char*, const float*, float*, const float*, int) {{
    std::cerr << "XRT not available; returning stub result." << std::endl;
    return {{0.0, 0.0, 0.0}};
}}
#endif

int main(int argc, char** argv) {{
    if (argc < 2) {{
        std::cerr << "Usage: " << argv[0] << " <xclbin_path>" << std::endl;
        return 1;
    }}
    std::vector<float> input(INPUT_SIZE, 1.0f);
    std::vector<float> output(OUTPUT_SIZE, 0.0f);
    std::vector<float> weights(WEIGHT_SIZE, 0.0f);

    auto result = run_kernel(argv[1], input.data(), output.data(), weights.data(), 1);
    std::cout << "Latency: " << result.latency_us << " us" << std::endl;
    std::cout << "Throughput: " << result.throughput_gbps << " GB/s" << std::endl;
    std::cout << "Tokens/sec: " << result.tokens_per_sec << std::endl;
    return 0;
}}
"""


def estimate_tokens_per_sec(
    latency_cycles: int,
    clock_mhz: float = 300.0,
    batch_size: int = 1,
) -> float:
    """Estimate tokens/sec from cycle count and clock frequency."""
    if latency_cycles <= 0:
        return 0.0
    latency_sec = latency_cycles / (clock_mhz * 1e6)
    return batch_size / latency_sec
