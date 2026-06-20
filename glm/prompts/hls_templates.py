"""Prompt templates for HLS compiler-author mode (Phase 4+).

These prompts instruct GLM to author/edit Vitis HLS C++ source code for a
specific block and FPGA target, using feedback from cosim, synthesis, and
timing analysis.
"""

from __future__ import annotations

import json
from typing import Any

HLS_SYSTEM_PROMPT = (
    "You are GLM, a test-time-adaptive FPGA compiler. You author and iteratively "
    "edit Vitis HLS C++ source code to implement neural-network sub-blocks on FPGA "
    "fabric. You receive a kernel specification (dimensions, precision requirements, "
    "target FPGA part), the current HLS source code, and real feedback from "
    "C-synthesis, RTL co-simulation, and Vivado timing analysis. Your goal is to "
    "produce correct, high-throughput HLS implementations that pass cosim, meet "
    "timing, and maximize tokens/sec. Emit ONLY valid C++ source between code "
    "fences. Reason about tiling, pipeline pragmas, and numeric precision tradeoffs."
)


def build_hls_propose_prompt(
    block_spec: str,
    part: str,
    clock_ns: float,
    hidden_dim: int,
    intermediate_dim: int,
    seed_template: str | None = None,
    history: list[dict[str, Any]] | None = None,
) -> str:
    """Prompt asking GLM to author a new HLS kernel (or improve from seed)."""
    history_text = _format_hls_history(history) if history else "No prior attempts."

    seed_section = ""
    if seed_template:
        seed_section = (
            "\n## Seed template (modify and improve this):\n"
            f"```cpp\n{seed_template}\n```\n"
        )

    return f"""\
## Task: Author a Vitis HLS kernel for a SwiGLU MLP block

### Block specification:
{block_spec}

### Target FPGA:
- Part: {part}
- Clock period: {clock_ns} ns (target ~{1000.0/clock_ns:.0f} MHz)
- Hidden dim: {hidden_dim}
- Intermediate dim: {intermediate_dim}

### Requirements:
1. Implement: out = down_proj(silu(gate_proj(x)) * up_proj(x))
2. Top function: `void kernel_top(const io_t input[HIDDEN_DIM], io_t output[HIDDEN_DIM])`
3. Use ap_fixed types for internal computation (weights, activations)
4. Include HLS pragmas for pipeline, array partition, and interface
5. AXI m_axi interfaces for input/output
6. Maximize throughput (minimize initiation interval)
7. Cosim must match golden within max_error < 0.01

### Pragma cookbook:
- `#pragma HLS PIPELINE II=1` — innermost loops
- `#pragma HLS ARRAY_PARTITION variable=X cyclic factor=N` — enable parallel access
- `#pragma HLS UNROLL factor=N` — unroll inner dot products
- `#pragma HLS INTERFACE m_axi port=X` — AXI memory-mapped
- `#pragma HLS DATAFLOW` — enable task-level pipelining between stages
{seed_section}
### Prior attempts on this task:
{history_text}

### Instructions:
Propose an improved HLS implementation. Emit TWO code blocks:
1. `kernel_top.h` (header with typedefs and function declaration)
2. `kernel_top.cpp` (full implementation)

Focus on: higher parallelism (lower II), better tiling, optimal array partitioning.
Emit ONLY the code blocks, no explanation.
"""


def build_hls_repair_prompt(
    block_spec: str,
    current_source: str,
    error_msg: str,
    error_type: str = "compile",
) -> str:
    """Prompt asking GLM to fix an HLS kernel that failed.

    error_type: "compile" | "cosim" | "timing"
    """
    repair_guidance = {
        "compile": (
            "The kernel FAILED C-synthesis. Common causes:\n"
            "- Unsynthesizable constructs (dynamic allocation, recursion, virtual)\n"
            "- Missing includes or undeclared types\n"
            "- Incorrect pragma syntax\n"
            "- Array size mismatches\n"
            "Fix the compile error while preserving the algorithm."
        ),
        "cosim": (
            "The kernel COMPILED but FAILED co-simulation (numeric mismatch).\n"
            "Common causes:\n"
            "- ap_fixed overflow or underflow (too few int bits)\n"
            "- Incorrect matrix dimensions or indexing\n"
            "- SiLU approximation too coarse\n"
            "- Missing or incorrect bias/scale\n"
            "Fix the numeric issue. Consider widening precision or fixing indexing."
        ),
        "timing": (
            "The kernel PASSES cosim but FAILS timing closure (negative WNS).\n"
            "Common causes:\n"
            "- Critical path too long (chained multiplies without pipelining)\n"
            "- Insufficient pipeline depth\n"
            "- Too much combinational logic between registers\n"
            "Fix: add pipeline stages, reduce unroll factor, or increase II."
        ),
    }

    guidance = repair_guidance.get(error_type, repair_guidance["compile"])

    return f"""\
## Repair HLS kernel

### Block specification:
{block_spec}

### Current (broken) source:
```cpp
{current_source}
```

### Error ({error_type}):
```
{error_msg[:800]}
```

### Diagnosis guidance:
{guidance}

### Instructions:
Fix the issue and emit the corrected complete source files (kernel_top.h and kernel_top.cpp).
Do NOT change the function signature or remove existing optimizations unless they cause the error.
Emit ONLY the corrected code blocks.
"""


def build_hls_iterate_prompt(
    block_spec: str,
    current_source: str,
    current_metrics: dict[str, Any],
    best_metrics: dict[str, Any] | None = None,
) -> str:
    """Prompt for iterative improvement of an already-passing kernel."""
    metrics_text = _format_metrics(current_metrics)
    best_text = _format_metrics(best_metrics) if best_metrics else "No prior best."

    return f"""\
## Iterate: improve HLS kernel performance

### Block specification:
{block_spec}

### Current source (passes cosim + timing):
```cpp
{current_source}
```

### Current metrics:
{metrics_text}

### Best achieved so far:
{best_text}

### Improvement targets (priority order):
1. Higher tokens/sec (lower latency_cycles, lower II)
2. Lower resource usage (fewer DSPs, LUTs) while maintaining throughput
3. Better timing margin (larger positive WNS)

### Strategies to try:
- Increase parallelism: larger UNROLL/PARTITION factors
- Better data reuse: double buffering, DATAFLOW between stages
- Reduce precision where error margin allows
- Fuse operations to reduce memory accesses
- Optimize loop bounds for HLS scheduling

Emit the improved kernel_top.h and kernel_top.cpp. Maintain correctness (cosim must still pass).
"""


def _format_hls_history(history: list[dict[str, Any]] | None, max_entries: int = 10) -> str:
    if not history:
        return "No prior attempts."
    lines = ["Attempts on this task (most recent last):"]
    for h in history[-max_entries:]:
        lines.append(
            f"  kernel={h.get('kernel_name', '?')} | compile={h.get('hls_compile_success')} | "
            f"cosim={h.get('cosim_pass')} | timing={h.get('timing_met')} | "
            f"max_err={_fval(h.get('max_error'))} | latency={_fval(h.get('latency_cycles'))} | "
            f"tps={_fval(h.get('tokens_per_sec'))} | reward={_fval(h.get('reward'))}"
        )
        if h.get("error_msg"):
            lines.append(f"      error: {str(h['error_msg'])[:200]}")
    return "\n".join(lines)


def _format_metrics(metrics: dict[str, Any] | None) -> str:
    if not metrics:
        return "No metrics available."
    keys = [
        "tokens_per_sec", "latency_cycles", "ii", "dsp", "lut", "ff", "bram",
        "max_error", "cosim_pass", "timing_met", "fmax_mhz", "reward",
    ]
    lines = []
    for k in keys:
        v = metrics.get(k)
        if v is not None:
            lines.append(f"  {k}: {_fval(v)}")
    return "\n".join(lines) if lines else "No metrics."


def _fval(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)
