"""C/RTL cosimulation harness: compare HLS output vs golden reference I/O.

Two modes:
1. **Vitis HLS cosim** (real): runs ``cosim_design`` via TCL if toolchain present.
2. **Software cosim** (fallback): compiles the C++ kernel source with g++ and
   checks outputs against golden vectors — available everywhere, catches numeric
   bugs but not RTL-level issues.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from compiler.golden import GoldenIO
from compiler.hls_build import HLSBuildResult, run_cosim as _vitis_cosim, vitis_hls_available
from paths import get_logger

logger = get_logger("burnttt.compiler.cosim")


@lru_cache(maxsize=1)
def ap_types_include_dirs() -> tuple[str, ...]:
    """Locate Xilinx ``ap_*`` headers so software cosim can compile ``ap_fixed.h``.

    The kernels ``#include <ap_fixed.h>``; without these on the include path g++
    fails before any numeric check runs. Sources, in order: ``BURN_AP_TYPES_INCLUDE``
    (colon-separated), then hls4ml's bundled ``ap_types`` directory if installed.
    """
    dirs: list[str] = []
    env = os.environ.get("BURN_AP_TYPES_INCLUDE", "").strip()
    if env:
        dirs.extend(p for p in env.split(os.pathsep) if p)
    try:
        import hls4ml  # noqa: F401

        cand = Path(hls4ml.__file__).parent / "templates" / "vivado" / "ap_types"
        if cand.is_dir():
            dirs.append(str(cand))
    except Exception:  # noqa: BLE001 - hls4ml is optional
        pass
    # De-dup while preserving order.
    seen: set[str] = set()
    return tuple(d for d in dirs if not (d in seen or seen.add(d)))


@dataclass
class CosimResult:
    """Result of a cosimulation run."""

    passed: bool
    max_error: float
    mean_error: float
    method: str  # "vitis" or "software"
    error_msg: str = ""
    details: dict[str, Any] | None = None


def run_cosim_vs_golden(
    project_dir: Path,
    golden: GoldenIO,
    max_error_threshold: float = 0.01,
    prefer_vitis: bool = True,
) -> CosimResult:
    """Run cosim and compare against golden outputs.

    If Vitis HLS is available and ``prefer_vitis`` is True, uses RTL cosim.
    Otherwise falls back to software C compilation + comparison.
    """
    if prefer_vitis and vitis_hls_available():
        return _vitis_cosim_vs_golden(project_dir, golden, max_error_threshold)
    return _software_cosim(project_dir, golden, max_error_threshold)


def _vitis_cosim_vs_golden(
    project_dir: Path, golden: GoldenIO, threshold: float
) -> CosimResult:
    """Run Vitis HLS cosim and check pass/fail."""
    # Write golden data files for the testbench to read
    golden.save(project_dir / "golden_data")

    result = _vitis_cosim(project_dir)
    if result.cosim_pass:
        return CosimResult(
            passed=True,
            max_error=0.0,
            mean_error=0.0,
            method="vitis",
        )
    return CosimResult(
        passed=False,
        max_error=float("inf"),
        mean_error=float("inf"),
        method="vitis",
        error_msg=result.error_msg,
    )


def _software_cosim(
    project_dir: Path, golden: GoldenIO, threshold: float
) -> CosimResult:
    """Compile kernel C++ with g++ and run against golden vectors.

    This is a lightweight cosim that catches numerical bugs without needing
    Vitis HLS. The testbench reads golden inputs, runs the kernel function,
    and writes outputs to a file. We compare the outputs here.
    """
    src_dir = project_dir / "src"
    if not src_dir.exists():
        return CosimResult(
            passed=False, max_error=float("inf"), mean_error=float("inf"),
            method="software", error_msg="No src/ directory in project",
        )

    # Write golden inputs for the software testbench
    golden.save(project_dir / "golden_data")

    # Generate a software testbench that exercises the kernel
    tb_source = _generate_sw_testbench(project_dir, golden)
    tb_path = project_dir / "sw_tb.cpp"
    tb_path.write_text(tb_source)

    # Compile
    src_files = list(src_dir.glob("*.cpp"))
    if not src_files:
        return CosimResult(
            passed=False, max_error=float("inf"), mean_error=float("inf"),
            method="software", error_msg="No .cpp source files found",
        )

    exe_path = project_dir / "sw_cosim"
    ap_includes: list[str] = []
    for d in ap_types_include_dirs():
        ap_includes += ["-I", d]
    compile_cmd = [
        "g++", "-O2", "-std=c++14",
        "-I", str(src_dir),
        "-I", str(project_dir),
        *ap_includes,
        str(tb_path),
        *[str(f) for f in src_files],
        "-o", str(exe_path),
        "-lm",
    ]

    try:
        proc = subprocess.run(
            compile_cmd, capture_output=True, text=True, timeout=60,
        )
        if proc.returncode != 0:
            return CosimResult(
                passed=False, max_error=float("inf"), mean_error=float("inf"),
                method="software",
                error_msg=f"Compile failed:\n{proc.stderr[:500]}",
            )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return CosimResult(
            passed=False, max_error=float("inf"), mean_error=float("inf"),
            method="software", error_msg=f"Compile error: {exc}",
        )

    # Run
    try:
        proc = subprocess.run(
            [str(exe_path)],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            return CosimResult(
                passed=False, max_error=float("inf"), mean_error=float("inf"),
                method="software",
                error_msg=f"Execution failed:\n{proc.stderr[:500]}",
            )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return CosimResult(
            passed=False, max_error=float("inf"), mean_error=float("inf"),
            method="software", error_msg=f"Execution error: {exc}",
        )

    # Compare outputs
    output_path = project_dir / "sw_output.bin"
    if not output_path.exists():
        return CosimResult(
            passed=False, max_error=float("inf"), mean_error=float("inf"),
            method="software", error_msg="No output file produced",
        )

    try:
        hw_out = np.fromfile(str(output_path), dtype=np.float32)
        golden_out = next(iter(golden.outputs.values())).flatten().astype(np.float32)
        if hw_out.shape != golden_out.shape:
            # Try to reshape
            hw_out = hw_out[: golden_out.size]
            if hw_out.size != golden_out.size:
                return CosimResult(
                    passed=False, max_error=float("inf"), mean_error=float("inf"),
                    method="software",
                    error_msg=f"Shape mismatch: got {hw_out.size}, expected {golden_out.size}",
                )

        diff = np.abs(hw_out - golden_out)
        max_err = float(diff.max())
        mean_err = float(diff.mean())
        passed = max_err <= threshold

        logger.info("Software cosim: max_error=%.6f mean_error=%.6f (threshold=%.4f) -> %s",
                    max_err, mean_err, threshold, "PASS" if passed else "FAIL")
        return CosimResult(
            passed=passed, max_error=max_err, mean_error=mean_err, method="software",
        )
    except Exception as exc:  # noqa: BLE001
        return CosimResult(
            passed=False, max_error=float("inf"), mean_error=float("inf"),
            method="software", error_msg=f"Output comparison failed: {exc}",
        )


def _generate_sw_testbench(project_dir: Path, golden: GoldenIO) -> str:
    """Generate a minimal C++ testbench for software cosim."""
    in_tensor = next(iter(golden.inputs.values()))
    out_tensor = next(iter(golden.outputs.values()))
    n_samples = in_tensor.shape[0]
    in_size = int(np.prod(in_tensor.shape[1:]))
    out_size = int(np.prod(out_tensor.shape[1:]))

    return f"""\
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "kernel_top.h"

// Read binary numpy data (raw float32 after npy header skip)
static float* read_npy_data(const char* path, int expected_elems) {{
    FILE* f = fopen(path, "rb");
    if (!f) {{ fprintf(stderr, "Cannot open %s\\n", path); exit(1); }}
    // Skip npy header: find '\\n' after magic + header
    char buf[256];
    fread(buf, 1, 10, f);  // magic + version + header_len
    unsigned short hlen = *(unsigned short*)(buf + 8);
    fseek(f, 10 + hlen, SEEK_SET);
    float* data = (float*)malloc(expected_elems * sizeof(float));
    fread(data, sizeof(float), expected_elems, f);
    fclose(f);
    return data;
}}

int main() {{
    const int N = {n_samples};
    const int IN_SIZE = {in_size};
    const int OUT_SIZE = {out_size};

    float* inputs = read_npy_data("golden_data/input_x.npy", N * IN_SIZE);
    float* outputs = (float*)malloc(N * OUT_SIZE * sizeof(float));

    for (int i = 0; i < N; i++) {{
        kernel_top(inputs + i * IN_SIZE, outputs + i * OUT_SIZE);
    }}

    // Write output
    FILE* fout = fopen("sw_output.bin", "wb");
    fwrite(outputs, sizeof(float), N * OUT_SIZE, fout);
    fclose(fout);

    free(inputs);
    free(outputs);
    return 0;
}}
"""
