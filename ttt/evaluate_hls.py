"""Evaluate a :class:`KernelBundle` end-to-end through the custom HLS pipeline.

Pipeline (mirrors evaluate_config.py but for compiler-author mode):

1. Write HLS project from KernelBundle sources.
2. Run C-synthesis (Vitis HLS) or skip if unavailable.
3. Run cosim vs golden (RTL or software fallback).
4. Optionally run Vivado P&R for post-route timing/resources.
5. Compute tokens/sec (from cycle count + clock, or on-board).
6. Reward computation via reward_hls.

Returns a flat result dict suitable for a CSV row.
"""

from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from compiler.cosim import CosimResult, run_cosim_vs_golden
from compiler.golden import GoldenIO
from compiler.hls_build import (
    HLSBuildResult,
    run_csynth,
    vitis_hls_available,
    write_hls_project,
)
from compiler.host import estimate_tokens_per_sec
from compiler.vivado import VivadoResult, estimate_post_route, run_vivado_synth, vivado_available
from paths import BUILD_DIR, get_logger
from ttt.config_space import KernelBundle
from ttt.reward_hls import reward_hls

logger = get_logger("burnttt.evaluate_hls")


def evaluate_hls(
    bundle: KernelBundle,
    golden: GoldenIO,
    output_dir: str | Path | None = None,
    run_vivado: bool = False,
    max_error_threshold: float = 0.01,
    cleanup: bool = False,
) -> dict[str, Any]:
    """Evaluate a KernelBundle and return a result dict.

    Stages:
    1. Write HLS project
    2. C-synthesis (if Vitis HLS available)
    3. Cosim vs golden
    4. Vivado P&R (if requested and available)
    5. Tokens/sec estimate
    6. Reward
    """
    t0 = time.time()

    if output_dir is None:
        output_dir = BUILD_DIR / "eval_hls" / f"{bundle.short_name()}_{uuid.uuid4().hex[:6]}"
    output_dir = Path(output_dir)

    result: dict[str, Any] = {
        "kernel_name": bundle.short_name(),
        "top_function": bundle.top_function,
        "part": bundle.part,
        "clock_ns": bundle.clock_ns,
        "hidden_dim": bundle.hidden_dim,
        "intermediate_dim": bundle.intermediate_dim,
        "weight_bits": bundle.weight_bits,
        "act_bits": bundle.act_bits,
        "tile_hidden": bundle.tile_hidden,
        "tile_inter": bundle.tile_inter,
        "hls_compile_success": False,
        "cosim_pass": False,
        "timing_met": False,
        "max_error": None,
        "mean_error": None,
        "max_error_threshold": max_error_threshold,
        "latency_cycles": None,
        "ii": None,
        "dsp": None,
        "lut": None,
        "ff": None,
        "bram": None,
        "fmax_mhz": None,
        "tokens_per_sec": None,
        "power_w": None,
        "wns_violation_ns": 0.0,
        "output_dir": str(output_dir),
        "error_msg": "",
    }

    # --- Stage 1: Write HLS project ----------------------------------------
    try:
        # Separate header from cpp for testbench generation
        testbench_src = bundle.sources.get("tb_main.cpp", _default_testbench())
        kernel_sources = {k: v for k, v in bundle.sources.items() if k != "tb_main.cpp"}

        write_hls_project(
            kernel_sources=kernel_sources,
            testbench_source=testbench_src,
            project_dir=output_dir,
            top_function=bundle.top_function,
            part=bundle.part,
            clock_ns=str(bundle.clock_ns),
        )
    except Exception as exc:  # noqa: BLE001
        result["error_msg"] = f"Project write failed: {exc}"
        result["reward"] = reward_hls(result)
        return result

    # --- Stage 2: C-synthesis -----------------------------------------------
    csynth_result: HLSBuildResult | None = None
    if vitis_hls_available():
        csynth_result = run_csynth(output_dir)
        result["hls_compile_success"] = csynth_result.success
        if csynth_result.success:
            result["latency_cycles"] = csynth_result.latency_cycles
            result["ii"] = csynth_result.ii
            result["dsp"] = csynth_result.dsp
            result["lut"] = csynth_result.lut
            result["ff"] = csynth_result.ff
            result["bram"] = csynth_result.bram
        else:
            result["error_msg"] = csynth_result.error_msg
    else:
        # CI mode: mark compile as "success" so cosim can proceed via software path
        result["hls_compile_success"] = True
        logger.info("No Vitis HLS; skipping C-synthesis (software cosim path).")

    # --- Stage 3: Cosim vs golden -------------------------------------------
    if result["hls_compile_success"]:
        cosim = run_cosim_vs_golden(
            output_dir, golden,
            max_error_threshold=max_error_threshold,
            prefer_vitis=vitis_hls_available(),
        )
        result["cosim_pass"] = cosim.passed
        result["max_error"] = cosim.max_error
        result["mean_error"] = cosim.mean_error
        if not cosim.passed and not result["error_msg"]:
            result["error_msg"] = cosim.error_msg or f"Cosim failed: max_error={cosim.max_error:.6f}"

    # --- Stage 4: Vivado P&R -----------------------------------------------
    if result["cosim_pass"] and run_vivado:
        if vivado_available() and csynth_result and csynth_result.success:
            vivado_res = run_vivado_synth(output_dir, part=bundle.part, clock_ns=bundle.clock_ns)
            if vivado_res.success:
                result["timing_met"] = vivado_res.timing_met
                result["wns_violation_ns"] = max(0, -(vivado_res.wns_ns or 0))
                result["fmax_mhz"] = vivado_res.fmax_mhz
                result["power_w"] = vivado_res.power_w
                # Use post-route resource numbers if available
                if vivado_res.lut is not None:
                    result["lut"] = vivado_res.lut
                if vivado_res.ff is not None:
                    result["ff"] = vivado_res.ff
                if vivado_res.dsp is not None:
                    result["dsp"] = vivado_res.dsp
        elif csynth_result and csynth_result.success:
            # Estimate post-route from HLS numbers
            est = estimate_post_route(csynth_result, bundle.clock_ns)
            result["timing_met"] = est.timing_met
            result["fmax_mhz"] = est.fmax_mhz
            if est.lut is not None:
                result["lut"] = est.lut
    elif result["cosim_pass"] and not run_vivado:
        # Default: assume timing met for reward calc when not running Vivado
        result["timing_met"] = True

    # --- Stage 5: Tokens/sec estimate ---------------------------------------
    if result["cosim_pass"]:
        lat = result.get("latency_cycles")
        if lat and lat > 0:
            clock_mhz = result.get("fmax_mhz") or (1000.0 / bundle.clock_ns)
            result["tokens_per_sec"] = estimate_tokens_per_sec(lat, clock_mhz)
        else:
            # Rough analytical estimate based on dimensions
            total_ops = 2 * bundle.hidden_dim * bundle.intermediate_dim * 3  # 3 matmuls
            assumed_clock_mhz = 300.0
            # Assume throughput ~ (ops / cycle) * clock; very rough
            cycles_est = total_ops // max(1, bundle.tile_hidden * bundle.tile_inter)
            result["latency_cycles"] = cycles_est
            result["tokens_per_sec"] = estimate_tokens_per_sec(cycles_est, assumed_clock_mhz)
            result["timing_met"] = True  # Estimated

    # --- Stage 6: Reward ----------------------------------------------------
    result["reward"] = reward_hls(result)
    result["eval_seconds"] = round(time.time() - t0, 2)

    if cleanup and output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)

    logger.info(
        "%s: cosim=%s max_err=%s tps=%s reward=%.1f (%.1fs)",
        bundle.short_name(),
        result["cosim_pass"],
        f"{result['max_error']:.5f}" if result["max_error"] is not None else "n/a",
        f"{result['tokens_per_sec']:.0f}" if result["tokens_per_sec"] else "n/a",
        result["reward"],
        result["eval_seconds"],
    )
    return result


def _default_testbench() -> str:
    """Minimal testbench when none is provided in the bundle."""
    return """\
#include <cstdio>
#include "kernel_top.h"

int main() {
    printf("No custom testbench; use software cosim.\\n");
    return 0;
}
"""
