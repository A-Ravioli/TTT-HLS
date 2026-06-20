"""Evaluate a single :class:`BurnConfig` end-to-end.

Pipeline (each stage degrades gracefully):

1. hls4ml conversion (Keras -> HLS project).
2. C-compile + bit-accurate prediction vs golden outputs (real quantization error).
3. HLS C-sim / synthesis if a Vivado/Vitis toolchain is available.
4. Report parsing for real latency/resource numbers.
5. Analytical hardware estimate as a fallback when synthesis is unavailable.
6. Reward computation.

Returns a flat result dict suitable for a CSV row.
"""

from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from tensorflow import keras

from compiler.build_hls4ml_project import build_project, compile_project
from compiler.estimate_resources import estimate_hardware
from compiler.parse_reports import REPORT_FIELDS, parse_reports
from compiler.run_hls import hls_tool_available, run_build
from models.export_model import load_golden, load_model
from paths import BUILD_DIR, get_logger, get_target_part
from ttt.config_space import BurnConfig
from ttt.reward import fits_board, resource_pct, reward

logger = get_logger("burnttt.evaluate")


def _prediction_errors(y_hls: np.ndarray, golden: np.ndarray) -> tuple[float, float]:
    y_hls = np.asarray(y_hls, dtype=np.float64).reshape(golden.shape)
    diff = np.abs(y_hls - golden.astype(np.float64))
    return float(diff.max()), float(diff.mean())


def evaluate_config(
    config: BurnConfig,
    model: keras.Model | None = None,
    x_test: np.ndarray | None = None,
    golden: np.ndarray | None = None,
    output_dir: str | Path | None = None,
    run_synth: bool = False,
    cleanup: bool = False,
) -> dict[str, Any]:
    """Evaluate ``config`` and return a result dict (see module docstring)."""
    t0 = time.time()
    if model is None:
        model = load_model()
    if x_test is None or golden is None:
        x_test, golden = load_golden()

    if output_dir is None:
        output_dir = BUILD_DIR / "eval" / f"{config.short_name()}_{uuid.uuid4().hex[:6]}"
    output_dir = Path(output_dir)

    result: dict[str, Any] = {
        **config.to_dict(),
        "config_name": config.short_name(),
        "target_part": get_target_part(),
        "compile_success": False,
        "sim_success": None,
        "synth_success": None,
        "max_error": None,
        "mean_error": None,
        "estimated_hw": False,
        "output_dir": str(output_dir),
    }
    for f in REPORT_FIELDS:
        result.setdefault(f, None)

    # --- Stage 1+2: convert, C-compile, bit-accurate predict ----------------
    try:
        hls_model = build_project(model, config, output_dir=output_dir)
        compiled = compile_project(hls_model)
        result["compile_success"] = compiled
        if compiled:
            y_hls = hls_model.predict(np.ascontiguousarray(x_test, dtype=np.float32))
            max_err, mean_err = _prediction_errors(y_hls, golden)
            result["max_error"] = max_err
            result["mean_error"] = mean_err
            result["sim_success"] = True  # C-level (compiled) prediction succeeded
            logger.info("%s: max_error=%.5f mean_error=%.5f", config.short_name(), max_err, mean_err)
    except Exception as exc:  # noqa: BLE001 - never crash the search on one config
        logger.warning("Conversion/compile failed for %s: %s", config.short_name(), exc)
        result["compile_success"] = False
        result["error_msg"] = str(exc)

    # --- Stage 3+4: real synthesis + report parsing (if toolchain present) --
    used_real_reports = False
    if result["compile_success"] and run_synth and hls_tool_available():
        build_status = run_build(hls_model, csim=True, synth=True)
        result["synth_success"] = build_status.get("synth_success")
        if build_status.get("sim_success") is not None:
            result["sim_success"] = build_status.get("sim_success")
        parsed = parse_reports(output_dir)
        if any(parsed.get(f) is not None for f in REPORT_FIELDS):
            for f in REPORT_FIELDS:
                result[f] = parsed.get(f)
            used_real_reports = True

    # --- Stage 5: analytical estimate fallback ------------------------------
    if result["compile_success"] and not used_real_reports:
        est = estimate_hardware(model, config)
        for k, v in est.items():
            if k == "estimated":
                continue
            result[k] = v
        result["estimated_hw"] = True

    # --- Stage 6: reward + derived fields -----------------------------------
    result["reward"] = reward(result)
    result["fits_board"] = fits_board(result)
    result.update(resource_pct(result))
    result["eval_seconds"] = round(time.time() - t0, 2)

    if cleanup and output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)

    return result
