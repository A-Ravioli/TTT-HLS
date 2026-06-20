"""Run C-simulation / HLS synthesis when a Vivado/Vitis toolchain is present.

Everything here degrades gracefully: if no toolchain is installed (the common
case on a laptop or CI runner), the functions report ``available=False`` instead
of raising, so the search loop keeps running on Python/C-level signals alone.
"""

from __future__ import annotations

import shutil
from typing import Any

from paths import get_logger

logger = get_logger("burnttt.run_hls")

# Candidate HLS executables across Vivado HLS and Vitis HLS generations.
HLS_EXECUTABLES = ("vitis_hls", "vivado_hls", "vivado")


def detect_hls_tool() -> str | None:
    """Return the first HLS executable found on PATH, else ``None``."""
    for exe in HLS_EXECUTABLES:
        path = shutil.which(exe)
        if path:
            return path
    return None


def hls_tool_available() -> bool:
    return detect_hls_tool() is not None


def run_build(
    hls_model,
    csim: bool = True,
    synth: bool = True,
    cosim: bool = False,
    export: bool = False,
) -> dict[str, Any]:
    """Run the backend build (C-sim + synthesis) if a toolchain is available.

    Returns a dict::

        {"available": bool, "synth_success": bool|None,
         "sim_success": bool|None, "error": str|None}
    """
    tool = detect_hls_tool()
    if tool is None:
        logger.info("No HLS toolchain (vitis_hls/vivado_hls) on PATH; skipping synthesis.")
        return {
            "available": False,
            "synth_success": None,
            "sim_success": None,
            "error": "no_hls_toolchain",
        }

    logger.info("Found HLS tool at %s; running build (csim=%s synth=%s)", tool, csim, synth)
    try:
        hls_model.build(csim=csim, synth=synth, cosim=cosim, export=export)
        return {
            "available": True,
            "synth_success": bool(synth),
            "sim_success": bool(csim),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - tool failures must be non-fatal
        logger.warning("HLS build failed: %s", exc)
        return {
            "available": True,
            "synth_success": False,
            "sim_success": False,
            "error": str(exc),
        }
