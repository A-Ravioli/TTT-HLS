"""Optional real-FPGA deployment for PYNQ-style boards.

On a real PYNQ board (with the ``pynq`` package and a generated ``.bit`` +
``.hwh`` overlay) this loads the overlay, streams an input vector through the
accelerator, and compares the FPGA output against the golden output.

Off-board (no ``pynq``, no bitstream) every entry point degrades to a clear,
actionable message describing exactly what is missing and where the generated
project lives. It never raises.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from paths import get_logger

logger = get_logger("burnttt.deploy")


def pynq_available() -> bool:
    """True if the ``pynq`` runtime is importable (i.e. we're on a board)."""
    try:
        import pynq  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def find_bitstream(build_dir: str | Path) -> Path | None:
    """Locate a generated ``.bit`` file under the build directory, if any."""
    build_dir = Path(build_dir)
    if not build_dir.exists():
        return None
    bits = sorted(build_dir.rglob("*.bit"))
    return bits[0] if bits else None


def deploy_and_run(
    build_dir: str | Path,
    x: np.ndarray,
    golden: np.ndarray | None = None,
) -> dict[str, Any]:
    """Attempt a real FPGA inference run; return a structured status dict.

    Always returns a dict with at least ``{"deployed": bool, "reason": str}``.
    """
    build_dir = Path(build_dir)
    bitfile = find_bitstream(build_dir)

    if not pynq_available():
        msg = (
            "pynq runtime not available — not running on a PYNQ board. "
            f"Generated project is at: {build_dir}"
        )
        logger.warning(msg)
        return {"deployed": False, "reason": "no_pynq_runtime", "build_dir": str(build_dir)}

    if bitfile is None:
        msg = (
            f"No .bit bitstream found under {build_dir}. Build one with "
            "scripts/03_build_best_bitstream.py on a machine with Vivado + the "
            "VivadoAccelerator backend."
        )
        logger.warning(msg)
        return {"deployed": False, "reason": "no_bitstream", "build_dir": str(build_dir)}

    # On-board path. Import lazily so the module stays importable off-board.
    try:
        from pynq import Overlay, allocate  # type: ignore

        logger.info("Loading overlay: %s", bitfile)
        overlay = Overlay(str(bitfile))

        # hls4ml's VivadoAccelerator overlays expose a NeuralNetworkOverlay
        # helper; prefer it when present, else fall back to manual DMA.
        try:
            from hls4ml.report import NeuralNetworkOverlay  # type: ignore

            nn = NeuralNetworkOverlay(str(bitfile), x.shape, golden.shape if golden is not None else x.shape)
            y_fpga, _, _ = nn.predict(x.astype(np.float32), profile=False)
        except Exception:  # noqa: BLE001
            in_buf = allocate(shape=x.shape, dtype=np.float32)
            out_shape = golden.shape if golden is not None else x.shape
            out_buf = allocate(shape=out_shape, dtype=np.float32)
            in_buf[:] = x
            dma = overlay.axi_dma_0
            dma.sendchannel.transfer(in_buf)
            dma.recvchannel.transfer(out_buf)
            dma.sendchannel.wait()
            dma.recvchannel.wait()
            y_fpga = np.array(out_buf)

        result: dict[str, Any] = {"deployed": True, "reason": "ok", "y_fpga": y_fpga}
        if golden is not None:
            max_err = float(np.max(np.abs(y_fpga - golden)))
            result["max_error"] = max_err
            result["pass"] = bool(max_err < 0.25)
            logger.info("FPGA max error vs golden: %s (pass=%s)", max_err, result["pass"])
        return result
    except Exception as exc:  # noqa: BLE001 - hardware paths are flaky; never crash
        logger.warning("On-board deployment failed: %s", exc)
        return {"deployed": False, "reason": f"runtime_error: {exc}", "build_dir": str(build_dir)}
