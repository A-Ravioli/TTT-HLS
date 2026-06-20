"""Vitis HLS project scaffold: emit project directory, run_hls.tcl, and testbench.

Given a :class:`~ttt.config_space.KernelBundle` (HLS sources + metadata), this
module writes out a complete Vitis HLS project directory ready for
``vitis_hls -f run_hls.tcl``.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paths import BUILD_DIR, get_logger

logger = get_logger("burnttt.compiler.hls_build")

# Default clock period (ns) targeting ~300 MHz on Alveo U250.
DEFAULT_CLOCK_NS = "3.3"
DEFAULT_PART = "xcu250-figd2104-2l-e"


@dataclass
class HLSBuildResult:
    """Result from a Vitis HLS C-synthesis or cosim run."""

    success: bool
    project_dir: Path
    log: str = ""
    latency_cycles: int | None = None
    ii: int | None = None
    dsp: int | None = None
    lut: int | None = None
    ff: int | None = None
    bram: int | None = None
    fmax_mhz: float | None = None
    error_msg: str = ""
    cosim_pass: bool | None = None
    max_error: float | None = None


def write_hls_project(
    kernel_sources: dict[str, str],
    testbench_source: str,
    project_dir: Path,
    top_function: str = "kernel_top",
    part: str = DEFAULT_PART,
    clock_ns: str = DEFAULT_CLOCK_NS,
) -> Path:
    """Write a complete Vitis HLS project directory.

    Args:
        kernel_sources: mapping of filename -> C++ source content
        testbench_source: C++ testbench source (for cosim)
        project_dir: where to write the project
        top_function: HLS top-level function name
        part: FPGA part string
        clock_ns: target clock period in nanoseconds

    Returns:
        Path to the generated ``run_hls.tcl`` script.
    """
    project_dir.mkdir(parents=True, exist_ok=True)
    src_dir = project_dir / "src"
    tb_dir = project_dir / "tb"
    src_dir.mkdir(exist_ok=True)
    tb_dir.mkdir(exist_ok=True)

    # Write kernel sources
    src_files = []
    for filename, content in kernel_sources.items():
        fpath = src_dir / filename
        fpath.write_text(content)
        src_files.append(f"src/{filename}")

    # Write testbench
    tb_path = tb_dir / "tb_main.cpp"
    tb_path.write_text(testbench_source)

    # Write TCL script
    tcl = _generate_tcl(
        top_function=top_function,
        src_files=src_files,
        tb_files=["tb/tb_main.cpp"],
        part=part,
        clock_ns=clock_ns,
        project_name="hls_project",
    )
    tcl_path = project_dir / "run_hls.tcl"
    tcl_path.write_text(tcl)

    logger.info("HLS project written to %s (%d source files)", project_dir, len(src_files))
    return tcl_path


def run_csynth(project_dir: Path, timeout: int = 600) -> HLSBuildResult:
    """Run Vitis HLS C-synthesis on the project. Returns parsed results.

    Requires ``vitis_hls`` on PATH. Returns a failed result gracefully if unavailable.
    """
    tcl_path = project_dir / "run_hls.tcl"
    if not tcl_path.exists():
        return HLSBuildResult(success=False, project_dir=project_dir, error_msg="No run_hls.tcl found")

    if not shutil.which("vitis_hls"):
        logger.warning("vitis_hls not on PATH; returning analytical stub.")
        return HLSBuildResult(
            success=False, project_dir=project_dir,
            error_msg="vitis_hls not available (CI mode)",
        )

    try:
        proc = subprocess.run(
            ["vitis_hls", "-f", "run_hls.tcl"],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        log = proc.stdout + "\n" + proc.stderr
        success = proc.returncode == 0
        result = HLSBuildResult(success=success, project_dir=project_dir, log=log)
        if not success:
            result.error_msg = _extract_error(log)
        else:
            _parse_synth_report(result)
        return result
    except subprocess.TimeoutExpired:
        return HLSBuildResult(
            success=False, project_dir=project_dir,
            error_msg=f"C-synthesis timed out after {timeout}s",
        )
    except Exception as exc:  # noqa: BLE001
        return HLSBuildResult(
            success=False, project_dir=project_dir,
            error_msg=f"C-synthesis exception: {exc}",
        )


def run_cosim(project_dir: Path, timeout: int = 900) -> HLSBuildResult:
    """Run C/RTL co-simulation. Requires prior successful C-synthesis."""
    if not shutil.which("vitis_hls"):
        return HLSBuildResult(
            success=False, project_dir=project_dir,
            error_msg="vitis_hls not available for cosim",
        )

    cosim_tcl = project_dir / "run_cosim.tcl"
    cosim_tcl.write_text(
        "open_project hls_project\n"
        "open_solution solution1\n"
        "cosim_design\n"
        "exit\n"
    )
    try:
        proc = subprocess.run(
            ["vitis_hls", "-f", "run_cosim.tcl"],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        log = proc.stdout + "\n" + proc.stderr
        success = proc.returncode == 0 and "PASS" in log.upper()
        result = HLSBuildResult(
            success=success,
            project_dir=project_dir,
            log=log,
            cosim_pass=success,
        )
        if not success:
            result.error_msg = _extract_error(log)
        return result
    except subprocess.TimeoutExpired:
        return HLSBuildResult(
            success=False, project_dir=project_dir,
            error_msg=f"Cosim timed out after {timeout}s",
        )
    except Exception as exc:  # noqa: BLE001
        return HLSBuildResult(
            success=False, project_dir=project_dir,
            error_msg=f"Cosim exception: {exc}",
        )


def vitis_hls_available() -> bool:
    """Check if Vitis HLS is on PATH."""
    return shutil.which("vitis_hls") is not None


def _generate_tcl(
    top_function: str,
    src_files: list[str],
    tb_files: list[str],
    part: str,
    clock_ns: str,
    project_name: str,
) -> str:
    src_adds = "\n".join(f"add_files {f}" for f in src_files)
    tb_adds = "\n".join(f"add_files -tb {f}" for f in tb_files)
    return f"""\
open_project {project_name}
set_top {top_function}
{src_adds}
{tb_adds}
open_solution "solution1"
set_part {{{part}}}
create_clock -period {clock_ns} -name default
csynth_design
exit
"""


def _extract_error(log: str) -> str:
    """Extract the most relevant error lines from HLS log."""
    lines = log.splitlines()
    error_lines = [l for l in lines if "error" in l.lower() or "ERROR" in l]
    if error_lines:
        return "\n".join(error_lines[:10])
    return log[-500:] if len(log) > 500 else log


def _parse_synth_report(result: HLSBuildResult) -> None:
    """Parse synthesis report XML/log for resource and latency numbers."""
    # Look for the synthesis report in the project directory
    report_dir = result.project_dir / "hls_project" / "solution1" / "syn" / "report"
    if not report_dir.exists():
        return
    # Try to find the summary XML
    xmls = list(report_dir.glob("*.xml"))
    if not xmls:
        return
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(xmls[0])
        root = tree.getroot()
        # Parse latency
        perf = root.find(".//PerformanceEstimates/SummaryOfOverallLatency")
        if perf is not None:
            lat = perf.findtext("Best-caseLatency") or perf.findtext("Worst-caseLatency")
            if lat and lat.isdigit():
                result.latency_cycles = int(lat)
            ii_elem = perf.findtext("Interval-min") or perf.findtext("Best-caseLatency")
            if ii_elem and ii_elem.isdigit():
                result.ii = int(ii_elem)
        # Parse resources
        res = root.find(".//AreaEstimates/Resources")
        if res is not None:
            for tag, attr in [("DSP", "dsp"), ("LUT", "lut"), ("FF", "ff"), ("BRAM_18K", "bram")]:
                val = res.findtext(tag)
                if val and val.isdigit():
                    setattr(result, attr, int(val))
    except Exception:  # noqa: BLE001
        pass
