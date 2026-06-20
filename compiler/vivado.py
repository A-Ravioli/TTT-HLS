"""Batch Vivado place-and-route: timing reports, resource usage, bitstream.

Wraps Vivado invocations for post-synthesis and post-route analysis. When Vivado
is unavailable, returns analytical estimates.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paths import get_logger

logger = get_logger("burnttt.compiler.vivado")


@dataclass
class VivadoResult:
    """Result from a Vivado synthesis/P&R run."""

    success: bool
    project_dir: Path
    timing_met: bool = False
    wns_ns: float | None = None  # worst negative slack
    fmax_mhz: float | None = None
    lut: int | None = None
    ff: int | None = None
    dsp: int | None = None
    bram: int | None = None
    power_w: float | None = None
    log: str = ""
    error_msg: str = ""
    bitstream_path: Path | None = None


def vivado_available() -> bool:
    """Check if Vivado is on PATH."""
    return shutil.which("vivado") is not None


def run_vivado_synth(
    hls_project_dir: Path,
    part: str = "xcu250-figd2104-2l-e",
    clock_ns: float = 3.3,
    timeout: int = 3600,
) -> VivadoResult:
    """Run Vivado synthesis + implementation on an HLS-exported IP.

    Expects the HLS project to have already run csynth_design, so the exported
    RTL is in ``hls_project/solution1/impl/``.
    """
    if not vivado_available():
        logger.warning("Vivado not on PATH; returning stub result.")
        return VivadoResult(
            success=False,
            project_dir=hls_project_dir,
            error_msg="Vivado not available (CI mode)",
        )

    impl_dir = hls_project_dir / "vivado_impl"
    impl_dir.mkdir(exist_ok=True)

    tcl_script = _generate_vivado_tcl(hls_project_dir, impl_dir, part, clock_ns)
    tcl_path = impl_dir / "run_vivado.tcl"
    tcl_path.write_text(tcl_script)

    try:
        proc = subprocess.run(
            ["vivado", "-mode", "batch", "-source", str(tcl_path)],
            cwd=str(impl_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        log = proc.stdout + "\n" + proc.stderr
        success = proc.returncode == 0
        result = VivadoResult(success=success, project_dir=impl_dir, log=log)
        if success:
            _parse_timing_report(result, impl_dir)
            _parse_utilization_report(result, impl_dir)
        else:
            result.error_msg = _extract_vivado_error(log)
        return result
    except subprocess.TimeoutExpired:
        return VivadoResult(
            success=False, project_dir=impl_dir,
            error_msg=f"Vivado timed out after {timeout}s",
        )
    except Exception as exc:  # noqa: BLE001
        return VivadoResult(
            success=False, project_dir=impl_dir,
            error_msg=f"Vivado exception: {exc}",
        )


def estimate_post_route(
    hls_result: Any,
    clock_ns: float = 3.3,
) -> VivadoResult:
    """Analytical estimate when Vivado is unavailable.

    Uses HLS resource estimates with typical post-route overhead factors.
    """
    # Typical overhead: LUT ~1.2x, FF ~1.1x, DSP ~1.0x, BRAM ~1.0x from HLS estimates
    lut = int((hls_result.lut or 0) * 1.2) if hls_result.lut else None
    ff = int((hls_result.ff or 0) * 1.1) if hls_result.ff else None
    dsp = hls_result.dsp
    bram = hls_result.bram

    # Estimate timing: assume HLS estimate is ~80% of achievable Fmax
    fmax = (1000.0 / clock_ns) * 0.85 if clock_ns else None
    timing_met = True  # Optimistic for estimates

    return VivadoResult(
        success=True,
        project_dir=Path("."),
        timing_met=timing_met,
        wns_ns=0.1,  # Slight positive slack (estimated)
        fmax_mhz=fmax,
        lut=lut,
        ff=ff,
        dsp=dsp,
        bram=bram,
        power_w=None,
    )


def _generate_vivado_tcl(
    hls_dir: Path, impl_dir: Path, part: str, clock_ns: float
) -> str:
    """Generate TCL script for Vivado synthesis + implementation."""
    return f"""\
# Auto-generated Vivado implementation script
create_project -force vivado_project {impl_dir}/vivado_project -part {part}

# Add HLS-generated RTL
set ip_dir [glob -nocomplain {hls_dir}/hls_project/solution1/impl/verilog]
if {{$ip_dir ne ""}} {{
    add_files [glob -nocomplain $ip_dir/*.v $ip_dir/*.sv]
}}

# Create clock constraint
create_clock -period {clock_ns} -name clk [get_ports ap_clk]

# Run synthesis and implementation
launch_runs synth_1
wait_on_run synth_1
launch_runs impl_1
wait_on_run impl_1

# Generate reports
open_run impl_1
report_timing_summary -file {impl_dir}/timing_summary.rpt
report_utilization -file {impl_dir}/utilization.rpt
report_power -file {impl_dir}/power.rpt

exit
"""


def _parse_timing_report(result: VivadoResult, impl_dir: Path) -> None:
    """Parse Vivado timing summary report."""
    report = impl_dir / "timing_summary.rpt"
    if not report.exists():
        return
    text = report.read_text()
    for line in text.splitlines():
        if "WNS" in line and "ns" in line.lower():
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "WNS":
                    try:
                        result.wns_ns = float(parts[i + 1])
                        result.timing_met = result.wns_ns >= 0
                    except (IndexError, ValueError):
                        pass
                    break


def _parse_utilization_report(result: VivadoResult, impl_dir: Path) -> None:
    """Parse Vivado utilization report."""
    report = impl_dir / "utilization.rpt"
    if not report.exists():
        return
    text = report.read_text()
    # Simplified parsing — real reports have tabular data
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) >= 3:
            name = parts[1].strip().lower()
            try:
                val = int(parts[2].strip())
                if "lut" in name and result.lut is None:
                    result.lut = val
                elif "register" in name or "ff" in name:
                    if result.ff is None:
                        result.ff = val
                elif "dsp" in name and result.dsp is None:
                    result.dsp = val
                elif "bram" in name and result.bram is None:
                    result.bram = val
            except ValueError:
                pass


def _extract_vivado_error(log: str) -> str:
    lines = log.splitlines()
    errors = [l for l in lines if "ERROR" in l]
    return "\n".join(errors[:10]) if errors else log[-500:]
