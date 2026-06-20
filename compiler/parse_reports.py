"""Tolerant parsers for hls4ml / Vivado HLS / Vitis HLS reports.

Deliberately ugly and defensive: we recursively hunt for whatever report files
exist under a build directory and extract latency / II / resource numbers from
them. Any missing file or field simply stays ``None`` instead of crashing the
search loop. (See plan.md section 7 — "this is a crowbar hour".)
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from paths import get_logger

logger = get_logger("burnttt.parse")

# Fields we try to populate from any available report.
REPORT_FIELDS = ("latency_cycles", "ii", "bram", "dsp", "ff", "lut", "timing_met")


def _empty_report() -> dict[str, Any]:
    return {f: None for f in REPORT_FIELDS} | {"report_files": []}


def _to_int(text: str | None) -> int | None:
    if text is None:
        return None
    text = text.strip()
    if not text or text.lower() in {"-", "n/a", "na"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _parse_csynth_xml(path: Path, report: dict[str, Any]) -> None:
    """Parse a Vivado/Vitis ``*_csynth.xml`` synthesis report."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        logger.debug("Could not parse XML %s: %s", path, exc)
        return

    perf = root.find(".//PerformanceEstimates/SummaryOfOverallLatency")
    if perf is not None:
        worst = perf.findtext("Worst-caseLatency")
        best = perf.findtext("Best-caseLatency")
        report["latency_cycles"] = _to_int(worst) or _to_int(best)
        ii = perf.findtext("Interval-max") or perf.findtext("Interval-min")
        report["ii"] = _to_int(ii)

    res = root.find(".//AreaEstimates/Resources")
    if res is not None:
        # Tag names differ across tool generations (DSP48E vs DSP).
        report["bram"] = _to_int(res.findtext("BRAM_18K"))
        report["dsp"] = _to_int(res.findtext("DSP48E") or res.findtext("DSP"))
        report["ff"] = _to_int(res.findtext("FF"))
        report["lut"] = _to_int(res.findtext("LUT"))


_RPT_PATTERNS = {
    "latency_cycles": re.compile(r"\|\s*\d+\s*\|\s*(\d+)\s*\|.*Latency", re.IGNORECASE),
    "bram": re.compile(r"BRAM_18K\s*\|\s*(\d+)", re.IGNORECASE),
    "dsp": re.compile(r"DSP(?:48E)?\s*\|\s*(\d+)", re.IGNORECASE),
    "ff": re.compile(r"\bFF\s*\|\s*(\d+)", re.IGNORECASE),
    "lut": re.compile(r"\bLUT\s*\|\s*(\d+)", re.IGNORECASE),
}


def _parse_csynth_rpt(path: Path, report: dict[str, Any]) -> None:
    """Best-effort regex parse of a textual ``*_csynth.rpt`` report."""
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return

    # Latency / interval block: rows of "| min | max | min | max | type |".
    lat = re.search(
        r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(?:dataflow|function|none|pipeline)?\s*\|",
        text,
        re.IGNORECASE,
    )
    if lat and report.get("latency_cycles") is None:
        report["latency_cycles"] = _to_int(lat.group(2))
        report["ii"] = _to_int(lat.group(4))

    for field in ("bram", "dsp", "ff", "lut"):
        if report.get(field) is None:
            m = _RPT_PATTERNS[field].search(text)
            if m:
                report[field] = _to_int(m.group(1))

    if "Timing met" in text or "All user specified timing constraints are met" in text:
        report["timing_met"] = True
    elif re.search(r"Timing.*(not met|violation)", text, re.IGNORECASE):
        report["timing_met"] = False


def parse_reports(build_dir: str | Path) -> dict[str, Any]:
    """Recursively parse any HLS reports under ``build_dir``.

    Returns a dict with ``latency_cycles``, ``ii``, ``bram``, ``dsp``, ``ff``,
    ``lut``, ``timing_met`` (any may be ``None``) and ``report_files`` (the
    list of files inspected).
    """
    build_dir = Path(build_dir)
    report = _empty_report()
    if not build_dir.exists():
        return report

    xml_files = sorted(build_dir.rglob("*_csynth.xml"))
    rpt_files = sorted(build_dir.rglob("*_csynth.rpt")) + sorted(build_dir.rglob("*csynth*.rpt"))

    for path in xml_files:
        report["report_files"].append(str(path))
        _parse_csynth_xml(path, report)

    for path in rpt_files:
        if str(path) not in report["report_files"]:
            report["report_files"].append(str(path))
        _parse_csynth_rpt(path, report)

    if report["report_files"]:
        logger.info(
            "Parsed %d report file(s) under %s: latency=%s ii=%s dsp=%s lut=%s",
            len(report["report_files"]),
            build_dir,
            report["latency_cycles"],
            report["ii"],
            report["dsp"],
            report["lut"],
        )
    else:
        logger.debug("No HLS report files found under %s", build_dir)
    return report
