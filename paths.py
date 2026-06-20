"""Shared paths, logging, and target-part helpers for BurnTTT.

Keeping this tiny module at the repo root lets every package (``models``,
``compiler``, ``ttt``) and every script agree on where artifacts live without a
heavyweight config system.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
BUILD_DIR = REPO_ROOT / "build"
RESULTS_DIR = REPO_ROOT / "results"

# Canonical artifact locations (see plan.md section 1).
MODEL_PATH = ARTIFACTS_DIR / "tiny_ffn.keras"
TEST_INPUTS_PATH = ARTIFACTS_DIR / "test_inputs.npy"
GOLDEN_OUTPUTS_PATH = ARTIFACTS_DIR / "golden_outputs.npy"

# Search history consumed by the dashboard.
RUNS_CSV = RESULTS_DIR / "runs.csv"

# Default FPGA part: PYNQ-Z2 (Zynq-7020). Override with BURN_TARGET_PART.
DEFAULT_TARGET_PART = "xc7z020clg400-1"


def get_target_part() -> str:
    """Return the FPGA part, honoring the ``BURN_TARGET_PART`` env var."""
    return os.environ.get("BURN_TARGET_PART", DEFAULT_TARGET_PART).strip() or DEFAULT_TARGET_PART


def ensure_dirs() -> None:
    """Create the standard output directories if they do not yet exist."""
    for d in (ARTIFACTS_DIR, BUILD_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


_LOGGERS_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger with a consistent, human-readable format."""
    global _LOGGERS_CONFIGURED
    if not _LOGGERS_CONFIGURED:
        logging.basicConfig(
            level=os.environ.get("BURN_LOG_LEVEL", "INFO"),
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        _LOGGERS_CONFIGURED = True
    return logging.getLogger(name)
