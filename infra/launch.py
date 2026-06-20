"""GPU placement / launch helpers for test-time finetuning.

Small and honest: detect available accelerators and report how a finetuning job
would be placed. Real multi-GPU/distributed orchestration is intentionally out of
scope for the in-process test-time loop (which runs on a single device), but this
gives scripts a single place to query device capability and to stage larger jobs.
"""

from __future__ import annotations

from typing import Any

from paths import get_logger

logger = get_logger("burnttt.infra.launch")


def device_report() -> dict[str, Any]:
    """Report torch device availability (cuda/mps/cpu) without requiring torch."""
    info: dict[str, Any] = {"torch": False, "cuda": False, "mps": False, "n_gpus": 0, "devices": []}
    try:
        import torch
    except Exception:  # noqa: BLE001
        return info
    info["torch"] = True
    if torch.cuda.is_available():
        info["cuda"] = True
        info["n_gpus"] = torch.cuda.device_count()
        info["devices"] = [torch.cuda.get_device_name(i) for i in range(info["n_gpus"])]
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        info["mps"] = True
    return info


def best_device() -> str:
    info = device_report()
    if info["cuda"]:
        return "cuda"
    if info["mps"]:
        return "mps"
    return "cpu"


def describe_placement() -> str:
    info = device_report()
    if not info["torch"]:
        return "torch not installed; test-time finetuning will use the heuristic backend."
    if info["cuda"]:
        return f"{info['n_gpus']} CUDA GPU(s): {', '.join(info['devices'])}; LoRA test-time finetuning enabled."
    if info["mps"]:
        return "Apple MPS available; small LoRA finetuning possible (slow)."
    return "CPU only; real LoRA finetuning will be very slow -- prefer the heuristic backend."
