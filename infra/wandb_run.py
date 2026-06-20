"""Optional Weights & Biases logging for BurnTTT runs."""

from __future__ import annotations

import os
from typing import Any

from paths import get_logger

logger = get_logger("burnttt.infra.wandb_run")

_run = None
_enabled: bool | None = None
_best_reward: dict[str, float] = {}


def wandb_available() -> bool:
    global _enabled
    if _enabled is not None:
        return _enabled
    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        _enabled = False
        return False
    try:
        import wandb  # noqa: F401

        _enabled = True
    except ImportError:
        _enabled = False
    return _enabled


def init_run(name: str | None = None, config: dict[str, Any] | None = None) -> Any:
    """Start a wandb run if available; return run handle or None."""
    global _run
    if not wandb_available():
        return None
    if _run is not None:
        return _run

    import wandb

    run_name = name or os.environ.get("WANDB_RUN_NAME", "burnttt")
    project = os.environ.get("WANDB_PROJECT", "burnttt")
    entity = os.environ.get("WANDB_ENTITY") or None
    _run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=config or {},
        reinit=True,
    )
    logger.info("wandb run: %s", _run.url if _run else "none")
    return _run


def log_eval(step: int, method: str, result: dict[str, Any], round_idx: int | None = None) -> None:
    if _run is None:
        return
    reward = float(result.get("reward", 0))
    _best_reward[method] = max(_best_reward.get(method, reward), reward)
    metrics = {
        "eval/step": step,
        "eval/reward": reward,
        "eval/best_reward": _best_reward[method],
        "eval/compile_success": int(bool(result.get("compile_success"))),
        "eval/max_error": result.get("max_error"),
        "eval/latency_cycles": result.get("latency_cycles"),
    }
    if round_idx is not None:
        metrics["eval/round"] = round_idx
    if result.get("cosim_pass") is not None:
        metrics["eval/cosim_pass"] = int(bool(result.get("cosim_pass")))
    if result.get("tokens_per_sec") is not None:
        metrics["eval/tokens_per_sec"] = result.get("tokens_per_sec")
    _run.log({f"{method}/{k.split('/', 1)[1]}": v for k, v in metrics.items()}, step=step)


def log_ttt_step(step: int, info: dict[str, Any]) -> None:
    if _run is None:
        return
    flat = {f"ttt/{k}": v for k, v in info.items() if isinstance(v, (int, float, bool, str))}
    _run.log(flat, step=step)


def finish() -> None:
    global _run
    if _run is not None:
        _run.finish()
        _run = None
