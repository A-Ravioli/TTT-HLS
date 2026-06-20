"""The GLM generator agent: propose + repair, with validation and safe fallbacks.

This is the policy that *replaces the random forest*. It asks a GLM backend
(:mod:`glm.serving`) to author hardware configs for a task given the feedback
seen so far, validates/de-duplicates the result, and guarantees it always returns
enough usable configs (topping up with random exploration if the model under-
delivers). The repair path turns a compile error into a corrected config -- the
agentic compiler loop.
"""

from __future__ import annotations

import random
from typing import Any

from glm.serving import GLMBackend, load_backend
from glm.tasks import FpgaTask
from paths import get_logger
from ttt.config_space import BurnConfig, sample_random_configs

logger = get_logger("burnttt.glm.agent")


def result_to_history_row(result: dict[str, Any]) -> dict[str, Any]:
    """Turn an :func:`ttt.evaluate_config.evaluate_config` result into a history row.

    Carries a parsed ``_config_obj`` so the heuristic backend can measure config
    similarity without re-parsing.
    """
    try:
        cfg_obj = BurnConfig.from_dict(result)
    except Exception:  # noqa: BLE001
        cfg_obj = None
    return {
        "config": {
            "weight_bits": result.get("weight_bits"),
            "activation_bits": result.get("activation_bits"),
            "int_bits": result.get("int_bits"),
            "reuse_dense_1": result.get("reuse_dense_1"),
            "reuse_dense_2": result.get("reuse_dense_2"),
            "strategy": result.get("strategy"),
        },
        "compile_success": result.get("compile_success"),
        "max_error": result.get("max_error"),
        "latency_cycles": result.get("latency_cycles"),
        "dsp": result.get("dsp"),
        "lut": result.get("lut"),
        "fits_board": result.get("fits_board"),
        "reward": result.get("reward"),
        "error_msg": result.get("error_msg"),
        "_config_obj": cfg_obj,
    }


class GLMGenerator:
    """Authors configs for a task, wrapping a (real or heuristic) GLM backend."""

    def __init__(self, backend: GLMBackend | None = None, seed: int = 0):
        self.backend = backend or load_backend()
        self._rng = random.Random(seed)

    @property
    def backend_name(self) -> str:
        return self.backend.name

    def propose(
        self,
        task: FpgaTask,
        history: list[dict[str, Any]],
        n: int = 3,
        exclude: set[str] | None = None,
    ) -> list[BurnConfig]:
        exclude = set(exclude or set())
        task_desc = task.describe()
        try:
            proposals = self.backend.propose_configs(task_desc, history, n, exclude, self._rng)
        except Exception as exc:  # noqa: BLE001 - never let the LLM crash the search
            logger.warning("Backend propose failed (%s); using random fallback.", exc)
            proposals = []

        picks: list[BurnConfig] = []
        seen = set(exclude)
        for cfg in proposals:
            if cfg.short_name() not in seen:
                seen.add(cfg.short_name())
                picks.append(cfg)

        # Top up with exploration if the model under-delivered.
        guard = 0
        while len(picks) < n and guard < 200:
            guard += 1
            cfg = sample_random_configs(1, self._rng)[0]
            if cfg.short_name() not in seen:
                seen.add(cfg.short_name())
                picks.append(cfg)
        return picks[:n]

    def repair(
        self,
        task: FpgaTask,
        failed_config: BurnConfig,
        error_msg: str,
    ) -> BurnConfig | None:
        try:
            fixed = self.backend.repair_config(
                task.describe(), failed_config.to_dict(), error_msg or "", self._rng
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Backend repair failed (%s); no fix produced.", exc)
            return None
        if fixed is not None and fixed.short_name() != failed_config.short_name():
            return fixed
        return None

    def adapt(self, trajectories: list[dict[str, Any]]) -> dict[str, Any]:
        """Delegate test-time training to the backend."""
        return self.backend.adapt(trajectories)
