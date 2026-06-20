"""Staged evaluation funnel: don't synthesize everything the GLM proposes.

    Stage 1 (cheap):  hls4ml convert + bit-accurate C-sim error + analytical
                      resource estimate -- run for ALL candidates.
    Stage 2 (medium): real HLS synthesis -- only the top-k by Stage-1 reward.
    Stage 3 (costly): bitstream build -- only the single best (left to script 03).

This keeps the GLM loop fast while still grounding the final pick in real
synthesis when a toolchain is available. ``evaluate_config`` is imported lazily so
this module stays importable without TensorFlow/hls4ml.
"""

from __future__ import annotations

from typing import Any, Callable

from paths import get_logger
from ttt.config_space import BurnConfig

logger = get_logger("burnttt.infra.staged_eval")


def staged_evaluate(
    configs: list[BurnConfig],
    evaluate: Callable[[BurnConfig, bool], dict[str, Any]],
    top_synth: int = 3,
) -> list[dict[str, Any]]:
    """Evaluate ``configs`` cheaply, then re-evaluate the best ``top_synth`` with synthesis.

    ``evaluate(config, run_synth)`` must return a result dict with a ``reward``
    key (e.g. a thin wrapper over :func:`ttt.evaluate_config.evaluate_config` or a
    ``SearchContext.evaluate``).
    """
    # Stage 1: cheap evaluation for everyone.
    stage1: list[dict[str, Any]] = []
    for cfg in configs:
        res = evaluate(cfg, False)
        res["stage"] = 1
        stage1.append(res)
    logger.info("Stage 1 complete: %d configs evaluated (sim/estimate).", len(stage1))

    # Stage 2: synthesis for the most promising.
    ranked = sorted(stage1, key=lambda r: float(r.get("reward", -1e9)), reverse=True)
    promoted = [r for r in ranked if r.get("compile_success")][:top_synth]
    promoted_names = {r.get("config_name") for r in promoted}

    results = list(stage1)
    for res in promoted:
        try:
            cfg = BurnConfig.from_dict(res)
        except Exception:  # noqa: BLE001
            continue
        synth_res = evaluate(cfg, True)
        synth_res["stage"] = 2
        results.append(synth_res)
    if promoted_names:
        logger.info("Stage 2 complete: synthesized top %d (%s).", len(promoted), ", ".join(map(str, promoted_names)))
    return results
