"""The GLM compiler agent for HLS mode: propose, compile, repair with iterative edit.

Mirrors :mod:`glm.agent` (config mode) but the artifact is a :class:`KernelBundle`
instead of a :class:`BurnConfig`. The agent:
1. Proposes new HLS kernels (from seed template + task spec).
2. Repairs kernels that fail (compile error, cosim mismatch, timing).
3. Iterates on passing kernels to improve throughput.
4. Branches repair strategy by failure mode.
"""

from __future__ import annotations

import random
from typing import Any

from glm.parsing.hls import HLSParseResult, parse_hls_from_text, validate_kernel_bundle
from glm.prompts.hls_templates import (
    HLS_SYSTEM_PROMPT,
    build_hls_iterate_prompt,
    build_hls_propose_prompt,
    build_hls_repair_prompt,
)
from glm.serving import GLMBackend, HFBackend, HeuristicBackend, load_backend
from glm.tasks import FpgaTask
from compiler.kernel_lib.swiglu_mlp import SwiGLUConfig, generate_full_bundle
from paths import get_logger
from ttt.config_space import KernelBundle

logger = get_logger("burnttt.glm.agent_hls")


def result_to_hls_history_row(result: dict[str, Any]) -> dict[str, Any]:
    """Turn an :func:`ttt.evaluate_hls.evaluate_hls` result into a history row."""
    return {
        "kernel_name": result.get("kernel_name"),
        "hls_compile_success": result.get("hls_compile_success"),
        "cosim_pass": result.get("cosim_pass"),
        "timing_met": result.get("timing_met"),
        "max_error": result.get("max_error"),
        "latency_cycles": result.get("latency_cycles"),
        "tokens_per_sec": result.get("tokens_per_sec"),
        "dsp": result.get("dsp"),
        "lut": result.get("lut"),
        "reward": result.get("reward"),
        "error_msg": result.get("error_msg"),
    }


class GLMCompilerAgent:
    """Authors and iterates on HLS kernels for an FPGA task.

    The agent wraps a GLM backend (real LLM or heuristic) and drives the
    propose → compile → repair → iterate loop for custom HLS.
    """

    def __init__(
        self,
        backend: GLMBackend | None = None,
        seed: int = 0,
        max_repair_attempts: int = 3,
    ):
        self.backend = backend or load_backend()
        self._rng = random.Random(seed)
        self.max_repair_attempts = max_repair_attempts

    @property
    def backend_name(self) -> str:
        return self.backend.name

    def propose(
        self,
        task: FpgaTask,
        history: list[dict[str, Any]],
        hidden_dim: int,
        intermediate_dim: int,
        seed_bundle: KernelBundle | None = None,
    ) -> KernelBundle:
        """Propose a new HLS kernel for the task.

        If a real LLM backend is available, generates via prompting.
        Otherwise returns a parametrically varied seed template.
        """
        if isinstance(self.backend, HFBackend):
            return self._propose_llm(task, history, hidden_dim, intermediate_dim, seed_bundle)
        return self._propose_heuristic(task, history, hidden_dim, intermediate_dim, seed_bundle)

    def repair(
        self,
        task: FpgaTask,
        bundle: KernelBundle,
        result: dict[str, Any],
    ) -> KernelBundle | None:
        """Repair a failed kernel based on error type.

        Returns a new KernelBundle or None if repair is not possible.
        """
        error_type = _classify_failure(result)
        if error_type is None:
            return None  # Not a failure

        if isinstance(self.backend, HFBackend):
            return self._repair_llm(task, bundle, result, error_type)
        return self._repair_heuristic(task, bundle, result, error_type)

    def iterate(
        self,
        task: FpgaTask,
        bundle: KernelBundle,
        current_metrics: dict[str, Any],
        best_metrics: dict[str, Any] | None = None,
    ) -> KernelBundle:
        """Iterate on a passing kernel to improve performance."""
        if isinstance(self.backend, HFBackend):
            return self._iterate_llm(task, bundle, current_metrics, best_metrics)
        return self._iterate_heuristic(task, bundle, current_metrics, best_metrics)

    def adapt(self, trajectories: list[dict[str, Any]]) -> dict[str, Any]:
        """Delegate test-time training to the backend."""
        return self.backend.adapt(trajectories)

    # -- LLM paths ----------------------------------------------------------

    def _propose_llm(self, task, history, hidden_dim, intermediate_dim, seed_bundle):
        seed_src = None
        if seed_bundle:
            seed_src = seed_bundle.sources.get("kernel_top.cpp", "")

        prompt = build_hls_propose_prompt(
            block_spec=task.describe(),
            part=task.target_part,
            clock_ns=3.3,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            seed_template=seed_src,
            history=history,
        )
        texts = self._generate(prompt)
        for text in texts:
            parsed = parse_hls_from_text(text)
            if parsed.success:
                return KernelBundle(
                    sources=parsed.sources,
                    hidden_dim=hidden_dim,
                    intermediate_dim=intermediate_dim,
                    part=task.target_part,
                )
        # Fallback to seed
        logger.warning("LLM propose failed to produce valid HLS; using seed template.")
        return self._seed_bundle(hidden_dim, intermediate_dim, task.target_part)

    def _repair_llm(self, task, bundle, result, error_type):
        current_src = bundle.sources.get("kernel_top.cpp", "")
        prompt = build_hls_repair_prompt(
            block_spec=task.describe(),
            current_source=current_src,
            error_msg=result.get("error_msg", ""),
            error_type=error_type,
        )
        texts = self._generate(prompt)
        for text in texts:
            parsed = parse_hls_from_text(text)
            if parsed.success:
                return KernelBundle(
                    sources=parsed.sources,
                    hidden_dim=bundle.hidden_dim,
                    intermediate_dim=bundle.intermediate_dim,
                    part=bundle.part,
                    weight_bits=bundle.weight_bits,
                    act_bits=bundle.act_bits,
                )
        return None

    def _iterate_llm(self, task, bundle, current_metrics, best_metrics):
        current_src = bundle.sources.get("kernel_top.cpp", "")
        prompt = build_hls_iterate_prompt(
            block_spec=task.describe(),
            current_source=current_src,
            current_metrics=current_metrics,
            best_metrics=best_metrics,
        )
        texts = self._generate(prompt)
        for text in texts:
            parsed = parse_hls_from_text(text)
            if parsed.success:
                return KernelBundle(
                    sources=parsed.sources,
                    hidden_dim=bundle.hidden_dim,
                    intermediate_dim=bundle.intermediate_dim,
                    part=bundle.part,
                )
        return bundle  # Return unchanged if parsing fails

    def _generate(self, user_prompt: str) -> list[str]:
        """Generate text from the backend."""
        backend: HFBackend = self.backend  # type: ignore[assignment]
        return backend._generate(HLS_SYSTEM_PROMPT, user_prompt, n=1)

    # -- Heuristic paths (no LLM) -------------------------------------------

    def _propose_heuristic(self, task, history, hidden_dim, intermediate_dim, seed_bundle):
        """Parametrically vary the seed template based on history."""
        if seed_bundle and not history:
            return seed_bundle

        # Vary precision and tiling based on what worked/failed
        cfg = self._pick_config_from_history(history, hidden_dim, intermediate_dim)
        sources = generate_full_bundle(cfg)
        return KernelBundle(
            sources=sources,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            part=task.target_part,
            weight_bits=cfg.weight_bits,
            weight_int_bits=cfg.weight_int_bits,
            act_bits=cfg.act_bits,
            act_int_bits=cfg.act_int_bits,
            tile_hidden=cfg.tile_hidden,
            tile_inter=cfg.tile_inter,
        )

    def _repair_heuristic(self, task, bundle, result, error_type):
        """Heuristic repair: widen precision for cosim, reduce parallelism for timing."""
        if error_type == "cosim":
            # Widen precision
            new_wbits = min(32, bundle.weight_bits + 4)
            new_abits = min(32, bundle.act_bits + 4)
            cfg = SwiGLUConfig(
                hidden_dim=bundle.hidden_dim,
                intermediate_dim=bundle.intermediate_dim,
                weight_bits=new_wbits,
                weight_int_bits=bundle.weight_int_bits + 2,
                act_bits=new_abits,
                act_int_bits=bundle.act_int_bits + 2,
                tile_hidden=bundle.tile_hidden,
                tile_inter=bundle.tile_inter,
            )
        elif error_type == "timing":
            # Reduce parallelism (increase II, reduce partitioning)
            cfg = SwiGLUConfig(
                hidden_dim=bundle.hidden_dim,
                intermediate_dim=bundle.intermediate_dim,
                weight_bits=bundle.weight_bits,
                weight_int_bits=bundle.weight_int_bits,
                act_bits=bundle.act_bits,
                act_int_bits=bundle.act_int_bits,
                tile_hidden=max(1, bundle.tile_hidden // 2),
                tile_inter=max(1, bundle.tile_inter // 2),
                pipeline_ii=2,
            )
        else:
            # Compile error: try safer defaults
            cfg = SwiGLUConfig(
                hidden_dim=bundle.hidden_dim,
                intermediate_dim=bundle.intermediate_dim,
                weight_bits=16,
                weight_int_bits=8,
                act_bits=16,
                act_int_bits=8,
                tile_hidden=4,
                tile_inter=4,
            )

        sources = generate_full_bundle(cfg)
        return KernelBundle(
            sources=sources,
            hidden_dim=bundle.hidden_dim,
            intermediate_dim=bundle.intermediate_dim,
            part=bundle.part,
            weight_bits=cfg.weight_bits,
            weight_int_bits=cfg.weight_int_bits,
            act_bits=cfg.act_bits,
            act_int_bits=cfg.act_int_bits,
            tile_hidden=cfg.tile_hidden,
            tile_inter=cfg.tile_inter,
        )

    def _iterate_heuristic(self, task, bundle, current_metrics, best_metrics):
        """Heuristic iteration: try more parallelism or tighter precision."""
        # Try increasing tile sizes for more parallelism
        new_tile_h = min(bundle.hidden_dim, bundle.tile_hidden * 2)
        new_tile_i = min(bundle.intermediate_dim, bundle.tile_inter * 2)

        # Occasionally try reducing precision if error margin allows
        max_err = current_metrics.get("max_error", 1.0)
        new_wbits = bundle.weight_bits
        new_abits = bundle.act_bits
        if max_err < 0.005 and self._rng.random() > 0.5:
            new_wbits = max(8, bundle.weight_bits - 2)
            new_abits = max(8, bundle.act_bits - 2)

        cfg = SwiGLUConfig(
            hidden_dim=bundle.hidden_dim,
            intermediate_dim=bundle.intermediate_dim,
            weight_bits=new_wbits,
            weight_int_bits=bundle.weight_int_bits,
            act_bits=new_abits,
            act_int_bits=bundle.act_int_bits,
            tile_hidden=new_tile_h,
            tile_inter=new_tile_i,
        )
        sources = generate_full_bundle(cfg)
        return KernelBundle(
            sources=sources,
            hidden_dim=bundle.hidden_dim,
            intermediate_dim=bundle.intermediate_dim,
            part=bundle.part,
            weight_bits=cfg.weight_bits,
            weight_int_bits=cfg.weight_int_bits,
            act_bits=cfg.act_bits,
            act_int_bits=cfg.act_int_bits,
            tile_hidden=cfg.tile_hidden,
            tile_inter=cfg.tile_inter,
        )

    def _pick_config_from_history(self, history, hidden_dim, intermediate_dim) -> SwiGLUConfig:
        """Pick SwiGLUConfig parameters informed by history."""
        # Start with defaults
        wbits = 16
        ibits = 6
        abits = 16
        aibits = 6
        tile_h = 8
        tile_i = 16

        if history:
            # Find best passing config
            passing = [h for h in history if h.get("cosim_pass")]
            if passing:
                best = max(passing, key=lambda h: h.get("reward", -1e9))
                # Inherit precision from best
                wbits = best.get("weight_bits", wbits) or wbits
                abits = best.get("act_bits", abits) or abits
            else:
                # All failed — try wider precision
                wbits = 20
                abits = 20
                ibits = 8
                aibits = 8

            # Vary tile sizes randomly around best
            tile_h = self._rng.choice([4, 8, 12, 16])
            tile_i = self._rng.choice([8, 16, 24, 32])

        return SwiGLUConfig(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            weight_bits=wbits,
            weight_int_bits=ibits,
            act_bits=abits,
            act_int_bits=aibits,
            tile_hidden=tile_h,
            tile_inter=tile_i,
        )

    def _seed_bundle(self, hidden_dim, intermediate_dim, part) -> KernelBundle:
        cfg = SwiGLUConfig(hidden_dim=hidden_dim, intermediate_dim=intermediate_dim)
        sources = generate_full_bundle(cfg)
        return KernelBundle(
            sources=sources,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            part=part,
        )


def _classify_failure(result: dict[str, Any]) -> str | None:
    """Classify the failure mode of an HLS result."""
    if not result.get("hls_compile_success"):
        return "compile"
    if not result.get("cosim_pass"):
        return "cosim"
    if not result.get("timing_met", True):
        return "timing"
    return None
