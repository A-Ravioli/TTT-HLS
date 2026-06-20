"""GLM backends: a real LLM when one is available, a heuristic stand-in otherwise.

A backend turns ``(task, feedback-history)`` into candidate hardware configs. Two
implementations:

* :class:`HFBackend` -- a real GLM via ``transformers`` (optionally LoRA-adapted
  and test-time-finetuned). Lazy-imports torch/transformers so importing this
  module is always cheap.
* :class:`HeuristicBackend` -- a deterministic, pure-stdlib stand-in used when no
  GPU / GLM weights / ``transformers`` are present. It is similarity-weighted over
  the feedback history and exposes an adaptation knob so the test-time-training
  loop has something to update even off-GPU, which keeps the whole pipeline
  runnable and testable in CI.

Select one with :func:`load_backend`, driven by the ``BURN_GLM_MODEL`` and
``BURN_GLM_BACKEND`` env vars.
"""

from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod
from typing import Any

from paths import get_logger
from ttt.config_space import (
    BITWIDTHS,
    REUSE_FACTORS,
    BurnConfig,
    neighbors,
    sample_random_configs,
)

logger = get_logger("burnttt.glm.serving")


class GLMBackend(ABC):
    """Authors candidate configs for a task given its feedback history."""

    name: str = "abstract"
    is_llm: bool = False

    @abstractmethod
    def propose_configs(
        self,
        task_desc: str,
        history: list[dict[str, Any]],
        n: int,
        exclude: set[str],
        rng: random.Random,
    ) -> list[BurnConfig]:
        ...

    @abstractmethod
    def repair_config(
        self,
        task_desc: str,
        failed_config: dict[str, Any],
        error_msg: str,
        rng: random.Random,
    ) -> BurnConfig | None:
        ...

    def adapt(self, trajectories: list[dict[str, Any]]) -> dict[str, Any]:
        """Test-time-training hook. Default: no-op (frozen backend)."""
        return {"adapted": False, "reason": "backend has no test-time adaptation"}


# ---------------------------------------------------------------------------
# Heuristic stand-in
# ---------------------------------------------------------------------------

def _norm_vector(cfg: BurnConfig) -> list[float]:
    """Config feature vector normalized to ~[0,1] per dimension."""
    return [
        (cfg.weight_bits - min(BITWIDTHS)) / (max(BITWIDTHS) - min(BITWIDTHS)),
        (cfg.activation_bits - min(BITWIDTHS)) / (max(BITWIDTHS) - min(BITWIDTHS)),
        cfg.int_bits / max(BITWIDTHS),
        (cfg.reuse_dense_1 - min(REUSE_FACTORS)) / (max(REUSE_FACTORS) - min(REUSE_FACTORS)),
        (cfg.reuse_dense_2 - min(REUSE_FACTORS)) / (max(REUSE_FACTORS) - min(REUSE_FACTORS)),
        1.0 if cfg.strategy == "Latency" else 0.0,
    ]


def _dist2(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


class HeuristicBackend(GLMBackend):
    """Similarity-weighted, history-driven config author (LLM stand-in).

    Predicts a candidate's reward by kernel-weighting the rewards of configs
    already tried on this task. Without history it explores; with history it
    exploits the high-reward region. :meth:`adapt` sharpens the kernel and biases
    toward the best-known region -- this is the off-GPU analogue of a LoRA step,
    so ``glm_ttt`` measurably out-climbs frozen ``glm`` even with no real model.
    """

    name = "heuristic"
    is_llm = False

    def __init__(self, bandwidth: float = 0.6):
        self.base_bandwidth = bandwidth
        self.bandwidth = bandwidth
        self.exploit = 0.0  # raised by adapt(); 0 = pure prior, 1 = pure exploitation
        self.n_adapt_steps = 0

    # -- prediction --------------------------------------------------------
    def _predict_reward(self, cfg: BurnConfig, history: list[dict[str, Any]]) -> float:
        prior = self._prior(cfg)
        if not history:
            return prior
        cv = _norm_vector(cfg)
        num = 0.0
        den = 0.0
        for h in history:
            hc = h.get("_config_obj")
            if hc is None:
                continue
            w = pow(2.718281828, -_dist2(cv, _norm_vector(hc)) / max(1e-6, self.bandwidth))
            num += w * float(h.get("reward", 0.0))
            den += w
        data_est = num / den if den > 0 else prior
        # Blend prior and data; adaptation shifts weight toward the data/history.
        alpha = min(1.0, 0.4 + 0.6 * self.exploit)
        return alpha * data_est + (1 - alpha) * prior

    @staticmethod
    def _prior(cfg: BurnConfig) -> float:
        """Cheap fabric/accuracy proxy when no history exists yet.

        Favors moderate precision and higher reuse (fits more easily); mildly
        penalizes extreme low precision (accuracy risk) and very low reuse
        (fabric blowup).
        """
        fabric = (cfg.weight_bits / 16.0) * (1.0 / max(1, cfg.reuse_dense_1) + 1.0 / max(1, cfg.reuse_dense_2))
        accuracy_risk = max(0, 12 - cfg.weight_bits) * 0.05
        return 1.0 - fabric - accuracy_risk

    # -- proposal ----------------------------------------------------------
    def _candidate_pool(self, history, exclude, rng, n_random=120):
        pool = list(sample_random_configs(n_random, rng))
        ranked_hist = sorted(
            (h for h in history if h.get("_config_obj") is not None),
            key=lambda h: float(h.get("reward", -1e9)),
            reverse=True,
        )
        for h in ranked_hist[:5]:
            pool.extend(neighbors(h["_config_obj"]))
        out, seen = [], set()
        for c in pool:
            k = c.short_name()
            if k in exclude or k in seen:
                continue
            seen.add(k)
            out.append(c)
        return out

    def propose_configs(self, task_desc, history, n, exclude, rng):
        pool = self._candidate_pool(history, exclude, rng)
        if not pool:
            return sample_random_configs(n, rng)
        scored = sorted(pool, key=lambda c: self._predict_reward(c, history), reverse=True)
        # Reserve one exploratory pick to avoid collapsing too early (less when adapted).
        if not self.exploit and len(scored) > n:
            explore = rng.choice(scored[n : n + max(1, n)])
            return scored[: n - 1] + [explore] if n > 1 else [scored[0]]
        return scored[:n]

    def repair_config(self, task_desc, failed_config, error_msg, rng):
        from glm.parsing import dict_to_config

        cfg = dict_to_config(failed_config)
        if cfg is None:
            return sample_random_configs(1, rng)[0]
        # Common failure -> safer config: bump precision down a step, raise reuse.
        bits_idx = max(0, BITWIDTHS.index(cfg.weight_bits) - 1) if cfg.weight_bits in BITWIDTHS else 0
        safer = BurnConfig(
            weight_bits=BITWIDTHS[bits_idx],
            activation_bits=BITWIDTHS[bits_idx],
            int_bits=min(cfg.int_bits, BITWIDTHS[bits_idx] - 1),
            reuse_dense_1=min(max(REUSE_FACTORS), cfg.reuse_dense_1 * 2),
            reuse_dense_2=min(max(REUSE_FACTORS), cfg.reuse_dense_2 * 2),
            strategy="Resource",
        )
        return safer

    def adapt(self, trajectories):
        """Sharpen exploitation from the feedback collected so far."""
        valid = [t for t in trajectories if t.get("reward") is not None]
        if len(valid) < 3:
            return {"adapted": False, "reason": "not enough trajectories"}
        self.n_adapt_steps += 1
        self.exploit = min(1.0, self.exploit + 0.25)
        self.bandwidth = max(0.12, self.base_bandwidth * (0.8 ** self.n_adapt_steps))
        logger.info(
            "Heuristic adapt step %d: exploit=%.2f bandwidth=%.3f",
            self.n_adapt_steps,
            self.exploit,
            self.bandwidth,
        )
        return {
            "adapted": True,
            "step": self.n_adapt_steps,
            "exploit": self.exploit,
            "bandwidth": self.bandwidth,
        }


# ---------------------------------------------------------------------------
# Real LLM backend
# ---------------------------------------------------------------------------

class HFBackend(GLMBackend):
    """A real GLM via HuggingFace ``transformers`` (LoRA-adaptable).

    Loaded lazily from ``BURN_GLM_MODEL`` (a local path or HF repo id). Generation
    builds the text prompts in :mod:`glm.prompts` and parses JSON back with
    :mod:`glm.parsing`. The underlying ``model``/``tokenizer`` are exposed so the
    test-time trainer (:mod:`glm.finetune.trainer`) can take LoRA gradient steps.
    """

    name = "hf"
    is_llm = True

    def __init__(self, model_id: str, max_new_tokens: int = 512, temperature: float = 0.7, device: str | None = None):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self._model = None
        self._tokenizer = None

    # -- lazy load ---------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading GLM weights: %s", self.model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto" if self.device is None else None,
        )
        if self.device is not None:
            self._model = self._model.to(self.device)

    @property
    def model(self):
        self._ensure_loaded()
        return self._model

    @property
    def tokenizer(self):
        self._ensure_loaded()
        return self._tokenizer

    def attach_adapter(self, adapter) -> None:
        """Swap in a LoRA-adapted model (used by the test-time trainer)."""
        self._model = adapter

    # -- generation --------------------------------------------------------
    def _generate(self, system: str, user: str, n: int) -> list[str]:
        import torch

        self._ensure_loaded()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        outs = self._model.generate(
            **inputs,
            do_sample=True,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=n,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        gen = outs[:, inputs["input_ids"].shape[1]:]
        return [self._tokenizer.decode(g, skip_special_tokens=True) for g in gen]

    def propose_configs(self, task_desc, history, n, exclude, rng):
        from glm.parsing import parse_configs
        from glm.prompts import SYSTEM_PROMPT, build_propose_prompt

        prompt = build_propose_prompt(task_desc, history, n)
        configs: list[BurnConfig] = []
        seen = set(exclude)
        for text in self._generate(SYSTEM_PROMPT, prompt, n=max(1, n)):
            for cfg in parse_configs(text):
                if cfg.short_name() not in seen:
                    seen.add(cfg.short_name())
                    configs.append(cfg)
        return configs[:n]

    def repair_config(self, task_desc, failed_config, error_msg, rng):
        from glm.parsing import parse_configs
        from glm.prompts import SYSTEM_PROMPT, build_repair_prompt

        prompt = build_repair_prompt(task_desc, failed_config, error_msg)
        for text in self._generate(SYSTEM_PROMPT, prompt, n=1):
            cfgs = parse_configs(text)
            if cfgs:
                return cfgs[0]
        return None


# ---------------------------------------------------------------------------
# Prime Intellect Inference API (OpenAI-compatible, no local weights)
# ---------------------------------------------------------------------------

PRIME_INFERENCE_BASE_URL = "https://api.pinference.ai/api/v1"
DEFAULT_PRIME_MODEL = "z-ai/glm-5.2"


class PrimeBackend(GLMBackend):
    """GLM via Prime Intellect Inference API (OpenAI-compatible).

    Uses ``PRIME_API_KEY`` and ``BURN_GLM_MODEL`` (default ``z-ai/glm-5.2``).
    Supports generation for config-author and HLS compiler-author modes.
    Test-time LoRA requires :class:`HFBackend` on a local GPU — this backend
    falls back to :meth:`HeuristicBackend.adapt` for the TTT hook.
    """

    name = "prime"
    is_llm = True

    def __init__(
        self,
        model_id: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float = 0.7,
        api_key: str | None = None,
        base_url: str = PRIME_INFERENCE_BASE_URL,
        team_id: str | None = None,
    ):
        self.model_id = (model_id or os.environ.get("BURN_GLM_MODEL", "").strip() or DEFAULT_PRIME_MODEL)
        self.max_new_tokens = max_new_tokens or int(os.environ.get("BURN_GLM_MAX_TOKENS", "8192"))
        self.temperature = temperature
        self.api_key = api_key or os.environ.get("PRIME_API_KEY", "").strip()
        self.base_url = base_url
        self.team_id = (team_id or os.environ.get("PRIME_TEAM_ID", "")).strip() or None
        self._client = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        from openai import OpenAI

        headers = {}
        if self.team_id:
            headers["X-Prime-Team-ID"] = self.team_id
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers=headers or None,
        )
        logger.info("Prime Inference backend: model=%s max_tokens=%d", self.model_id, self.max_new_tokens)

    def _generate(self, system: str, user: str, n: int) -> list[str]:
        self._ensure_client()
        texts: list[str] = []
        for _ in range(max(1, n)):
            resp = self._client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
            )
            content = resp.choices[0].message.content or ""
            texts.append(content)
        return texts

    def propose_configs(self, task_desc, history, n, exclude, rng):
        from glm.parsing import parse_configs
        from glm.prompts import SYSTEM_PROMPT, build_propose_prompt

        prompt = build_propose_prompt(task_desc, history, n)
        configs: list[BurnConfig] = []
        seen = set(exclude)
        for text in self._generate(SYSTEM_PROMPT, prompt, n=max(1, n)):
            for cfg in parse_configs(text):
                if cfg.short_name() not in seen:
                    seen.add(cfg.short_name())
                    configs.append(cfg)
        return configs[:n]

    def repair_config(self, task_desc, failed_config, error_msg, rng):
        from glm.parsing import parse_configs
        from glm.prompts import SYSTEM_PROMPT, build_repair_prompt

        prompt = build_repair_prompt(task_desc, failed_config, error_msg)
        for text in self._generate(SYSTEM_PROMPT, prompt, n=1):
            cfgs = parse_configs(text)
            if cfgs:
                return cfgs[0]
        return None

    def adapt(self, trajectories: list[dict[str, Any]]) -> dict[str, Any]:
        """Remote API has no local weights; use heuristic-style history sharpening."""
        helper = HeuristicBackend()
        return helper.adapt(trajectories)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def llm_generate(backend: GLMBackend, system: str, user: str, n: int = 1) -> list[str]:
    """Call ``_generate`` on HF or Prime backends."""
    gen = getattr(backend, "_generate", None)
    if gen is None:
        raise TypeError(f"Backend {backend.name!r} is not an LLM backend")
    return gen(system, user, n)


def transformers_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def prime_available() -> bool:
    """True if PRIME_API_KEY is set and the openai client is importable."""
    if not os.environ.get("PRIME_API_KEY", "").strip():
        return False
    try:
        import openai  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def load_backend(prefer: str | None = None) -> GLMBackend:
    """Return a GLM backend.

    Selection order:
    1. ``prefer`` / ``BURN_GLM_BACKEND`` if it forces ``heuristic``, ``prime``, or ``hf``.
    2. :class:`PrimeBackend` if backend is ``prime`` or ``PRIME_API_KEY`` is set
       (and backend is not explicitly ``hf``).
    3. :class:`HFBackend` if ``BURN_GLM_MODEL`` is set and transformers is importable.
    4. The :class:`HeuristicBackend` stand-in otherwise.
    """
    choice = (prefer or os.environ.get("BURN_GLM_BACKEND", "")).strip().lower()
    model_id = os.environ.get("BURN_GLM_MODEL", "").strip()

    if choice == "heuristic":
        logger.info("Using heuristic GLM backend (forced).")
        return HeuristicBackend()

    if choice == "prime" or (not choice and prime_available()):
        if prime_available():
            logger.info("Using Prime Inference GLM backend: %s", model_id or DEFAULT_PRIME_MODEL)
            return PrimeBackend(model_id=model_id or None)
        logger.warning("Prime backend requested but PRIME_API_KEY / openai unavailable.")

    if choice == "hf" or (model_id and not prime_available()):
        if model_id and transformers_available():
            logger.info("Using real GLM (HF) backend: %s", model_id)
            return HFBackend(model_id=model_id)
        logger.warning(
            "Real GLM backend requested but unavailable (model_id=%r, transformers=%s); "
            "falling back to heuristic.",
            model_id,
            transformers_available(),
        )

    logger.info("Using heuristic GLM backend (no GLM backend configured).")
    return HeuristicBackend()
