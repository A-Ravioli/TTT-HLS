"""Test-time trainer: update the GLM generator on this task's own feedback.

Within a single run, after each batch of evaluations, :meth:`TestTimeTrainer.step`
adapts the generator so it proposes better configs next round:

* Real LLM backend + ``peft``/``torch``: take LoRA SFT gradient steps on the
  high-reward ``(prompt -> config)`` examples mined from the trajectories.
* Heuristic backend (off-GPU): sharpen the history kernel via
  :meth:`~glm.serving.HeuristicBackend.adapt`.

Either way the interface is identical, so the search loop and dashboard don't care
which one ran.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

from glm.agent import GLMGenerator
from glm.finetune import lora as lora_mod
from glm.finetune.dataset import to_preference_pairs, to_repair_preference_pairs, to_sft_examples
from glm.serving import HFBackend
from glm.tasks import FpgaTask
from paths import REPO_ROOT, get_logger

logger = get_logger("burnttt.glm.trainer")

ADAPTERS_DIR = REPO_ROOT / "data" / "adapters"


class TestTimeTrainer:
    """Adapt a :class:`~glm.agent.GLMGenerator` to a task at test time."""

    def __init__(
        self,
        generator: GLMGenerator,
        task: FpgaTask,
        lr: float = 1e-4,
        steps_per_round: int = 4,
        max_seq_len: int = 1024,
        use_dpo: bool = True,
        dpo_weight: float = 0.5,
        dpo_beta: float = 0.1,
        anchor_regression_threshold: float = -50.0,
        run_name: str = "glm_ttt",
        evaluate_fn: Callable[[Any], dict[str, Any]] | None = None,
    ):
        self.generator = generator
        self.task = task
        self.lr = lr
        self.steps_per_round = steps_per_round
        self.max_seq_len = max_seq_len
        self.use_dpo = use_dpo
        self.dpo_weight = dpo_weight
        self.dpo_beta = dpo_beta
        self.anchor_regression_threshold = anchor_regression_threshold
        self.run_name = run_name
        self.evaluate_fn = evaluate_fn
        self._optimizer = None
        self._lora_ready = False
        self._skip_next_step = False
        self.anchor_config: dict[str, Any] | None = None
        self.anchor_reward: float | None = None
        self.is_real = isinstance(generator.backend, HFBackend) and lora_mod.peft_available()
        logger.info(
            "TestTimeTrainer mode: %s (backend=%s)",
            "REAL LoRA" if self.is_real else "heuristic adapt",
            generator.backend.name,
        )

    def set_anchor(self, config: dict[str, Any], reward: float) -> None:
        if self.anchor_config is None:
            self.anchor_config = config
            self.anchor_reward = reward
            logger.info("Anchor config set: reward=%.1f", reward)

    def step(self, rows: list[dict[str, Any]], round_idx: int = 0) -> dict[str, Any]:
        if not self.is_real:
            return self.generator.adapt(rows)
        if self._skip_next_step:
            self._skip_next_step = False
            return {"adapted": False, "reason": "skipped due to anchor regression guard"}
        return self._lora_step(rows, round_idx)

    def save_adapter(self) -> Path | None:
        if not self.is_real or not self._lora_ready:
            return None
        backend: HFBackend = self.generator.backend  # type: ignore[assignment]
        out = ADAPTERS_DIR / self.run_name
        out.mkdir(parents=True, exist_ok=True)
        backend.model.save_pretrained(out)
        logger.info("Saved LoRA adapter to %s", out)
        return out

    def _ensure_lora(self) -> None:
        if self._lora_ready:
            return

        backend: HFBackend = self.generator.backend  # type: ignore[assignment]
        adapted = lora_mod.apply_lora(backend.model)
        backend.attach_adapter(adapted)
        adapted.train()
        params = [p for p in adapted.parameters() if p.requires_grad]
        self._optimizer = __import__("torch").optim.AdamW(params, lr=self.lr)
        self._lora_ready = True

    def _check_anchor(self) -> float | None:
        if self.anchor_config is None or self.evaluate_fn is None or self.anchor_reward is None:
            return None
        from ttt.config_space import BurnConfig

        try:
            cfg = BurnConfig.from_dict(self.anchor_config)
        except (KeyError, TypeError, ValueError):
            return None
        result = self.evaluate_fn(cfg)
        delta = float(result.get("reward", 0)) - self.anchor_reward
        if delta < self.anchor_regression_threshold:
            logger.warning(
                "Anchor regression: delta=%.1f (threshold %.1f); will skip next TTT step",
                delta,
                self.anchor_regression_threshold,
            )
            self._skip_next_step = True
        return delta

    def _lora_step(self, rows: list[dict[str, Any]], round_idx: int) -> dict[str, Any]:
        examples = to_sft_examples(self.task, rows)
        pref_pairs = to_preference_pairs(self.task, rows)
        pref_pairs.extend(to_repair_preference_pairs(self.task, rows))

        if not examples and not pref_pairs:
            return {"adapted": False, "reason": "no high-reward examples yet"}

        self._ensure_lora()
        backend: HFBackend = self.generator.backend  # type: ignore[assignment]
        tok = backend.tokenizer
        model = backend.model
        device = next(model.parameters()).device

        sft_losses: list[float] = []
        dpo_losses: list[float] = []

        # dpo_weight trades off the two objectives; when only one is present it
        # gets the full step (the weight only splits a combined SFT+DPO update).
        do_dpo = bool(self.use_dpo and pref_pairs)
        both = bool(examples) and do_dpo
        sft_scale = (1.0 - self.dpo_weight) if both else 1.0
        dpo_scale = self.dpo_weight if both else 1.0

        for _ in range(self.steps_per_round):
            ran = False
            self._optimizer.zero_grad()
            if examples:
                total = 0.0
                for ex in examples:
                    input_ids, labels = self._encode(tok, ex, device)
                    out = model(input_ids=input_ids, labels=labels)
                    (out.loss * sft_scale / len(examples)).backward()
                    total += float(out.loss.detach())
                sft_losses.append(total / len(examples))
                ran = True

            if do_dpo:
                import torch

                n = min(4, len(pref_pairs))
                dpo_total = 0.0
                for pair in pref_pairs[:n]:
                    pi_c = -self._nll(model, tok, pair.system, pair.prompt, pair.chosen, device)
                    pi_r = -self._nll(model, tok, pair.system, pair.prompt, pair.rejected, device)
                    ref_ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
                    with ref_ctx:
                        with torch.no_grad():
                            ref_c = -float(self._nll(model, tok, pair.system, pair.prompt, pair.chosen, device))
                            ref_r = -float(self._nll(model, tok, pair.system, pair.prompt, pair.rejected, device))
                    logit = self.dpo_beta * ((pi_c - pi_r) - (ref_c - ref_r))
                    loss = -torch.nn.functional.logsigmoid(logit)
                    (loss * dpo_scale / n).backward()
                    dpo_total += float(loss.detach())
                dpo_losses.append(dpo_total / n)
                ran = True

            if ran:
                self._optimizer.step()
            else:
                break

        anchor_delta = self._check_anchor()

        info: dict[str, Any] = {
            "adapted": True,
            "n_sft": len(examples),
            "n_dpo": len(pref_pairs),
            "round": round_idx,
        }
        if sft_losses:
            info["sft_loss_first"] = sft_losses[0]
            info["sft_loss_last"] = sft_losses[-1]
        if dpo_losses:
            info["dpo_loss_first"] = dpo_losses[0]
            info["dpo_loss_last"] = dpo_losses[-1]
        if anchor_delta is not None:
            info["anchor_reward_delta"] = anchor_delta

        logger.info("LoRA step: %s", info)
        return info

    def _nll(self, model, tok, system, prompt, completion, device):
        prompt_text = tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        eos = tok.eos_token or ""
        full_ids = tok(
            prompt_text + completion + eos,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_len,
        ).input_ids.to(device)
        prompt_ids = tok(prompt_text, return_tensors="pt").input_ids[0]
        labels = full_ids.clone()
        labels[:, : min(len(prompt_ids), labels.shape[1])] = -100
        # DPO compares sequence log-probabilities, so return the *summed* NLL over
        # the completion tokens rather than HF's length-normalized mean -- otherwise
        # the preference signal is confounded by chosen/rejected length differences.
        out = model(input_ids=full_ids, labels=labels)
        n_tok = (labels != -100).sum().clamp(min=1)
        return out.loss * n_tok

    def _encode(self, tok, ex, device):
        prompt_text = tok.apply_chat_template(
            [{"role": "system", "content": ex.system}, {"role": "user", "content": ex.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tok(prompt_text, return_tensors="pt").input_ids[0]
        full_ids = tok(prompt_text + ex.completion + tok.eos_token, return_tensors="pt").input_ids[0]
        full_ids = full_ids[: self.max_seq_len]
        labels = full_ids.clone()
        labels[: min(len(prompt_ids), len(labels))] = -100
        return full_ids.unsqueeze(0).to(device), labels.unsqueeze(0).to(device)
