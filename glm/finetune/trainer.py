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

from typing import Any

from glm.agent import GLMGenerator
from glm.finetune import lora as lora_mod
from glm.finetune.dataset import to_sft_examples
from glm.serving import HFBackend
from glm.tasks import FpgaTask
from paths import get_logger

logger = get_logger("burnttt.glm.trainer")


class TestTimeTrainer:
    """Adapt a :class:`~glm.agent.GLMGenerator` to a task at test time."""

    def __init__(
        self,
        generator: GLMGenerator,
        task: FpgaTask,
        lr: float = 1e-4,
        steps_per_round: int = 4,
        max_seq_len: int = 1024,
    ):
        self.generator = generator
        self.task = task
        self.lr = lr
        self.steps_per_round = steps_per_round
        self.max_seq_len = max_seq_len
        self._optimizer = None
        self._lora_ready = False
        self.is_real = isinstance(generator.backend, HFBackend) and lora_mod.peft_available()
        logger.info(
            "TestTimeTrainer mode: %s (backend=%s)",
            "REAL LoRA" if self.is_real else "heuristic adapt",
            generator.backend.name,
        )

    # -- public API --------------------------------------------------------
    def step(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not self.is_real:
            return self.generator.adapt(rows)
        return self._lora_step(rows)

    # -- real LoRA path ----------------------------------------------------
    def _ensure_lora(self) -> None:
        if self._lora_ready:
            return
        import torch

        backend: HFBackend = self.generator.backend  # type: ignore[assignment]
        adapted = lora_mod.apply_lora(backend.model)
        backend.attach_adapter(adapted)
        adapted.train()
        params = [p for p in adapted.parameters() if p.requires_grad]
        self._optimizer = torch.optim.AdamW(params, lr=self.lr)
        self._lora_ready = True

    def _lora_step(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        examples = to_sft_examples(self.task, rows)
        if not examples:
            return {"adapted": False, "reason": "no high-reward examples yet"}

        self._ensure_lora()
        backend: HFBackend = self.generator.backend  # type: ignore[assignment]
        tok = backend.tokenizer
        model = backend.model
        device = next(model.parameters()).device

        losses = []
        for _ in range(self.steps_per_round):
            total = 0.0
            self._optimizer.zero_grad()
            for ex in examples:
                input_ids, labels = self._encode(tok, ex, device)
                out = model(input_ids=input_ids, labels=labels)
                (out.loss / len(examples)).backward()
                total += float(out.loss.detach())
            self._optimizer.step()
            losses.append(total / len(examples))
        logger.info("LoRA step: %d examples, loss %.4f -> %.4f", len(examples), losses[0], losses[-1])
        return {"adapted": True, "examples": len(examples), "loss_first": losses[0], "loss_last": losses[-1]}

    def _encode(self, tok, ex, device):
        import torch

        prompt_text = tok.apply_chat_template(
            [{"role": "system", "content": ex.system}, {"role": "user", "content": ex.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tok(prompt_text, return_tensors="pt").input_ids[0]
        full_ids = tok(prompt_text + ex.completion + tok.eos_token, return_tensors="pt").input_ids[0]
        full_ids = full_ids[: self.max_seq_len]
        labels = full_ids.clone()
        labels[: min(len(prompt_ids), len(labels))] = -100  # mask the prompt
        return full_ids.unsqueeze(0).to(device), labels.unsqueeze(0).to(device)
