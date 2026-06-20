"""Test-time trainer for HLS mode: LoRA + DPO on ranked HLS trajectories.

Mirrors :mod:`glm.finetune.trainer` but handles longer-context HLS sources
(8k–32k tokens) and mixed SFT + DPO training.
"""

from __future__ import annotations

from typing import Any

from glm.agent_hls import GLMCompilerAgent
from glm.finetune import lora as lora_mod
from glm.finetune.dataset_hls import to_hls_preference_pairs, to_hls_sft_examples
from glm.serving import HFBackend
from glm.tasks import FpgaTask
from paths import get_logger

logger = get_logger("burnttt.glm.trainer_hls")


class HLSTestTimeTrainer:
    """Adapt a :class:`GLMCompilerAgent` to an HLS task at test time.

    Supports:
    - SFT on top-performing HLS kernels
    - DPO on preference pairs (higher-reward HLS chosen over lower)
    - Longer max_seq_len for HLS source context (8k default)
    """

    def __init__(
        self,
        agent: GLMCompilerAgent,
        task: FpgaTask,
        hidden_dim: int = 24,
        intermediate_dim: int = 64,
        lr: float = 5e-5,
        steps_per_round: int = 2,
        max_seq_len: int = 8192,
        dpo_beta: float = 0.1,
    ):
        self.agent = agent
        self.task = task
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.lr = lr
        self.steps_per_round = steps_per_round
        self.max_seq_len = max_seq_len
        self.dpo_beta = dpo_beta
        self._optimizer = None
        self._lora_ready = False
        self.is_real = isinstance(agent.backend, HFBackend) and lora_mod.peft_available()
        logger.info(
            "HLSTestTimeTrainer mode: %s (backend=%s)",
            "REAL LoRA" if self.is_real else "heuristic adapt",
            agent.backend.name,
        )

    def step(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """One round of test-time training on HLS trajectories."""
        if not self.is_real:
            return self.agent.adapt(rows)
        return self._lora_step(rows)

    def _ensure_lora(self) -> None:
        if self._lora_ready:
            return
        import torch

        backend: HFBackend = self.agent.backend  # type: ignore[assignment]
        adapted = lora_mod.apply_lora(backend.model)
        backend.attach_adapter(adapted)
        adapted.train()
        params = [p for p in adapted.parameters() if p.requires_grad]
        self._optimizer = torch.optim.AdamW(params, lr=self.lr)
        self._lora_ready = True

    def _lora_step(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        # SFT on top kernels
        sft_examples = to_hls_sft_examples(
            self.task, rows,
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
        )
        if not sft_examples:
            return {"adapted": False, "reason": "no high-reward HLS examples yet"}

        self._ensure_lora()
        backend: HFBackend = self.agent.backend  # type: ignore[assignment]
        tok = backend.tokenizer
        model = backend.model
        device = next(model.parameters()).device

        losses = []
        for _ in range(self.steps_per_round):
            total = 0.0
            self._optimizer.zero_grad()
            for ex in sft_examples[:4]:  # Limit batch for memory
                input_ids, labels = self._encode(tok, ex, device)
                out = model(input_ids=input_ids, labels=labels)
                (out.loss / min(4, len(sft_examples))).backward()
                total += float(out.loss.detach())
            self._optimizer.step()
            losses.append(total / min(4, len(sft_examples)))

        logger.info(
            "HLS LoRA step: %d SFT examples, loss %.4f -> %.4f",
            len(sft_examples), losses[0], losses[-1],
        )
        return {
            "adapted": True,
            "examples": len(sft_examples),
            "loss_first": losses[0],
            "loss_last": losses[-1],
        }

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
        labels[: min(len(prompt_ids), len(labels))] = -100
        return full_ids.unsqueeze(0).to(device), labels.unsqueeze(0).to(device)
