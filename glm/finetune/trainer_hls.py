"""Test-time trainer for HLS mode: LoRA + DPO on ranked HLS trajectories."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

from glm.agent_hls import GLMCompilerAgent
from glm.finetune import lora as lora_mod
from glm.finetune.dataset_hls import (
    to_hls_preference_pairs,
    to_hls_repair_preference_pairs,
    to_hls_sft_examples,
)
from glm.serving import HFBackend
from glm.tasks import FpgaTask
from paths import REPO_ROOT, get_logger

logger = get_logger("burnttt.glm.trainer_hls")

ADAPTERS_DIR = REPO_ROOT / "data" / "adapters"


class HLSTestTimeTrainer:
    """Adapt a :class:`GLMCompilerAgent` to an HLS task at test time."""

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
        use_dpo: bool = True,
        run_name: str = "glm_hls_ttt",
    ):
        self.agent = agent
        self.task = task
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.lr = lr
        self.steps_per_round = steps_per_round
        self.max_seq_len = max_seq_len
        self.dpo_beta = dpo_beta
        self.use_dpo = use_dpo
        self.run_name = run_name
        self._optimizer = None
        self._lora_ready = False
        self.is_real = isinstance(agent.backend, HFBackend) and lora_mod.peft_available()
        logger.info(
            "HLSTestTimeTrainer mode: %s (backend=%s)",
            "REAL LoRA" if self.is_real else "heuristic adapt",
            agent.backend.name,
        )

    def step(self, rows: list[dict[str, Any]], round_idx: int = 0) -> dict[str, Any]:
        if not self.is_real:
            return self.agent.adapt(rows)
        return self._lora_step(rows, round_idx)

    def save_adapter(self) -> Path | None:
        if not self.is_real or not self._lora_ready:
            return None
        backend: HFBackend = self.agent.backend  # type: ignore[assignment]
        out = ADAPTERS_DIR / self.run_name
        out.mkdir(parents=True, exist_ok=True)
        backend.model.save_pretrained(out)
        logger.info("Saved HLS LoRA adapter to %s", out)
        return out

    def _ensure_lora(self) -> None:
        if self._lora_ready:
            return
        backend: HFBackend = self.agent.backend  # type: ignore[assignment]
        adapted = lora_mod.apply_lora(backend.model)
        backend.attach_adapter(adapted)
        adapted.train()
        params = [p for p in adapted.parameters() if p.requires_grad]
        self._optimizer = __import__("torch").optim.AdamW(params, lr=self.lr)
        self._lora_ready = True

    def _lora_step(self, rows: list[dict[str, Any]], round_idx: int) -> dict[str, Any]:
        sft_examples = to_hls_sft_examples(
            self.task,
            rows,
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
        )
        pref_pairs = to_hls_preference_pairs(
            self.task,
            rows,
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
        )
        pref_pairs.extend(
            to_hls_repair_preference_pairs(
                self.task,
                rows,
                hidden_dim=self.hidden_dim,
                intermediate_dim=self.intermediate_dim,
            )
        )

        if not sft_examples and not pref_pairs:
            return {"adapted": False, "reason": "no high-reward HLS examples yet"}

        self._ensure_lora()
        backend: HFBackend = self.agent.backend  # type: ignore[assignment]
        tok = backend.tokenizer
        model = backend.model
        device = next(model.parameters()).device

        sft_losses: list[float] = []
        dpo_losses: list[float] = []

        for _ in range(self.steps_per_round):
            ran = False
            self._optimizer.zero_grad()
            n_sft = min(4, len(sft_examples))
            if sft_examples:
                total = 0.0
                for ex in sft_examples[:n_sft]:
                    input_ids, labels = self._encode(tok, ex, device)
                    out = model(input_ids=input_ids, labels=labels)
                    (out.loss / n_sft).backward()
                    total += float(out.loss.detach())
                sft_losses.append(total / n_sft)
                ran = True

            if self.use_dpo and pref_pairs:
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
                    (loss / n).backward()
                    dpo_total += float(loss.detach())
                dpo_losses.append(dpo_total / n)
                ran = True

            if ran:
                self._optimizer.step()
            else:
                break

        info: dict[str, Any] = {
            "adapted": True,
            "n_sft": len(sft_examples),
            "n_dpo": len(pref_pairs),
            "round": round_idx,
        }
        if sft_losses:
            info["sft_loss_first"] = sft_losses[0]
            info["sft_loss_last"] = sft_losses[-1]
        if dpo_losses:
            info["dpo_loss_first"] = dpo_losses[0]
            info["dpo_loss_last"] = dpo_losses[-1]
        logger.info("HLS LoRA step: %s", info)
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
        # Summed NLL over completion tokens (see config-mode trainer): DPO needs
        # sequence log-probs, not HF's length-normalized mean.
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
