"""Minimal DPO loss for test-time LoRA (no trl dependency)."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Protocol


class PreferenceExample(Protocol):
    system: str
    prompt: str
    chosen: str
    rejected: str


def _completion_nll(
    model,
    tokenizer,
    system: str,
    prompt: str,
    completion: str,
    max_len: int,
    device,
) -> Any:
    """Negative log-likelihood on completion tokens only."""
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    eos = tokenizer.eos_token or ""
    full_ids = tokenizer(
        prompt_text + completion + eos,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    ).input_ids.to(device)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0]
    labels = full_ids.clone()
    labels[:, : min(len(prompt_ids), labels.shape[1])] = -100
    # Summed (not length-normalized) NLL over completion tokens: DPO compares
    # sequence log-probs, so a per-token mean would bias toward shorter sequences.
    out = model(input_ids=full_ids, labels=labels)
    n_tok = (labels != -100).sum().clamp(min=1)
    return out.loss * n_tok


def train_dpo_step(
    model,
    tokenizer,
    pairs: list[PreferenceExample],
    optimizer,
    max_len: int,
    device,
    beta: float = 0.1,
    max_pairs: int = 4,
) -> float:
    """One DPO optimizer step over up to ``max_pairs`` preference pairs."""
    import torch

    if not pairs:
        return 0.0

    n = min(max_pairs, len(pairs))
    total = 0.0
    optimizer.zero_grad()

    for pair in pairs[:n]:
        pi_c = -_completion_nll(model, tokenizer, pair.system, pair.prompt, pair.chosen, max_len, device)
        pi_r = -_completion_nll(model, tokenizer, pair.system, pair.prompt, pair.rejected, max_len, device)

        ref_ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
        with ref_ctx:
            with torch.no_grad():
                ref_c = -float(
                    _completion_nll(model, tokenizer, pair.system, pair.prompt, pair.chosen, max_len, device)
                )
                ref_r = -float(
                    _completion_nll(model, tokenizer, pair.system, pair.prompt, pair.rejected, max_len, device)
                )

        logit = beta * ((pi_c - pi_r) - (ref_c - ref_r))
        loss = -torch.nn.functional.logsigmoid(logit)
        (loss / n).backward()
        total += float(loss.detach())

    optimizer.step()
    return total / n
