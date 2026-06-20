"""Group Relative Policy Optimization for test-time LoRA on compiler rewards."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

from glm.prompts import SYSTEM_PROMPT, build_propose_prompt
from glm.tasks import FpgaTask


@dataclass
class GRPOExample:
    system: str
    prompt: str
    completion: str
    reward: float
    advantage: float


def _completion(config: dict[str, Any]) -> str:
    import json

    keep = ("weight_bits", "activation_bits", "int_bits", "reuse_dense_1", "reuse_dense_2", "strategy")
    return json.dumps({k: config.get(k) for k in keep})


def group_advantages(rewards: list[float], eps: float = 1e-6) -> list[float]:
    """Normalize rewards within a group (DeepSeek-style GRPO advantages)."""
    if not rewards:
        return []
    if len(rewards) == 1:
        return [0.0]
    mu = statistics.mean(rewards)
    std = statistics.pstdev(rewards)
    if std < eps:
        return [r - mu for r in rewards]
    return [(r - mu) / (std + eps) for r in rewards]


def to_grpo_group(
    task: FpgaTask,
    rows: list[dict[str, Any]],
    round_idx: int,
    *,
    min_group: int = 2,
) -> list[GRPOExample]:
    """Build GRPO training examples from one search round's evaluated configs."""
    group_rows = [r for r in rows if r.get("round") == round_idx and r.get("config")]
    if len(group_rows) < min_group:
        return []

    rewards = [float(r.get("reward", 0.0)) for r in group_rows]
    advantages = group_advantages(rewards)
    prompt = build_propose_prompt(task.describe(), [], 1)
    out: list[GRPOExample] = []
    for row, adv in zip(group_rows, advantages):
        out.append(
            GRPOExample(
                SYSTEM_PROMPT,
                prompt,
                _completion(row["config"]),
                float(row.get("reward", 0.0)),
                adv,
            )
        )
    return out


def grpo_policy_loss(
    model,
    tokenizer,
    examples: list[GRPOExample],
    device,
    max_len: int,
    apply_chat_template=None,
) -> tuple[Any, dict[str, float]]:
    """Policy-gradient loss: -E[ advantage * log pi(completion|prompt) ]."""
    import torch

    if not examples:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, {"grpo_loss": 0.0, "grpo_n": 0}

    total = torch.tensor(0.0, device=device)
    n = 0
    for ex in examples:
        if apply_chat_template is not None:
            prompt_text = apply_chat_template(
                [{"role": "system", "content": ex.system}, {"role": "user", "content": ex.prompt}]
            )
        else:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "system", "content": ex.system}, {"role": "user", "content": ex.prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        eos = tokenizer.eos_token or ""
        full_ids = tokenizer(
            prompt_text + ex.completion + eos,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
        ).input_ids.to(device)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0]
        labels = full_ids.clone()
        labels[:, : min(len(prompt_ids), labels.shape[1])] = -100
        nll = model(input_ids=full_ids, labels=labels).loss
        logp = -nll
        total = total + (-ex.advantage * logp)
        n += 1

    loss = total / max(1, n)
    stats = {
        "grpo_loss": float(loss.detach()),
        "grpo_n": n,
        "grpo_adv_max": max(ex.advantage for ex in examples),
        "grpo_adv_min": min(ex.advantage for ex in examples),
        "grpo_reward_max": max(ex.reward for ex in examples),
        "grpo_reward_min": min(ex.reward for ex in examples),
    }
    return loss, stats
