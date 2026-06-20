"""Turn feedback trajectories into GLM training examples.

Two flavors, both keyed on the task prompt:

* **SFT**: ``(prompt -> high-reward config JSON)`` -- imitate what worked.
* **Preference**: ``(prompt, chosen, rejected)`` -- a higher-reward config is
  preferred over a lower-reward one (for DPO-style updates).

Pure stdlib; consumed by :mod:`glm.finetune.trainer`.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from typing import Any

from glm.prompts import SYSTEM_PROMPT, build_propose_prompt
from glm.tasks import FpgaTask


@dataclass
class SFTExample:
    system: str
    prompt: str
    completion: str


@dataclass
class PreferencePair:
    system: str
    prompt: str
    chosen: str
    rejected: str


def _completion(config: dict[str, Any]) -> str:
    keep = ("weight_bits", "activation_bits", "int_bits", "reuse_dense_1", "reuse_dense_2", "strategy")
    return json.dumps({k: config.get(k) for k in keep})


def _compiled(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if r.get("compile_success") and r.get("reward") is not None and r.get("config"):
            out.append(r)
    return out


def to_sft_examples(task: FpgaTask, rows: list[dict[str, Any]], top_frac: float = 0.5) -> list[SFTExample]:
    """SFT on configs whose reward is in the top ``top_frac`` for this task."""
    compiled = _compiled(rows)
    if not compiled:
        return []
    rewards = [float(r["reward"]) for r in compiled]
    cutoff = statistics.median(rewards) if len(rewards) > 1 else min(rewards) - 1
    prompt = build_propose_prompt(task.describe(), [], 1)
    examples = []
    for r in sorted(compiled, key=lambda x: float(x["reward"]), reverse=True):
        if float(r["reward"]) >= cutoff:
            examples.append(SFTExample(SYSTEM_PROMPT, prompt, _completion(r["config"])))
    keep = max(1, int(len(examples) * 1.0))
    return examples[:keep]


def to_preference_pairs(task: FpgaTask, rows: list[dict[str, Any]], max_pairs: int = 32) -> list[PreferencePair]:
    """Pair higher-reward configs (chosen) against lower-reward ones (rejected)."""
    compiled = sorted(_compiled(rows), key=lambda x: float(x["reward"]), reverse=True)
    if len(compiled) < 2:
        return []
    prompt = build_propose_prompt(task.describe(), [], 1)
    pairs: list[PreferencePair] = []
    n = len(compiled)
    for i in range(n):
        for j in range(n - 1, i, -1):
            if float(compiled[i]["reward"]) - float(compiled[j]["reward"]) > 1e-6:
                pairs.append(
                    PreferencePair(
                        SYSTEM_PROMPT,
                        prompt,
                        _completion(compiled[i]["config"]),
                        _completion(compiled[j]["config"]),
                    )
                )
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs
