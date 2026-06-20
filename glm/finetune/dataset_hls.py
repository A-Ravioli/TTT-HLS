"""Turn HLS feedback trajectories into GLM training examples (Phase 4+).

Like :mod:`glm.finetune.dataset` but for HLS sources instead of JSON configs.
Produces:
- **SFT**: (prompt → high-reward HLS source) — imitate what compiled + passed cosim.
- **DPO pairs**: (prompt, chosen_hls, rejected_hls) — rank by reward.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from glm.prompts.hls_templates import HLS_SYSTEM_PROMPT, build_hls_propose_prompt
from glm.tasks import FpgaTask


@dataclass
class HLSSFTExample:
    system: str
    prompt: str
    completion: str  # The full HLS source (kernel_top.h + kernel_top.cpp)


@dataclass
class HLSPreferencePair:
    system: str
    prompt: str
    chosen: str
    rejected: str


def _sources_to_completion(sources: dict[str, str]) -> str:
    """Format kernel sources as the completion the LLM should have produced."""
    parts = []
    for filename in sorted(sources.keys()):
        parts.append(f"```cpp\n// file: {filename}\n{sources[filename]}\n```")
    return "\n\n".join(parts)


def _valid_hls(row: dict[str, Any]) -> bool:
    """A trajectory row is usable for SFT if it compiled and has sources."""
    return (
        row.get("hls_compile_success", False)
        and row.get("sources") is not None
        and row.get("reward") is not None
    )


def to_hls_sft_examples(
    task: FpgaTask,
    rows: list[dict[str, Any]],
    top_frac: float = 0.5,
    hidden_dim: int = 24,
    intermediate_dim: int = 64,
) -> list[HLSSFTExample]:
    """SFT on HLS kernels whose reward is in the top fraction for this task."""
    valid = [r for r in rows if _valid_hls(r)]
    if not valid:
        return []

    rewards = sorted([float(r["reward"]) for r in valid], reverse=True)
    cutoff_idx = max(1, int(len(rewards) * top_frac))
    cutoff = rewards[min(cutoff_idx, len(rewards) - 1)]

    prompt = build_hls_propose_prompt(
        block_spec=task.describe(),
        part=task.target_part,
        clock_ns=3.3,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
    )

    examples = []
    for r in sorted(valid, key=lambda x: float(x["reward"]), reverse=True):
        if float(r["reward"]) >= cutoff:
            completion = _sources_to_completion(r["sources"])
            examples.append(HLSSFTExample(HLS_SYSTEM_PROMPT, prompt, completion))
    return examples


def to_hls_preference_pairs(
    task: FpgaTask,
    rows: list[dict[str, Any]],
    max_pairs: int = 32,
    hidden_dim: int = 24,
    intermediate_dim: int = 64,
) -> list[HLSPreferencePair]:
    """DPO pairs: higher-reward HLS kernels preferred over lower-reward ones."""
    valid = sorted(
        [r for r in rows if _valid_hls(r)],
        key=lambda x: float(x["reward"]),
        reverse=True,
    )
    if len(valid) < 2:
        return []

    prompt = build_hls_propose_prompt(
        block_spec=task.describe(),
        part=task.target_part,
        clock_ns=3.3,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
    )

    pairs: list[HLSPreferencePair] = []
    n = len(valid)
    for i in range(n):
        for j in range(n - 1, i, -1):
            if float(valid[i]["reward"]) - float(valid[j]["reward"]) > 10.0:
                pairs.append(
                    HLSPreferencePair(
                        HLS_SYSTEM_PROMPT,
                        prompt,
                        _sources_to_completion(valid[i]["sources"]),
                        _sources_to_completion(valid[j]["sources"]),
                    )
                )
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs
