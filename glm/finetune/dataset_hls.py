"""Turn HLS feedback trajectories into GLM training examples (Phase 4+).

Like :mod:`glm.finetune.dataset` but for HLS sources instead of JSON configs.
Produces:
- **SFT**: (prompt → high-reward HLS source) — imitate what compiled + passed cosim.
- **DPO pairs**: (prompt, chosen_hls, rejected_hls) — rank by reward.
"""

from __future__ import annotations

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


# Files supplied to the kernel rather than authored by the model -- excluded from
# the training completion (we don't want the LLM learning to emit baked weights).
_PROVIDED_FILES = {"weights.h"}


def _sources_to_completion(sources: dict[str, str]) -> str:
    """Format kernel sources as the completion the LLM should have produced."""
    parts = []
    for filename in sorted(sources.keys()):
        if filename in _PROVIDED_FILES:
            continue
        parts.append(f"```cpp\n// file: {filename}\n{sources[filename]}\n```")
    return "\n\n".join(parts)


def _passes_accuracy(row: dict[str, Any], max_error_threshold: float) -> bool:
    max_err = row.get("max_error")
    if max_err is None:
        return True
    try:
        return float(max_err) <= max_error_threshold
    except (TypeError, ValueError):
        return False


def _valid_hls(row: dict[str, Any], max_error_threshold: float = 0.01) -> bool:
    """Row is usable for SFT/DPO chosen: compiled, cosim pass, within accuracy."""
    return (
        bool(row.get("hls_compile_success", False))
        and bool(row.get("cosim_pass", False))
        and row.get("sources") is not None
        and row.get("reward") is not None
        and _passes_accuracy(row, max_error_threshold)
    )


def _hls_propose_prompt(
    task: FpgaTask,
    hidden_dim: int,
    intermediate_dim: int,
) -> str:
    return build_hls_propose_prompt(
        block_spec=task.describe(),
        part=task.target_part,
        clock_ns=3.3,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
    )


def to_hls_sft_examples(
    task: FpgaTask,
    rows: list[dict[str, Any]],
    top_frac: float = 0.5,
    hidden_dim: int = 24,
    intermediate_dim: int = 64,
) -> list[HLSSFTExample]:
    """SFT on HLS kernels whose reward is in the top fraction for this task."""
    threshold = task.max_error_threshold
    valid = [r for r in rows if _valid_hls(r, threshold)]
    if not valid:
        return []

    rewards = sorted([float(r["reward"]) for r in valid], reverse=True)
    cutoff_idx = max(1, int(len(rewards) * top_frac))
    cutoff = rewards[min(cutoff_idx, len(rewards) - 1)]

    prompt = _hls_propose_prompt(task, hidden_dim, intermediate_dim)

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
    threshold = task.max_error_threshold
    valid = sorted(
        [r for r in rows if _valid_hls(r, threshold)],
        key=lambda x: float(x["reward"]),
        reverse=True,
    )
    if len(valid) < 2:
        return []

    prompt = _hls_propose_prompt(task, hidden_dim, intermediate_dim)

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


def to_hls_repair_preference_pairs(
    task: FpgaTask,
    rows: list[dict[str, Any]],
    max_pairs: int = 16,
    hidden_dim: int = 24,
    intermediate_dim: int = 64,
) -> list[HLSPreferencePair]:
    """Mine (failed, repaired) HLS pairs from repair-tagged trajectories."""
    threshold = task.max_error_threshold
    prompt = _hls_propose_prompt(task, hidden_dim, intermediate_dim)
    pairs: list[HLSPreferencePair] = []

    for r in rows:
        method = str(r.get("method", ""))
        if "_repair" not in method and not r.get("is_repair"):
            continue
        if not _valid_hls(r, threshold):
            continue
        round_idx = r.get("round_idx", r.get("round"))
        failed = None
        for prev in rows:
            if prev is r:
                break
            prev_round = prev.get("round_idx", prev.get("round"))
            if prev_round == round_idx and not prev.get("cosim_pass"):
                failed = prev
        if failed and failed.get("sources"):
            pairs.append(
                HLSPreferencePair(
                    HLS_SYSTEM_PROMPT,
                    prompt,
                    _sources_to_completion(r["sources"]),
                    _sources_to_completion(failed["sources"]),
                )
            )
            if len(pairs) >= max_pairs:
                return pairs

    return pairs
