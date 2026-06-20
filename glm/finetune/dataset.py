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


def _passes_accuracy(row: dict[str, Any], max_error_threshold: float) -> bool:
    max_err = row.get("max_error")
    if max_err is None:
        return True
    try:
        return float(max_err) <= max_error_threshold
    except (TypeError, ValueError):
        return False


def _valid_config(row: dict[str, Any], max_error_threshold: float) -> bool:
    """Row is usable for SFT: compiled, rewarded, and within accuracy budget."""
    return (
        bool(row.get("compile_success"))
        and row.get("reward") is not None
        and row.get("config") is not None
        and _passes_accuracy(row, max_error_threshold)
    )


def _valid_for_rejected(row: dict[str, Any], max_error_threshold: float) -> bool:
    """Rejected side of DPO: compile failures OK; compiled rows must pass accuracy."""
    if not row.get("config"):
        return False
    if row.get("compile_success"):
        return _passes_accuracy(row, max_error_threshold)
    return True


def _valid_for_chosen(row: dict[str, Any], max_error_threshold: float) -> bool:
    return _valid_config(row, max_error_threshold)


def to_sft_examples(
    task: FpgaTask,
    rows: list[dict[str, Any]],
    top_frac: float = 0.5,
) -> list[SFTExample]:
    """SFT on configs whose reward is in the top ``top_frac`` for this task."""
    threshold = task.max_error_threshold
    compiled = [r for r in rows if _valid_config(r, threshold)]
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


def to_preference_pairs(
    task: FpgaTask,
    rows: list[dict[str, Any]],
    max_pairs: int = 32,
) -> list[PreferencePair]:
    """Pair higher-reward configs (chosen) against lower-reward ones (rejected)."""
    threshold = task.max_error_threshold
    chosen_pool = sorted(
        [r for r in rows if _valid_for_chosen(r, threshold)],
        key=lambda x: float(x["reward"]),
        reverse=True,
    )
    if len(chosen_pool) < 1:
        return []
    prompt = build_propose_prompt(task.describe(), [], 1)
    pairs: list[PreferencePair] = []
    n = len(chosen_pool)
    for i in range(n):
        for j in range(n - 1, i, -1):
            rejected = chosen_pool[j]
            if not _valid_for_rejected(rejected, threshold):
                continue
            if float(chosen_pool[i]["reward"]) - float(rejected["reward"]) > 1e-6:
                pairs.append(
                    PreferencePair(
                        SYSTEM_PROMPT,
                        prompt,
                        _completion(chosen_pool[i]["config"]),
                        _completion(rejected["config"]),
                    )
                )
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def to_repair_preference_pairs(
    task: FpgaTask,
    rows: list[dict[str, Any]],
    max_pairs: int = 16,
) -> list[PreferencePair]:
    """Mine (failed, repaired) pairs from repair-tagged rows or trajectory metadata."""
    threshold = task.max_error_threshold
    prompt = build_propose_prompt(task.describe(), [], 1)
    pairs: list[PreferencePair] = []

    # Explicit repair tags from trajectory store or search loop.
    for i, r in enumerate(rows):
        method = str(r.get("method", ""))
        if not (method.endswith("_repair") or r.get("is_repair")):
            continue
        if not _valid_for_chosen(r, threshold):
            continue
        round_idx = r.get("round")
        failed = None
        for j in range(i - 1, -1, -1):
            prev = rows[j]
            if prev.get("round") != round_idx:
                break
            if not prev.get("compile_success"):
                failed = prev
                break
        if failed and failed.get("config"):
            pairs.append(
                PreferencePair(
                    SYSTEM_PROMPT,
                    prompt,
                    _completion(r["config"]),
                    _completion(failed["config"]),
                )
            )
            if len(pairs) >= max_pairs:
                return pairs

    # Consecutive same-round: fail then success with higher reward.
    for i in range(1, len(rows)):
        prev, cur = rows[i - 1], rows[i]
        if prev.get("round") != cur.get("round"):
            continue
        if prev.get("compile_success"):
            continue
        if not _valid_for_chosen(cur, threshold):
            continue
        if not cur.get("config") or not prev.get("config"):
            continue
        if float(cur.get("reward", -1e9)) > float(prev.get("reward", -1e9)):
            pairs.append(
                PreferencePair(
                    SYSTEM_PROMPT,
                    prompt,
                    _completion(cur["config"]),
                    _completion(prev["config"]),
                )
            )
            if len(pairs) >= max_pairs:
                return pairs

    return pairs
