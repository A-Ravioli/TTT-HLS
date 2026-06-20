"""Tests for TTT dataset gating and preference pair mining."""

from __future__ import annotations

from glm.finetune.dataset import (
    to_preference_pairs,
    to_repair_preference_pairs,
    to_sft_examples,
)
from glm.finetune.grpo import group_advantages, to_grpo_group
from glm.finetune.dataset_hls import to_hls_preference_pairs, to_hls_sft_examples
from glm.tasks import make_task, tiny_ffn_block
from ttt.reward import get_board_budget


def _task(threshold: float = 0.25):
    return make_task(
        block=tiny_ffn_block(),
        target_part="xc7z020clg400-1",
        budget=get_board_budget("xc7z020clg400-1"),
        max_error_threshold=threshold,
    )


def _cfg_row(reward: float, max_error: float | None = 0.01, compile_success: bool = True):
    return {
        "compile_success": compile_success,
        "reward": reward,
        "max_error": max_error,
        "config": {
            "weight_bits": 16,
            "activation_bits": 16,
            "int_bits": 6,
            "reuse_dense_1": 2,
            "reuse_dense_2": 2,
            "strategy": "Latency",
        },
    }


def test_sft_excludes_high_max_error():
    task = _task(0.25)
    rows = [_cfg_row(100, 0.01), _cfg_row(90, 0.5)]
    examples = to_sft_examples(task, rows)
    assert len(examples) == 1


def test_preference_pairs_require_chosen_accuracy():
    task = _task(0.25)
    rows = [_cfg_row(100, 0.01), _cfg_row(50, 0.01), _cfg_row(10, 0.5)]
    pairs = to_preference_pairs(task, rows)
    assert len(pairs) >= 1
    # Bad accuracy row should not appear as chosen (only one valid high-reward row)
    assert all("16" in p.chosen for p in pairs)


def test_repair_preference_pairs_from_same_round():
    task = _task(0.25)
    failed = _cfg_row(-100, None, compile_success=False)
    failed["round"] = 1
    failed["config"]["reuse_dense_1"] = 1
    fixed = _cfg_row(80, 0.01)
    fixed["round"] = 1
    fixed["is_repair"] = True
    fixed["config"]["reuse_dense_1"] = 8
    pairs = to_repair_preference_pairs(task, [failed, fixed])
    assert len(pairs) >= 1
    assert pairs[0].chosen != pairs[0].rejected


def test_hls_sft_requires_cosim_pass():
    task = _task(0.01)
    rows = [
        {
            "hls_compile_success": True,
            "cosim_pass": False,
            "sources": {"kernel_top.cpp": "// bad"},
            "reward": 100.0,
            "max_error": 0.5,
        },
        {
            "hls_compile_success": True,
            "cosim_pass": True,
            "sources": {"kernel_top.cpp": "// good"},
            "reward": 90.0,
            "max_error": 0.005,
        },
    ]
    examples = to_hls_sft_examples(task, rows)
    assert len(examples) == 1


def test_hls_dpo_both_pass_cosim():
    task = _task(0.01)
    rows = [
        {
            "hls_compile_success": True,
            "cosim_pass": True,
            "sources": {"kernel_top.cpp": "// a"},
            "reward": 200.0,
            "max_error": 0.001,
        },
        {
            "hls_compile_success": True,
            "cosim_pass": True,
            "sources": {"kernel_top.cpp": "// b"},
            "reward": 50.0,
            "max_error": 0.002,
        },
    ]
    pairs = to_hls_preference_pairs(task, rows)
    assert len(pairs) == 1


def test_grpo_group_advantages():
    adv = group_advantages([1.0, 3.0, 5.0])
    assert len(adv) == 3
    assert adv[1] == 0.0 or abs(adv[1]) < abs(adv[2])


def test_grpo_group_from_round():
    task = _task()
    rows = [
        {**_cfg_row(10.0), "round": 2},
        {**_cfg_row(50.0), "round": 2},
        {**_cfg_row(5.0), "round": 1},
    ]
    group = to_grpo_group(task, rows, round_idx=2)
    assert len(group) == 2
    assert group[1].advantage > group[0].advantage
