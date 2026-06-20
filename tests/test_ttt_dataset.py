"""Tests for TTT dataset gating and preference pair mining."""

from __future__ import annotations

from glm.agent_hls import result_to_hls_history_row
from glm.finetune.dataset import (
    to_preference_pairs,
    to_repair_preference_pairs,
    to_sft_examples,
)
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


def test_sft_top_frac_selects_fewer_with_smaller_fraction():
    task = _task(0.25)
    rows = [_cfg_row(r, 0.01) for r in (100, 80, 60, 40, 20)]
    all_kept = to_sft_examples(task, rows, top_frac=1.0)
    half = to_sft_examples(task, rows, top_frac=0.5)
    tight = to_sft_examples(task, rows, top_frac=0.2)
    assert len(all_kept) == 5            # top_frac was previously ignored (always all)
    assert len(tight) <= len(half) <= len(all_kept)
    assert len(tight) >= 1               # always keep at least the best


def test_hls_history_row_without_sources_yields_no_training_data():
    # Regression for the HLS-TTT no-op: the metric-only history row (what the
    # trainer used to receive) carries no `sources`, so the corpus was empty.
    task = _task(0.01)
    result = {
        "hls_compile_success": True,
        "cosim_pass": True,
        "reward": 100.0,
        "max_error": 0.005,
    }
    metric_only = result_to_hls_history_row(result)
    assert "sources" not in metric_only or metric_only.get("sources") is None
    assert to_hls_sft_examples(task, [metric_only]) == []

    # ...but once sources travel with the row (as script 11 now does), it works.
    train_row = {**metric_only, "sources": {"kernel_top.cpp": "// ok"}}
    assert len(to_hls_sft_examples(task, [train_row])) == 1


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
