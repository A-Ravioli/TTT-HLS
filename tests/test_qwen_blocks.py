"""Tests for Qwen decomposition, per-part budgets, and the richer BlockConfig."""

import pytest

from glm.tasks import make_task
from models.qwen.decompose import (
    attention_block_spec,
    decompose_layer,
    full_model_blocks,
    mlp_block_spec,
)
from models.qwen.load_qwen import load_qwen_arch
from ttt.config_space import BlockConfig, BurnConfig, LayerKnobs
from ttt.reward import fits_board, get_board_budget, reward


def test_qwen_arch_from_table():
    arch = load_qwen_arch("Qwen/Qwen2-1.5B")
    assert arch.hidden_size == 1536
    assert arch.intermediate_size == 8960
    assert arch.num_hidden_layers == 28


def test_mlp_decomposition_shapes():
    arch = load_qwen_arch("Qwen/Qwen2-1.5B")
    mlp = mlp_block_spec(arch)
    names = {l.name for l in mlp.layers}
    assert names == {"gate_proj", "up_proj", "down_proj"}
    # gate + up + down MACs.
    assert mlp.total_macs() == 1536 * 8960 + 1536 * 8960 + 8960 * 1536


def test_attention_is_research_stub():
    arch = load_qwen_arch("Qwen/Qwen2-1.5B")
    subs = decompose_layer(arch)
    by_ready = {sb.hls4ml_ready for sb in subs}
    assert by_ready == {True, False}  # mlp ready, attention not
    assert attention_block_spec(arch).total_macs() > 0


def test_full_model_enumerates_all_layers():
    arch = load_qwen_arch("Qwen/Qwen2-0.5B")
    blocks = full_model_blocks(arch)
    assert len(blocks) == arch.num_hidden_layers * 2


# -- per-part budgets -------------------------------------------------------

def test_per_part_budget_lookup():
    pynq = get_board_budget("xc7z020clg400-1")
    alveo = get_board_budget("xcu250-figd2104-2l-e")
    assert alveo["dsp"] > pynq["dsp"] * 10
    # Unknown part falls back to the default (PYNQ-Z2).
    assert get_board_budget("totally-unknown-part")["dsp"] == pynq["dsp"]


def test_reward_uses_part_budget():
    # 1000 DSPs over-budget on PYNQ-Z2 (220) but within an Alveo (12288).
    base = {"compile_success": True, "max_error": 0.01, "latency_cycles": 100, "dsp": 1000, "lut": 1000, "bram": 2}
    on_pynq = reward({**base, "target_part": "xc7z020clg400-1"})
    on_alveo = reward({**base, "target_part": "xcu250-figd2104-2l-e"})
    assert on_alveo > on_pynq  # no over-budget penalty on the big part
    assert fits_board({**base, "target_part": "xcu250-figd2104-2l-e"}) is True
    assert fits_board({**base, "target_part": "xc7z020clg400-1"}) is False


# -- richer BlockConfig -----------------------------------------------------

def test_block_config_roundtrip_and_uniform():
    arch = load_qwen_arch("Qwen/Qwen2-1.5B")
    names = [l.name for l in mlp_block_spec(arch).layers]
    base = BurnConfig(12, 12, 4, 8, 8, "Resource")
    bc = BlockConfig.uniform(names, base)
    bc2 = BlockConfig.from_dict(bc.to_dict())
    assert set(bc2.layers) == set(names)
    assert bc2.layers["gate_proj"] == LayerKnobs(12, 12, 4, 8)


def test_block_config_rejects_bad_io_type():
    with pytest.raises(ValueError):
        BlockConfig(layers={"a": LayerKnobs(8, 8, 3, 1)}, io_type="nonsense")


def test_layer_knobs_int_bits_constraint():
    with pytest.raises(ValueError):
        LayerKnobs(8, 8, 8, 1)


def test_make_task_for_mlp():
    arch = load_qwen_arch("Qwen/Qwen2-1.5B")
    task = make_task(mlp_block_spec(arch), "xcu250-figd2104-2l-e", get_board_budget("xcu250-figd2104-2l-e"))
    assert "mlp" in task.name
    assert task.budget["dsp"] == 12288
