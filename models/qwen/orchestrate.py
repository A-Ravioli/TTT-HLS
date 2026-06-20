"""North-star orchestration for full Qwen-2B on FPGA (research stubs).

Mapping one MLP sub-block is the first real milestone (Phase 3). Running the WHOLE
model is the north star and is intentionally stubbed here: this module makes the
remaining work explicit and provides rough, honest planning helpers (capacity
bin-packing, KV-cache sizing, weight-streaming bandwidth) rather than pretending
the hard parts are done.

What is still TODO for a real full-model deployment:
  * Per-block tiling + weight streaming (weights don't fit on-chip).
  * KV cache management across decode steps (off-chip, streamed).
  * Attention softmax/RoPE kernels (not hls4ml-native -> FINN or hand HLS).
  * Cross-block dataflow + multi-FPGA partitioning and host orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass

from models.qwen.decompose import SubBlock, full_model_blocks
from models.qwen.load_qwen import QwenArch
from ttt.reward import get_board_budget

# Rough DSPs needed if a block ran fully parallel at 1 DSP/MAC (upper bound; real
# designs time-multiplex heavily via ReuseFactor, so divide by the reuse factor).
DSP_PER_MAC = 1.0


@dataclass
class BlockPlacement:
    block_name: str
    macs: int
    part: str
    est_dsp_at_reuse: float
    reuse: int
    hls4ml_ready: bool


def est_block_dsp(sub: SubBlock, reuse: int) -> float:
    return DSP_PER_MAC * sub.spec.total_macs() / max(1, reuse)


def plan_full_model(arch: QwenArch, parts: list[str], reuse: int = 64) -> list[BlockPlacement]:
    """Greedy capacity bin-pack of every sub-block across the given FPGA parts.

    This is a *planning* aid, not a synthesized result: it shows roughly how many
    parts a full model would need at a given reuse factor.
    """
    blocks = full_model_blocks(arch)
    budgets = [(p, get_board_budget(p)["dsp"]) for p in parts]
    if not budgets:
        budgets = [("xcu250-figd2104-2l-e", get_board_budget("xcu250-figd2104-2l-e")["dsp"])]

    placements: list[BlockPlacement] = []
    part_idx = 0
    used = 0.0
    for sub in blocks:
        need = est_block_dsp(sub, reuse)
        part, cap = budgets[part_idx]
        if used + need > cap and part_idx < len(budgets) - 1:
            part_idx += 1
            used = 0.0
            part, cap = budgets[part_idx]
        used += need
        placements.append(
            BlockPlacement(
                block_name=sub.spec.name,
                macs=sub.spec.total_macs(),
                part=part,
                est_dsp_at_reuse=round(need, 1),
                reuse=reuse,
                hls4ml_ready=sub.hls4ml_ready,
            )
        )
    return placements


def kv_cache_bytes(arch: QwenArch, seq_len: int, dtype_bytes: int = 2) -> int:
    """KV-cache size for a full forward pass (off-chip, streamed). Stub helper."""
    per_layer = 2 * arch.num_key_value_heads * arch.head_dim * seq_len * dtype_bytes
    return per_layer * arch.num_hidden_layers


def weight_bytes(arch: QwenArch, weight_bits: int = 8) -> int:
    """Approximate total weight footprint (drives streaming bandwidth needs)."""
    blocks = full_model_blocks(arch)
    total_params = sum(sub.spec.total_macs() for sub in blocks)  # MACs ~= weights here
    return int(total_params * weight_bits / 8)


def describe_plan(arch: QwenArch, parts: list[str], reuse: int = 64, seq_len: int = 2048) -> str:
    placements = plan_full_model(arch, parts, reuse=reuse)
    parts_used = sorted({p.part for p in placements})
    not_ready = [p.block_name for p in placements if not p.hls4ml_ready]
    lines = [
        f"FULL-MODEL ORCHESTRATION PLAN (research stub): {arch.model_id}",
        f"  sub-blocks: {len(placements)}  reuse factor: {reuse}",
        f"  parts needed (capacity bin-pack): {len(parts_used)} -> {parts_used}",
        f"  weight footprint @8-bit: {weight_bytes(arch, 8) / 1e6:.1f} MB (needs streaming)",
        f"  KV cache @seq={seq_len}, fp16: {kv_cache_bytes(arch, seq_len) / 1e6:.1f} MB (off-chip)",
        f"  blocks NOT hls4ml-ready (need FINN/hand-HLS): {len(not_ready)} (all attention sub-blocks)",
        "  TODO: tiling, weight streaming, KV cache, softmax/RoPE kernels, host orchestration.",
    ]
    return "\n".join(lines)
