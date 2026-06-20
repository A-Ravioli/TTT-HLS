"""Decompose a Qwen2 decoder layer into FPGA-mappable sub-blocks.

A Qwen2 decoder layer is::

    x -> RMSNorm -> Attention(q,k,v,o + RoPE + softmax) -> +x
      -> RMSNorm -> MLP(SwiGLU: gate, up, down)         -> +x

For FPGA mapping via hls4ml we care about the matmul-bearing sub-blocks. The MLP
(SwiGLU) is the friendliest first target -- it is dense matmuls + an elementwise
gate -- so it is the first real (non-toy) milestone. Attention needs softmax /
RoPE which hls4ml does not express cleanly; it is flagged as a research stub.
"""

from __future__ import annotations

from dataclasses import dataclass

from glm.tasks import BlockSpec, LayerSpec
from models.qwen.load_qwen import QwenArch


@dataclass(frozen=True)
class SubBlock:
    spec: BlockSpec
    hls4ml_ready: bool
    note: str = ""


def mlp_block_spec(arch: QwenArch, layer_idx: int = 0) -> BlockSpec:
    """The SwiGLU MLP sub-block (gate_proj, up_proj, down_proj)."""
    h, inter = arch.hidden_size, arch.intermediate_size
    return BlockSpec(
        name=f"{arch.model_id.split('/')[-1]}.layer{layer_idx}.mlp",
        layers=(
            LayerSpec("gate_proj", "mlp_gate", h, inter),
            LayerSpec("up_proj", "mlp_up", h, inter),
            LayerSpec("down_proj", "mlp_down", inter, h),
        ),
        notes="SwiGLU: down(silu(gate(x)) * up(x)). Dense matmuls + elementwise gate.",
    )


def attention_block_spec(arch: QwenArch, layer_idx: int = 0) -> BlockSpec:
    """The attention projections (q/k/v/o). Softmax+RoPE handled outside hls4ml."""
    h = arch.hidden_size
    return BlockSpec(
        name=f"{arch.model_id.split('/')[-1]}.layer{layer_idx}.attn",
        layers=(
            LayerSpec("q_proj", "attention_qkv", h, arch.q_out),
            LayerSpec("k_proj", "attention_qkv", h, arch.kv_out),
            LayerSpec("v_proj", "attention_qkv", h, arch.kv_out),
            LayerSpec("o_proj", "attention_out", arch.q_out, h),
        ),
        notes="GQA projections; softmax(QK^T/sqrt(d))V and RoPE are NOT hls4ml-native.",
    )


def decompose_layer(arch: QwenArch, layer_idx: int = 0) -> list[SubBlock]:
    """Return the FPGA-relevant sub-blocks of one decoder layer."""
    return [
        SubBlock(
            spec=mlp_block_spec(arch, layer_idx),
            hls4ml_ready=True,
            note="First real milestone: dense projections are hls4ml-friendly.",
        ),
        SubBlock(
            spec=attention_block_spec(arch, layer_idx),
            hls4ml_ready=False,
            note="Projections compile; softmax/RoPE need FINN or hand-written HLS (Phase 4).",
        ),
    ]


def full_model_blocks(arch: QwenArch) -> list[SubBlock]:
    """RESEARCH STUB: every sub-block of every decoder layer.

    Full Qwen-2B has ``num_hidden_layers`` decoder layers; mapping all of them
    plus embeddings/lm_head and the cross-block dataflow (KV cache, weight
    streaming, multi-FPGA partitioning) is the north-star, not the first
    milestone. This enumerates the blocks so the scope is explicit; orchestration
    lives in Phase 4.
    """
    blocks: list[SubBlock] = []
    for i in range(arch.num_hidden_layers):
        blocks.extend(decompose_layer(arch, i))
    return blocks


def feasibility_report(arch: QwenArch) -> str:
    """Human-readable summary of what is mappable today vs research TODO."""
    mlp = mlp_block_spec(arch)
    attn = attention_block_spec(arch)
    lines = [
        f"Qwen decomposition feasibility: {arch.model_id}",
        f"  hidden={arch.hidden_size} intermediate={arch.intermediate_size} "
        f"layers={arch.num_hidden_layers} heads={arch.num_attention_heads}/{arch.num_key_value_heads}",
        f"  MLP block MACs/token : {mlp.total_macs():,}  (hls4ml-ready: YES, tile for fit)",
        f"  Attn proj MACs/token : {attn.total_macs():,}  (hls4ml-ready: projections only)",
        f"  Full model is {arch.num_hidden_layers} layers x (attn + mlp) -> needs tiling, "
        f"weight streaming, KV cache (Phase 4 stubs).",
    ]
    return "\n".join(lines)
