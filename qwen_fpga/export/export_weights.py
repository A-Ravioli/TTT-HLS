"""Export a Qwen checkpoint to the flat FPGA-friendly binary layout.

Produces, under ``--out`` (default ``qwen_fpga/weights/<model-tag>/``):

    manifest.json                         # arch + dtypes + group size + file index
    embed_tokens.fp16.bin                 # [vocab, hidden]  (CPU embedding lookup)
    norm.fp16.bin                         # final RMSNorm gamma [hidden]
    lm_head.int4.bin / .scale.bin         # [vocab, hidden]  (tied to embed by default)
    layers.<i>.input_layernorm.fp16.bin   # [hidden]
    layers.<i>.post_attention_layernorm.fp16.bin
    layers.<i>.attn.{q,k,v,o}_proj.int4.bin / .scale.bin
    layers.<i>.attn.{q,k,v}_proj.bias.fp16.bin   (Qwen2 has qkv bias)
    layers.<i>.mlp.{gate,up,down}_proj.int4.bin / .scale.bin

All linear weights are groupwise-symmetric INT4 (see ``quant.py``). Norm gammas,
biases and the embedding table stay FP16. The exact same layout is consumed by
the Python reference (``reference/qref.py``) and the C++/HLS host runtime.

Usage:
    python -m qwen_fpga.export.export_weights \
        --model Qwen/Qwen2.5-0.5B-Instruct --group-size 128
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qwen_fpga.export.quant import DEFAULT_GROUP_SIZE, quantize_weight_int4


def _to_np(t) -> np.ndarray:
    import torch

    return t.detach().to(torch.float32).cpu().numpy()


def _save_fp16(arr: np.ndarray, path: Path) -> int:
    a = np.asarray(arr, dtype=np.float16)
    a.tofile(path)
    return a.nbytes


def _save_int4(name: str, weight: np.ndarray, out_dir: Path, group_size: int) -> dict:
    qw = quantize_weight_int4(weight, group_size=group_size)
    qw.packed.tofile(out_dir / f"{name}.int4.bin")
    qw.scales.astype(np.float16).tofile(out_dir / f"{name}.scale.bin")
    return {
        "weight": f"{name}.int4.bin",
        "scale": f"{name}.scale.bin",
        "out_features": qw.out_features,
        "in_features": qw.in_features,
        "group_size": qw.group_size,
        "num_groups": qw.num_groups,
        "dtype": "int4_sym_groupwise",
    }


def export(model_id: str, out_dir: Path, group_size: int, max_layers: int | None) -> Path:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.float32
    )
    sd = model.state_dict()

    n_layers = cfg.num_hidden_layers if max_layers is None else min(max_layers, cfg.num_hidden_layers)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    tied = bool(getattr(cfg, "tie_word_embeddings", True))

    manifest: dict = {
        "model_id": model_id,
        "arch": {
            "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size,
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_exported_layers": n_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
            "head_dim": head_dim,
            "vocab_size": cfg.vocab_size,
            "rms_norm_eps": float(getattr(cfg, "rms_norm_eps", 1e-6)),
            "rope_theta": float(getattr(cfg, "rope_theta", 1e6)),
            "max_position_embeddings": int(getattr(cfg, "max_position_embeddings", 32768)),
            "tie_word_embeddings": tied,
        },
        "quant": {"weights": "int4_sym_groupwise", "group_size": group_size,
                  "activations": "int8_sym_pervec", "accum": "int32"},
        "tensors": {},
        "layers": [],
    }
    T = manifest["tensors"]

    embed = _to_np(sd["model.embed_tokens.weight"])
    _save_fp16(embed, out_dir / "embed_tokens.fp16.bin")
    T["embed_tokens"] = {"file": "embed_tokens.fp16.bin", "shape": list(embed.shape), "dtype": "fp16"}

    final_norm = _to_np(sd["model.norm.weight"])
    _save_fp16(final_norm, out_dir / "norm.fp16.bin")
    T["norm"] = {"file": "norm.fp16.bin", "shape": list(final_norm.shape), "dtype": "fp16"}

    lm_w = embed if tied else _to_np(sd["lm_head.weight"])
    T["lm_head"] = _save_int4("lm_head", lm_w, out_dir, group_size)

    for i in range(n_layers):
        p = f"model.layers.{i}."
        L: dict = {"index": i, "attn": {}, "mlp": {}, "norms": {}}
        for nm, key in (("input_layernorm", "input_layernorm.weight"),
                        ("post_attention_layernorm", "post_attention_layernorm.weight")):
            g = _to_np(sd[p + key])
            fn = f"layers.{i}.{nm}.fp16.bin"
            _save_fp16(g, out_dir / fn)
            L["norms"][nm] = {"file": fn, "shape": list(g.shape), "dtype": "fp16"}

        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            w = _to_np(sd[f"{p}self_attn.{proj}.weight"])
            L["attn"][proj] = _save_int4(f"layers.{i}.attn.{proj}", w, out_dir, group_size)
            bkey = f"{p}self_attn.{proj}.bias"
            if bkey in sd:
                b = _to_np(sd[bkey])
                fn = f"layers.{i}.attn.{proj}.bias.fp16.bin"
                _save_fp16(b, out_dir / fn)
                L["attn"][proj]["bias"] = {"file": fn, "shape": list(b.shape)}

        for proj in ("gate_proj", "up_proj", "down_proj"):
            w = _to_np(sd[f"{p}mlp.{proj}.weight"])
            L["mlp"][proj] = _save_int4(f"layers.{i}.mlp.{proj}", w, out_dir, group_size)

        manifest["layers"].append(L)
        print(f"  exported layer {i}")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote manifest + {len(manifest['layers'])} layers to {out_dir}")
    return out_dir / "manifest.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default=None, help="output dir (default qwen_fpga/weights/<tag>)")
    ap.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    ap.add_argument("--max-layers", type=int, default=None,
                    help="export only the first N layers (for quick milestone A/B)")
    args = ap.parse_args()

    tag = args.model.split("/")[-1]
    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "weights" / tag
    export(args.model, out_dir, args.group_size, args.max_layers)


if __name__ == "__main__":
    main()
