"""Export a TinyStories (GPT-Neo) checkpoint to the flat W8A8 FPGA layout.

Produces, under ``--out`` (default ``tinystories_z2/weights/<tag>/``):

    manifest.json                       # arch + quant + file index
    wte.fp16.bin                        # [vocab, hidden]  token embedding (lookup)
    wpe.fp16.bin                        # [max_pos, hidden] learned position embedding
    ln_f.{w,b}.fp16.bin                 # final LayerNorm
    lm_head.int8.bin / .scale.bin       # [vocab, hidden]  (tied to wte)
    L<i>.ln_1.{w,b}.fp16.bin  L<i>.ln_2.{w,b}.fp16.bin
    L<i>.{q,k,v,o}.int8.bin / .scale.bin    L<i>.o.bias.fp16.bin
    L<i>.{fc,proj}.int8.bin / .scale.bin    L<i>.{fc,proj}.bias.fp16.bin

All large linears are per-row-symmetric INT8 (``tinystories_z2.quant`` -- the same
W8A8 contract the Z2 FPGA GEMV enforces). Embeddings, LayerNorm params and biases
stay FP16. Consumed by ``tinystories_z2.model.QuantWeights``.

Usage:
    python -m tinystories_z2.export_tinystories --model roneneldan/TinyStories-1M
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tinystories_z2.model import NeoArch
from tinystories_z2.quant import quantize_weight_int8


def _to_np(t) -> np.ndarray:
    import torch

    return t.detach().to(torch.float32).cpu().numpy()


def _save_fp16(arr: np.ndarray, path: Path) -> None:
    np.asarray(arr, dtype=np.float16).tofile(path)


def _save_int8(name: str, weight: np.ndarray, out_dir: Path) -> dict:
    qw = quantize_weight_int8(weight)
    qw.q.tofile(out_dir / f"{name}.int8.bin")
    qw.scales.astype(np.float16).tofile(out_dir / f"{name}.scale.bin")
    return {
        "weight": f"{name}.int8.bin",
        "scale": f"{name}.scale.bin",
        "out_features": qw.out_features,
        "in_features": qw.in_features,
        "dtype": "int8_sym_perrow",
    }


def export(model_id: str, out_dir: Path, max_layers: int | None = None) -> Path:
    import torch
    from transformers import AutoModelForCausalLM

    out_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    arch = NeoArch.from_hf_config(model.config)
    sd = model.state_dict()

    n_layers = arch.num_layers if max_layers is None else min(max_layers, arch.num_layers)

    vectors: dict[str, str] = {}

    def save_vec(key: str, arr: np.ndarray) -> None:
        fn = f"{key}.fp16.bin"
        _save_fp16(arr, out_dir / fn)
        vectors[key] = {"file": fn, "shape": list(np.asarray(arr).shape)}

    save_vec("wte", _to_np(sd["transformer.wte.weight"]))
    save_vec("wpe", _to_np(sd["transformer.wpe.weight"]))
    save_vec("ln_f.w", _to_np(sd["transformer.ln_f.weight"]))
    save_vec("ln_f.b", _to_np(sd["transformer.ln_f.bias"]))

    tensors = {"lm_head": _save_int8("lm_head", _to_np(sd["lm_head.weight"]), out_dir)}

    layers = []
    for i in range(n_layers):
        p = f"transformer.h.{i}."
        a = p + "attn.attention."
        save_vec(f"L{i}.ln_1.w", _to_np(sd[p + "ln_1.weight"]))
        save_vec(f"L{i}.ln_1.b", _to_np(sd[p + "ln_1.bias"]))
        save_vec(f"L{i}.ln_2.w", _to_np(sd[p + "ln_2.weight"]))
        save_vec(f"L{i}.ln_2.b", _to_np(sd[p + "ln_2.bias"]))
        save_vec(f"L{i}.o.bias", _to_np(sd[a + "out_proj.bias"]))
        save_vec(f"L{i}.fc.bias", _to_np(sd[p + "mlp.c_fc.bias"]))
        save_vec(f"L{i}.proj.bias", _to_np(sd[p + "mlp.c_proj.bias"]))
        L = {
            "index": i,
            "attn": {
                "q": _save_int8(f"L{i}.q", _to_np(sd[a + "q_proj.weight"]), out_dir),
                "k": _save_int8(f"L{i}.k", _to_np(sd[a + "k_proj.weight"]), out_dir),
                "v": _save_int8(f"L{i}.v", _to_np(sd[a + "v_proj.weight"]), out_dir),
                "o": _save_int8(f"L{i}.o", _to_np(sd[a + "out_proj.weight"]), out_dir),
            },
            "mlp": {
                "fc": _save_int8(f"L{i}.fc", _to_np(sd[p + "mlp.c_fc.weight"]), out_dir),
                "proj": _save_int8(f"L{i}.proj", _to_np(sd[p + "mlp.c_proj.weight"]), out_dir),
            },
        }
        layers.append(L)
        print(f"  exported layer {i}")

    manifest = {
        "model_id": model_id,
        "arch": {
            "hidden_size": arch.hidden_size,
            "num_layers": n_layers,
            "num_heads": arch.num_heads,
            "head_dim": arch.head_dim,
            "vocab_size": arch.vocab_size,
            "max_position_embeddings": arch.max_position_embeddings,
            "intermediate_size": arch.intermediate_size,
            "layer_norm_epsilon": arch.layer_norm_epsilon,
            "window_size": arch.window_size,
            "attention_types": arch.attention_types[:n_layers],
            "tie_word_embeddings": arch.tie_word_embeddings,
        },
        "quant": {"weights": "int8_sym_perrow", "activations": "int8_sym_pervec",
                  "accum": "int32"},
        "vectors": vectors,
        "tensors": tensors,
        "layers": layers,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote manifest + {n_layers} layers to {out_dir}")
    return out_dir / "manifest.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="roneneldan/TinyStories-1M")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-layers", type=int, default=None)
    args = ap.parse_args()

    tag = args.model.split("/")[-1]
    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parent / "weights" / tag
    export(args.model, out_dir, args.max_layers)


if __name__ == "__main__":
    main()
