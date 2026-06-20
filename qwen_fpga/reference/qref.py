"""Quantized Qwen reference forward pass (the FPGA-equivalent numerics in Python).

Every large linear layer (q/k/v/o, gate/up/down, lm_head) is evaluated with
``gemv_int4_quantized`` -- the *exact* INT4-weight / INT8-activation / INT32-accum
datapath the FPGA GEMV kernel runs. Only the cheap glue (RMSNorm, RoPE, softmax,
SiLU) runs in fp32 on the "CPU", matching the host/FPGA split in the brief.

Two jobs:
  1. Prove the quantized model actually generates coherent tokens (Milestone C
     in software): if this is wrong, the FPGA can only ever be wrong.
  2. Emit golden input/output vectors for the C++/HLS kernel tests (Milestones
     A and B), via ``reference/make_golden.py``.

This file deliberately has no torch dependency at inference time -- it consumes
the exported binaries, so it doubles as a check that the export is self-contained.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qwen_fpga.export.quant import QuantizedWeight, gemv_int4_quantized


@dataclass
class Loaded:
    manifest: dict
    root: Path
    _cache: dict

    @property
    def arch(self) -> dict:
        return self.manifest["arch"]

    def fp16(self, rel: str, shape=None) -> np.ndarray:
        a = np.fromfile(self.root / rel, dtype=np.float16).astype(np.float32)
        return a.reshape(shape) if shape is not None else a

    def qweight(self, spec: dict) -> QuantizedWeight:
        key = spec["weight"]
        if key in self._cache:
            return self._cache[key]
        m, n = spec["out_features"], spec["in_features"]
        packed = np.fromfile(self.root / spec["weight"], dtype=np.uint8).reshape(m, n // 2)
        ng = spec["num_groups"]
        scales = np.fromfile(self.root / spec["scale"], dtype=np.float16).reshape(m, ng)
        qw = QuantizedWeight(packed=packed, scales=scales,
                             out_features=m, in_features=n, group_size=spec["group_size"])
        self._cache[key] = qw
        return qw


def load(manifest_path: str | Path) -> Loaded:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    return Loaded(manifest=manifest, root=manifest_path.parent, _cache={})


def rms_norm(x: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    x = x.astype(np.float32)
    var = np.mean(x * x)
    return (x / np.sqrt(var + eps)) * gamma


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def rope_tables(head_dim: int, theta: float, max_pos: int) -> tuple[np.ndarray, np.ndarray]:
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    pos = np.arange(max_pos, dtype=np.float64)[:, None]
    freqs = pos * inv_freq[None, :]              # [max_pos, head_dim/2]
    emb = np.concatenate([freqs, freqs], axis=1)  # [max_pos, head_dim]
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def apply_rope(vec: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """vec: [n_heads, head_dim]; cos/sin: [head_dim] for this position."""
    d = vec.shape[-1]
    half = d // 2
    rot = np.concatenate([-vec[..., half:], vec[..., :half]], axis=-1)
    return vec * cos[None, :] + rot * sin[None, :]


class QwenQuantRunner:
    """Greedy/sampling decode using the quantized (FPGA-equivalent) kernels."""

    def __init__(self, loaded: Loaded, backend=None):
        self.L = loaded
        # ``backend`` dispatches every GEMV (numpy / C++ / FPGA). None -> numpy.
        self.backend = backend
        a = loaded.arch
        self.hidden = a["hidden_size"]
        self.n_layers = a["num_exported_layers"]
        self.n_heads = a["num_attention_heads"]
        self.n_kv = a["num_key_value_heads"]
        self.head_dim = a["head_dim"]
        self.eps = a["rms_norm_eps"]
        self.vocab = a["vocab_size"]
        self.embed = loaded.fp16("embed_tokens.fp16.bin", (self.vocab, self.hidden))
        self.final_norm = loaded.fp16("norm.fp16.bin")
        self.cos, self.sin = rope_tables(self.head_dim, a["rope_theta"],
                                         min(a["max_position_embeddings"], 8192))
        self.lm_head = loaded.qweight(loaded.manifest["tensors"]["lm_head"])
        # per-layer KV cache: list of (k_list, v_list) appended per position
        self.reset()

    def reset(self) -> None:
        self.kcache = [[] for _ in range(self.n_layers)]
        self.vcache = [[] for _ in range(self.n_layers)]

    def _gemv(self, spec: dict, x: np.ndarray) -> np.ndarray:
        qw = self.L.qweight(spec)
        if self.backend is not None:
            return self.backend.gemv(qw, x)
        return gemv_int4_quantized(qw, x)

    def _maybe_bias(self, spec: dict, y: np.ndarray) -> np.ndarray:
        if "bias" in spec:
            b = self.L.fp16(spec["bias"]["file"])
            return y + b[: y.shape[0]]
        return y

    def layer(self, li: int, x: np.ndarray, pos: int) -> np.ndarray:
        man = self.L.manifest["layers"][li]
        residual = x
        h = rms_norm(x, self.L.fp16(man["norms"]["input_layernorm"]["file"]), self.eps)

        q = self._maybe_bias(man["attn"]["q_proj"], self._gemv(man["attn"]["q_proj"], h))
        k = self._maybe_bias(man["attn"]["k_proj"], self._gemv(man["attn"]["k_proj"], h))
        v = self._maybe_bias(man["attn"]["v_proj"], self._gemv(man["attn"]["v_proj"], h))

        q = q.reshape(self.n_heads, self.head_dim)
        k = k.reshape(self.n_kv, self.head_dim)
        v = v.reshape(self.n_kv, self.head_dim)
        q = apply_rope(q, self.cos[pos], self.sin[pos])
        k = apply_rope(k, self.cos[pos], self.sin[pos])

        self.kcache[li].append(k)
        self.vcache[li].append(v)
        K = np.stack(self.kcache[li], axis=0)  # [T, n_kv, head_dim]
        V = np.stack(self.vcache[li], axis=0)

        rep = self.n_heads // self.n_kv
        scale = 1.0 / np.sqrt(self.head_dim)
        attn_out = np.empty((self.n_heads, self.head_dim), dtype=np.float32)
        for hh in range(self.n_heads):
            kvh = hh // rep
            scores = (K[:, kvh, :] @ q[hh]) * scale       # [T]
            scores -= scores.max()
            w = np.exp(scores)
            w /= w.sum()
            attn_out[hh] = w @ V[:, kvh, :]               # [head_dim]

        attn_flat = attn_out.reshape(-1)
        o = self._gemv(man["attn"]["o_proj"], attn_flat)
        x = residual + o

        residual = x
        h = rms_norm(x, self.L.fp16(man["norms"]["post_attention_layernorm"]["file"]), self.eps)
        gate = self._gemv(man["mlp"]["gate_proj"], h)
        up = self._gemv(man["mlp"]["up_proj"], h)
        hidden = silu(gate) * up
        down = self._gemv(man["mlp"]["down_proj"], hidden)
        return residual + down

    def forward_token(self, token_id: int, pos: int) -> np.ndarray:
        x = self.embed[token_id].astype(np.float32).copy()
        for li in range(self.n_layers):
            x = self.layer(li, x, pos)
        x = rms_norm(x, self.final_norm, self.eps)
        if self.backend is not None:
            return self.backend.gemv(self.lm_head, x)  # logits
        return gemv_int4_quantized(self.lm_head, x)  # logits

    def generate(self, prompt_ids: list[int], max_new: int = 16,
                 temperature: float = 0.0, top_k: int = 40,
                 seed: int = 0) -> list[int]:
        self.reset()
        rng = np.random.default_rng(seed)
        pos = 0
        logits = None
        for t in prompt_ids:
            logits = self.forward_token(t, pos)
            pos += 1
        out: list[int] = []
        for _ in range(max_new):
            if temperature <= 0.0:
                nxt = int(np.argmax(logits))
            else:
                lg = logits / temperature
                if top_k:
                    idx = np.argpartition(lg, -top_k)[-top_k:]
                    mask = np.full_like(lg, -np.inf)
                    mask[idx] = lg[idx]
                    lg = mask
                lg -= lg.max()
                p = np.exp(lg)
                p /= p.sum()
                nxt = int(rng.choice(len(p), p=p))
            out.append(nxt)
            logits = self.forward_token(nxt, pos)
            pos += 1
        return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Quantized Qwen reference decode")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    loaded = load(args.manifest)
    runner = QwenQuantRunner(loaded)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(loaded.manifest["model_id"])
    msgs = [{"role": "user", "content": args.prompt}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                  return_dict=True)
    ids = [int(t) for t in np.asarray(enc["input_ids"]).reshape(-1).tolist()]
    print("prompt tokens:", len(ids))
    out = runner.generate(list(ids), max_new=args.max_new, temperature=args.temperature)
    print("=== generated ===")
    print(tok.decode(out, skip_special_tokens=True))


if __name__ == "__main__":
    main()
