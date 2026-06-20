"""GPT-Neo (TinyStories) decode with a pluggable linear-layer backend.

The transformer "glue" (LayerNorm, GeLU, learned position embeddings, attention
softmax) runs in fp32 here exactly as it will on the PYNQ Z2's ARM core. Every
large linear (q/k/v/out, c_fc/c_proj, lm_head) goes through a ``Weights`` object
so the *same* op schedule can be driven by either:

  * ``HFWeights``    -- exact fp32 weights from the Hugging Face model. Used to
    prove the architecture math matches ``GPTNeoForCausalLM`` (no quant error).
  * ``QuantWeights`` -- the exported INT8 bins, with each linear evaluated by the
    W8A8 GEMV datapath (numpy reference, the C++ kernel datapath, or -- on the
    board -- the FPGA). This is what actually maps onto hardware.

GPT-Neo specifics that this mirrors from the HF implementation:
  * learned absolute position embeddings (``wpe``) added at the input;
  * standard LayerNorm (weight + bias), eps from config;
  * **no 1/sqrt(head_dim) attention scaling** (a GPT-Neo trait);
  * ``gelu_new`` (tanh approximation) MLP activation;
  * attention alternates global / local(window) per layer;
  * ``lm_head`` tied to the token embedding.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tinystories_z2.quant import (
    QuantizedWeightInt8,
    dequantize_weight,
    gemv_int8_quantized,
)


# --------------------------------------------------------------------------- #
# arch description
# --------------------------------------------------------------------------- #
@dataclass
class NeoArch:
    hidden_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    intermediate_size: int
    layer_norm_epsilon: float
    window_size: int
    attention_types: list[str]  # per-layer "global" | "local", length num_layers
    tie_word_embeddings: bool

    @staticmethod
    def from_hf_config(cfg) -> "NeoArch":
        hidden = cfg.hidden_size
        heads = cfg.num_heads
        inter = cfg.intermediate_size or 4 * hidden
        # attention_types is e.g. [[['global','local'], 4]] -> flatten to per-layer
        types: list[str] = []
        for block, count in cfg.attention_types:
            for _ in range(count):
                types.extend(block)
        types = types[: cfg.num_layers]
        return NeoArch(
            hidden_size=hidden,
            num_layers=cfg.num_layers,
            num_heads=heads,
            head_dim=hidden // heads,
            vocab_size=cfg.vocab_size,
            max_position_embeddings=cfg.max_position_embeddings,
            intermediate_size=inter,
            layer_norm_epsilon=float(cfg.layer_norm_epsilon),
            window_size=int(getattr(cfg, "window_size", 256)),
            attention_types=types,
            tie_word_embeddings=bool(getattr(cfg, "tie_word_embeddings", True)),
        )

    @staticmethod
    def from_manifest(man: dict) -> "NeoArch":
        a = man["arch"]
        return NeoArch(
            hidden_size=a["hidden_size"],
            num_layers=a["num_layers"],
            num_heads=a["num_heads"],
            head_dim=a["head_dim"],
            vocab_size=a["vocab_size"],
            max_position_embeddings=a["max_position_embeddings"],
            intermediate_size=a["intermediate_size"],
            layer_norm_epsilon=a["layer_norm_epsilon"],
            window_size=a["window_size"],
            attention_types=a["attention_types"],
            tie_word_embeddings=a["tie_word_embeddings"],
        )


# --------------------------------------------------------------------------- #
# glue math (runs on the ARM core on-board)
# --------------------------------------------------------------------------- #
def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    x = x.astype(np.float32)
    mu = x.mean()
    var = x.var()  # biased (ddof=0), matches torch LayerNorm
    return (x - mu) / np.sqrt(var + eps) * w + b


def gelu_new(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


# --------------------------------------------------------------------------- #
# weight providers
# --------------------------------------------------------------------------- #
class Weights:
    """Provides per-name vectors (fp32) and linear matvecs for the runner."""

    def vec(self, key: str) -> np.ndarray:
        raise NotImplementedError

    def has(self, key: str) -> bool:
        raise NotImplementedError

    def linear(self, key: str, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class HFWeights(Weights):
    """Exact fp32 weights from a loaded ``GPTNeoForCausalLM`` (no quantization).

    The architecture-correctness ground truth: a runner on top of this must match
    HF's own forward pass to fp32 tolerance.
    """

    def __init__(self, model) -> None:
        import torch  # noqa: F401

        sd = model.state_dict()
        self._v: dict[str, np.ndarray] = {}
        self._m: dict[str, np.ndarray] = {}

        def to_np(t):
            return t.detach().to(dtype=__import__("torch").float32).cpu().numpy()

        self._v["wte"] = to_np(sd["transformer.wte.weight"])
        self._v["wpe"] = to_np(sd["transformer.wpe.weight"])
        self._v["ln_f.w"] = to_np(sd["transformer.ln_f.weight"])
        self._v["ln_f.b"] = to_np(sd["transformer.ln_f.bias"])
        # lm_head (tied -> equals wte)
        self._m["lm_head"] = to_np(sd["lm_head.weight"])

        n_layers = model.config.num_layers
        for i in range(n_layers):
            p = f"transformer.h.{i}."
            self._v[f"L{i}.ln_1.w"] = to_np(sd[p + "ln_1.weight"])
            self._v[f"L{i}.ln_1.b"] = to_np(sd[p + "ln_1.bias"])
            self._v[f"L{i}.ln_2.w"] = to_np(sd[p + "ln_2.weight"])
            self._v[f"L{i}.ln_2.b"] = to_np(sd[p + "ln_2.bias"])
            a = p + "attn.attention."
            self._m[f"L{i}.q"] = to_np(sd[a + "q_proj.weight"])
            self._m[f"L{i}.k"] = to_np(sd[a + "k_proj.weight"])
            self._m[f"L{i}.v"] = to_np(sd[a + "v_proj.weight"])
            self._m[f"L{i}.o"] = to_np(sd[a + "out_proj.weight"])
            self._v[f"L{i}.o.bias"] = to_np(sd[a + "out_proj.bias"])
            self._m[f"L{i}.fc"] = to_np(sd[p + "mlp.c_fc.weight"])
            self._v[f"L{i}.fc.bias"] = to_np(sd[p + "mlp.c_fc.bias"])
            self._m[f"L{i}.proj"] = to_np(sd[p + "mlp.c_proj.weight"])
            self._v[f"L{i}.proj.bias"] = to_np(sd[p + "mlp.c_proj.bias"])

    def has(self, key: str) -> bool:
        return key in self._v

    def vec(self, key: str) -> np.ndarray:
        return self._v[key]

    def linear(self, key: str, x: np.ndarray) -> np.ndarray:
        return (self._m[key] @ x.astype(np.float32)).astype(np.float32)


class QuantWeights(Weights):
    """The exported INT4 layout; linears go through the INT4 GEMV datapath.

    ``backend`` (a ``qwen_fpga.host.fpga_gemv.GemvBackend``) selects where the
    GEMV runs: ``None`` uses the numpy quantized reference; ``cpp`` runs the exact
    C++ kernel datapath; ``pynq``/``xrt`` run it on the FPGA. ``dequant=True``
    instead does a plain fp32 matmul on the *dequantized* weights -- isolating
    weight-quantization error from activation-quantization error.
    """

    def __init__(self, manifest_path: str | Path, backend=None, dequant: bool = False):
        manifest_path = Path(manifest_path)
        self.man = json.loads(manifest_path.read_text())
        self.root = manifest_path.parent
        self.backend = backend
        self.dequant = dequant
        self._vcache: dict[str, np.ndarray] = {}
        self._qcache: dict[str, QuantizedWeightInt8] = {}

    # name -> on-disk spec for a linear
    def _spec(self, key: str) -> dict:
        if key == "lm_head":
            return self.man["tensors"]["lm_head"]
        # key like "L3.q"
        li, sub = key[1:].split(".", 1)
        L = self.man["layers"][int(li)]
        return (L["attn"] if sub in ("q", "k", "v", "o") else L["mlp"])[sub]

    def _qweight(self, key: str) -> QuantizedWeightInt8:
        if key in self._qcache:
            return self._qcache[key]
        spec = self._spec(key)
        m, n = spec["out_features"], spec["in_features"]
        q = np.fromfile(self.root / spec["weight"], dtype=np.int8).reshape(m, n)
        scales = np.fromfile(self.root / spec["scale"], dtype=np.float16).reshape(m)
        qw = QuantizedWeightInt8(q=q, scales=scales, out_features=m, in_features=n)
        self._qcache[key] = qw
        return qw

    def has(self, key: str) -> bool:
        return key in self._flat_vecs()

    def _flat_vecs(self) -> dict[str, dict]:
        return self.man["vectors"]

    def vec(self, key: str) -> np.ndarray:
        if key in self._vcache:
            return self._vcache[key]
        spec = self._flat_vecs()[key]
        a = np.fromfile(self.root / spec["file"], dtype=np.float16).astype(np.float32)
        shape = tuple(spec["shape"])
        if len(shape) > 1:
            a = a.reshape(shape)  # wte [vocab,h], wpe [max_pos,h] must stay 2-D
        self._vcache[key] = a
        return a

    def linear(self, key: str, x: np.ndarray) -> np.ndarray:
        qw = self._qweight(key)
        x = x.astype(np.float32)
        if self.dequant:
            w = dequantize_weight(qw)[:, : x.shape[0]]
            return (w @ x).astype(np.float32)
        if self.backend is not None:
            return self.backend.gemv(qw, x)
        return gemv_int8_quantized(qw, x)


# --------------------------------------------------------------------------- #
# the decode runner
# --------------------------------------------------------------------------- #
class NeoRunner:
    """Autoregressive GPT-Neo decode over a ``Weights`` provider, with KV cache."""

    def __init__(self, arch: NeoArch, weights: Weights):
        self.a = arch
        self.w = weights
        self.reset()

    def reset(self) -> None:
        self.kcache = [[] for _ in range(self.a.num_layers)]
        self.vcache = [[] for _ in range(self.a.num_layers)]

    def _attention(self, li: int, q: np.ndarray, pos: int) -> np.ndarray:
        a = self.a
        K = np.stack(self.kcache[li], axis=0)  # [T, n_heads, head_dim]
        V = np.stack(self.vcache[li], axis=0)
        T = K.shape[0]
        # local layers only attend within the trailing window
        if a.attention_types[li] == "local":
            lo = max(0, pos - a.window_size + 1)
        else:
            lo = 0
        out = np.empty((a.num_heads, a.head_dim), dtype=np.float32)
        for h in range(a.num_heads):
            scores = K[lo:T, h, :] @ q[h]          # [t] -- NO 1/sqrt(d) scaling
            scores = scores - scores.max()
            ew = np.exp(scores)
            ew /= ew.sum()
            out[h] = ew @ V[lo:T, h, :]
        return out.reshape(-1)

    def layer(self, li: int, x: np.ndarray, pos: int) -> np.ndarray:
        a = self.a
        residual = x
        h = layer_norm(x, self.w.vec(f"L{li}.ln_1.w"), self.w.vec(f"L{li}.ln_1.b"),
                       a.layer_norm_epsilon)
        q = self.w.linear(f"L{li}.q", h).reshape(a.num_heads, a.head_dim)
        k = self.w.linear(f"L{li}.k", h).reshape(a.num_heads, a.head_dim)
        v = self.w.linear(f"L{li}.v", h).reshape(a.num_heads, a.head_dim)
        self.kcache[li].append(k)
        self.vcache[li].append(v)
        attn = self._attention(li, q, pos)
        o = self.w.linear(f"L{li}.o", attn) + self.w.vec(f"L{li}.o.bias")
        x = residual + o

        residual = x
        h = layer_norm(x, self.w.vec(f"L{li}.ln_2.w"), self.w.vec(f"L{li}.ln_2.b"),
                       a.layer_norm_epsilon)
        f = self.w.linear(f"L{li}.fc", h) + self.w.vec(f"L{li}.fc.bias")
        f = gelu_new(f)
        d = self.w.linear(f"L{li}.proj", f) + self.w.vec(f"L{li}.proj.bias")
        return residual + d

    def forward_token(self, token_id: int, pos: int) -> np.ndarray:
        a = self.a
        x = (self.w.vec("wte")[token_id] + self.w.vec("wpe")[pos]).astype(np.float32)
        for li in range(a.num_layers):
            x = self.layer(li, x, pos)
        x = layer_norm(x, self.w.vec("ln_f.w"), self.w.vec("ln_f.b"),
                       a.layer_norm_epsilon)
        return self.w.linear("lm_head", x)  # logits [vocab]

    def generate(self, prompt_ids: list[int], max_new: int = 32,
                 temperature: float = 0.0, top_k: int = 40, seed: int = 0) -> list[int]:
        self.reset()
        rng = np.random.default_rng(seed)
        pos = 0
        logits = None
        for t in prompt_ids:
            logits = self.forward_token(int(t), pos)
            pos += 1
        out: list[int] = []
        for _ in range(max_new):
            if temperature <= 0.0:
                nxt = int(np.argmax(logits))
            else:
                lg = logits / temperature
                if top_k:
                    idx = np.argpartition(lg, -top_k)[-top_k:]
                    masked = np.full_like(lg, -np.inf)
                    masked[idx] = lg[idx]
                    lg = masked
                lg = lg - lg.max()
                p = np.exp(lg)
                p /= p.sum()
                nxt = int(rng.choice(len(p), p=p))
            out.append(nxt)
            logits = self.forward_token(nxt, pos)
            pos += 1
        return out
