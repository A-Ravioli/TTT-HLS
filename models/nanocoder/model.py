"""NanoCoder architecture: byte tokenizer + GPT-Neo config/model builders.

Everything here is intentionally tiny and hardware-aware (see the package
docstring): vocab=256 byte-level, hidden=128, ReLU MLP. The config is a standard
``transformers.GPTNeoConfig`` so we get a real, trainable language model with a
known-good training path -- the FPGA constraints live entirely in the *choice* of
hyper-parameters, not in a custom model class.
"""

from __future__ import annotations

from dataclasses import dataclass

from paths import get_logger

logger = get_logger("burnttt.nanocoder.model")


@dataclass(frozen=True)
class NanoCoderArch:
    """The hardware-pinned NanoCoder dimensions.

    v2: BPE vocab (~4096) + hidden 256. The BPE vocab is the coherence win (params go
    to logic, not byte-spelling) and costs nothing on the FPGA -- the hardened MLP
    block is vocab-independent. hidden 256 still fits the PYNQ-Z2 (MLP 524,288 MACs ->
    ReuseFactor 2,384 on 220 DSP). (v1 was hidden 128 / vocab 256 byte-level.)
    """

    hidden_size: int = 256
    intermediate_size: int = 1024
    num_layers: int = 8
    num_heads: int = 16
    vocab_size: int = 4096         # BPE: token-level, params go to logic not spelling
    max_position_embeddings: int = 512
    activation: str = "relu"       # hls4ml-hardenable (GELU is not)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def mlp_macs(self) -> int:
        """MACs in one decoder-layer MLP (c_fc + c_proj)."""
        return 2 * self.hidden_size * self.intermediate_size

    def reuse_to_fit(self, dsps: int = 220) -> int:
        """Minimum ReuseFactor for the MLP block to fit ``dsps`` DSPs."""
        return -(-self.mlp_macs // dsps)  # ceil division


DEFAULT_ARCH = NanoCoderArch()


class ByteTokenizer:
    """Raw-byte tokenizer: vocab is exactly 256, one token per UTF-8 byte.

    No training, no merges, no OOV. ``\\n`` (byte 10) doubles as the document
    separator during training; there are no other special tokens, so the vocab
    stays at exactly 256 and the embedding table is a trivial 256 x hidden.
    """

    vocab_size = 256
    sep_id = 10  # newline, used to join documents

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8", errors="ignore"))

    def decode(self, ids) -> str:
        return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")


def load_tokenizer(ckpt_dir=None):
    """Return the NanoCoder tokenizer: BPE (tokenizer.json) if present, else byte-level.

    Both expose the same minimal interface used by serving: ``.encode(str) -> list[int]``
    and ``.decode(list[int]) -> str``, plus an ``eos_id`` attribute.
    """
    from pathlib import Path

    from paths import ARTIFACTS_DIR

    d = Path(ckpt_dir) if ckpt_dir else (ARTIFACTS_DIR / "nanocoder")
    tok_json = d / "tokenizer.json"
    if tok_json.exists():
        from transformers import PreTrainedTokenizerFast

        tk = PreTrainedTokenizerFast(
            tokenizer_file=str(tok_json), eos_token="<|endoftext|>", bos_token="<|endoftext|>",
            pad_token="<|endoftext|>", unk_token="<|endoftext|>",
        )
        tk.eos_id = tk.eos_token_id
        logger.info("Loaded NanoCoder BPE tokenizer (vocab=%d) from %s", tk.vocab_size, tok_json)
        return tk
    tb = ByteTokenizer()
    tb.eos_id = ByteTokenizer.sep_id
    return tb


def build_config(arch: NanoCoderArch = DEFAULT_ARCH):
    """Return a ``transformers.GPTNeoConfig`` for NanoCoder."""
    from transformers import GPTNeoConfig

    # GPT-Neo alternates global/local attention; one (global, local) pair repeated.
    n_pairs = max(1, arch.num_layers // 2)
    return GPTNeoConfig(
        vocab_size=arch.vocab_size,
        hidden_size=arch.hidden_size,
        num_layers=arch.num_layers,
        num_heads=arch.num_heads,
        intermediate_size=arch.intermediate_size,
        max_position_embeddings=arch.max_position_embeddings,
        activation_function=arch.activation,
        window_size=256,
        attention_types=[[["global", "local"], n_pairs]],
        resid_dropout=0.0,
        embed_dropout=0.0,
        attention_dropout=0.0,
        classifier_dropout=0.0,
        bos_token_id=0,
        eos_token_id=ByteTokenizer.sep_id,  # newline (byte 10) doubles as EOS/separator
    )


def build_model(arch: NanoCoderArch = DEFAULT_ARCH):
    """Build a fresh (untrained) NanoCoder ``GPTNeoForCausalLM``."""
    from transformers import GPTNeoForCausalLM

    cfg = build_config(arch)
    model = GPTNeoForCausalLM(cfg)
    n = sum(p.numel() for p in model.parameters())
    logger.info(
        "Built NanoCoder: %s params (hidden=%d inter=%d L=%d H=%d vocab=%d, %s MLP)",
        f"{n:,}", arch.hidden_size, arch.intermediate_size, arch.num_layers,
        arch.num_heads, arch.vocab_size, arch.activation,
    )
    return model
