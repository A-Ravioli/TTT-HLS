"""Load a Qwen2 model / its architecture config.

The decomposition only needs the architecture dimensions (hidden size,
intermediate size, head counts), so this works even without downloading weights:
it tries ``transformers.AutoConfig`` first and falls back to a small built-in spec
table for common Qwen2 sizes. Weights are only loaded when actually exporting a
block for compilation.
"""

from __future__ import annotations

from dataclasses import dataclass

from paths import get_logger

logger = get_logger("burnttt.qwen.load")

DEFAULT_MODEL_ID = "Qwen/Qwen2-1.5B"

# Built-in architecture specs (used when transformers is unavailable / offline).
QWEN_SPECS: dict[str, dict[str, int]] = {
    "Qwen/Qwen2-0.5B": dict(
        hidden_size=896, intermediate_size=4864, num_hidden_layers=24,
        num_attention_heads=14, num_key_value_heads=2, head_dim=64, vocab_size=151936,
    ),
    "Qwen/Qwen2-1.5B": dict(
        hidden_size=1536, intermediate_size=8960, num_hidden_layers=28,
        num_attention_heads=12, num_key_value_heads=2, head_dim=128, vocab_size=151936,
    ),
    "Qwen/Qwen2.5-1.5B": dict(
        hidden_size=1536, intermediate_size=8960, num_hidden_layers=28,
        num_attention_heads=12, num_key_value_heads=2, head_dim=128, vocab_size=151936,
    ),
}


@dataclass(frozen=True)
class QwenArch:
    """Architecture dimensions needed to decompose a Qwen2 model into blocks."""

    model_id: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int = 151936

    @property
    def q_out(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_out(self) -> int:
        return self.num_key_value_heads * self.head_dim


def load_qwen_arch(model_id: str = DEFAULT_MODEL_ID) -> QwenArch:
    """Return the :class:`QwenArch` for ``model_id`` (transformers if present, else table)."""
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        logger.info("Loaded Qwen arch from transformers AutoConfig: %s", model_id)
        return QwenArch(
            model_id=model_id,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            num_hidden_layers=cfg.num_hidden_layers,
            num_attention_heads=cfg.num_attention_heads,
            num_key_value_heads=getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
            head_dim=head_dim,
            vocab_size=getattr(cfg, "vocab_size", 151936),
        )
    except Exception as exc:  # noqa: BLE001
        if model_id not in QWEN_SPECS:
            raise ValueError(
                f"Could not load arch for {model_id!r} via transformers ({exc}) and it is "
                f"not in the built-in spec table {list(QWEN_SPECS)}."
            ) from exc
        logger.info("Using built-in Qwen arch spec (transformers unavailable): %s", model_id)
        return QwenArch(model_id=model_id, **QWEN_SPECS[model_id])


def get_default_arch() -> QwenArch:
    """Convenience wrapper: the :class:`QwenArch` for :data:`DEFAULT_MODEL_ID`.

    The Phase-4 HLS scripts (``10``--``14``) import this; keep it as a thin alias
    over :func:`load_qwen_arch` so they share one source of truth for the default.
    """
    return load_qwen_arch(DEFAULT_MODEL_ID)


def load_qwen_model(model_id: str = DEFAULT_MODEL_ID):
    """Load the actual Qwen2 weights (requires transformers + torch)."""
    from transformers import AutoModelForCausalLM

    logger.info("Loading Qwen weights: %s", model_id)
    return AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype="auto")
