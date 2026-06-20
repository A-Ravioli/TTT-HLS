"""LoRA configuration and application for the GLM generator.

Lazy-imports ``peft``/``torch`` so this module is importable without a GPU. When
those are present, :func:`apply_lora` wraps a HuggingFace causal-LM with trainable
LoRA adapters targeting the attention/MLP projections common to GLM/Qwen-style
decoders.
"""

from __future__ import annotations

from typing import Any

from paths import get_logger

logger = get_logger("burnttt.glm.lora")

# Projection module names common to GLM / Qwen2 / Llama-style decoders.
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    # GLM fused variants:
    "query_key_value",
    "dense",
    "dense_h_to_4h",
    "dense_4h_to_h",
]


def peft_available() -> bool:
    try:
        import peft  # noqa: F401
        import torch  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def default_lora_config(
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> Any:
    """Build a ``peft.LoraConfig`` (requires peft)."""
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules or DEFAULT_TARGET_MODULES,
        bias="none",
    )


def apply_lora(model: Any, config: Any | None = None, use_unsloth: bool = False) -> Any:
    """Return ``model`` wrapped with LoRA adapters (only the adapters train)."""
    if use_unsloth:
        from unsloth import FastLanguageModel

        cfg = config
        r = getattr(cfg, "r", 16) if cfg is not None else 16
        alpha = getattr(cfg, "lora_alpha", 32) if cfg is not None else 32
        dropout = getattr(cfg, "lora_dropout", 0.05) if cfg is not None else 0.05
        peft_model = FastLanguageModel.get_peft_model(
            model,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "gate_up_proj",
            ],
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        try:
            peft_model.print_trainable_parameters()
        except Exception:  # noqa: BLE001
            pass
        return peft_model

    from peft import get_peft_model

    config = config or default_lora_config()
    peft_model = get_peft_model(model, config)
    try:
        peft_model.print_trainable_parameters()
    except Exception:  # noqa: BLE001
        pass
    return peft_model
