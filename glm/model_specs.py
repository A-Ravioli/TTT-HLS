"""Hardware + loader hints for HuggingFace GLM checkpoints used in LoRA TTT."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPodSpec:
    gpu_count: int
    gpu_type: str
    use_unsloth: bool
    notes: str


# z.ai GLM-5.2: 744B MoE / 40B active. Official min inference: 8×H100 FP8.
# Prime Intellect currently tops out at 8×A100_80GB (~$22/hr) — best available there.
MODEL_POD_SPECS: dict[str, ModelPodSpec] = {
    "zai-org/GLM-5.2-FP8": ModelPodSpec(
        gpu_count=8,
        gpu_type="A100_80GB",
        use_unsloth=True,
        notes="744B MoE FP8; needs 8 GPUs. Prefer 8×H100 when available.",
    ),
    "zai-org/GLM-5.2": ModelPodSpec(
        gpu_count=8,
        gpu_type="A100_80GB",
        use_unsloth=True,
        notes="744B MoE BF16; needs 8×H100/H200 for practical load.",
    ),
    "zai-org/GLM-5.1-FP8": ModelPodSpec(8, "A100_80GB", True, "744B MoE FP8"),
    "zai-org/GLM-5-FP8": ModelPodSpec(8, "A100_80GB", True, "744B MoE FP8"),
    "THUDM/glm-4-9b-chat-hf": ModelPodSpec(1, "H100_80GB", False, "9B dense GlmForCausalLM"),
    "zai-org/GLM-4.7-Flash": ModelPodSpec(1, "H100_80GB", True, "~31B MoE-lite; GRPO LoRA on 1×H100"),
    "zai-org/GLM-4-32B-0414": ModelPodSpec(1, "H100_80GB", False, "~33B dense; tight on 80GB"),
    "Qwen/Qwen2.5-7B-Instruct": ModelPodSpec(1, "H100_80GB", False, "7B dense fallback"),
}

DEFAULT_POD_GLM_MODEL = "zai-org/GLM-5.2-FP8"


def pod_spec_for_model(model_id: str) -> ModelPodSpec:
    mid = model_id.strip()
    if mid in MODEL_POD_SPECS:
        return MODEL_POD_SPECS[mid]
    lower = mid.lower()
    if "glm-5" in lower or "glm_5" in lower:
        return MODEL_POD_SPECS["zai-org/GLM-5.2-FP8"]
    if "glm-4-9b" in lower:
        return MODEL_POD_SPECS["THUDM/glm-4-9b-chat-hf"]
    return ModelPodSpec(1, "H100_80GB", False, "default single-GPU dense")
