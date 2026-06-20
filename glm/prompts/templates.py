"""The text the GLM generator sees.

The prompt gives the model (1) the task (block + part + budget), (2) the legal
config schema, and (3) the feedback from every config tried so far on this exact
task. The model returns JSON hardware configs. This is the interface across which
test-time adaptation happens: the same prompt structure is used whether the model
is a frozen LLM, a LoRA-adapted LLM, or the heuristic fallback backend.
"""

from __future__ import annotations

import json
from typing import Any

from ttt.config_space import BITWIDTHS, INT_BITS, REUSE_FACTORS, STRATEGIES

SYSTEM_PROMPT = (
    "You are GLM, a test-time-adaptive compiler that maps neural-network blocks "
    "onto FPGA fabric via hls4ml. You author the hardware-generation config "
    "(quantization precision, per-layer reuse factors, and HLS strategy) for a "
    "specific model block and a specific FPGA part. You are given real feedback "
    "from bit-accurate simulation and synthesis/resource estimation for every "
    "config tried so far on THIS task, and you must propose configs that fit the "
    "board, keep output error below threshold, and minimize latency and resource "
    "use. Reason about the accuracy/resource tradeoff, then emit ONLY JSON."
)

_SCHEMA = {
    "weight_bits": f"int in {BITWIDTHS}",
    "activation_bits": f"int in {BITWIDTHS}",
    "int_bits": f"int in {INT_BITS}, MUST be < min(weight_bits, activation_bits)",
    "reuse_dense_1": f"int in {REUSE_FACTORS}",
    "reuse_dense_2": f"int in {REUSE_FACTORS}",
    "strategy": f"one of {STRATEGIES}",
    "rationale": "one short sentence (optional)",
}


def _schema_block() -> str:
    return (
        "Config JSON schema (all fields required except rationale):\n"
        + json.dumps(_SCHEMA, indent=2)
        + "\n\nTradeoff hints:\n"
        "- Lower bitwidth -> less fabric but more quantization error.\n"
        "- Fewer int_bits -> more fractional bits -> lower error (until integer overflow).\n"
        "- Higher reuse factor -> fewer DSPs, higher latency (Resource strategy).\n"
        "- Lower reuse factor -> more parallelism, lower latency, more DSPs (Latency strategy).\n"
    )


def format_history(history: list[dict[str, Any]], max_rows: int = 24) -> str:
    """Render prior (config -> feedback) attempts for this task as compact text."""
    if not history:
        return "No configs have been tried yet on this task."
    rows = history[-max_rows:]
    lines = ["Configs tried so far on THIS task (most recent last):"]
    for h in rows:
        cfg = h.get("config", {})
        lines.append(
            "  cfg={} | compile={} | max_error={} | latency={} | dsp={} | lut={} | "
            "fits_board={} | reward={}".format(
                _short(cfg),
                h.get("compile_success"),
                _fmt(h.get("max_error")),
                _fmt(h.get("latency_cycles")),
                _fmt(h.get("dsp")),
                _fmt(h.get("lut")),
                h.get("fits_board"),
                _fmt(h.get("reward")),
            )
        )
        if h.get("error_msg"):
            lines.append(f"      compile_error: {str(h['error_msg'])[:160]}")
    return "\n".join(lines)


def build_propose_prompt(task_desc: str, history: list[dict[str, Any]], n: int) -> str:
    """Prompt asking for ``n`` new, diverse, better configs as a JSON list."""
    return (
        f"{task_desc}\n\n"
        f"{_schema_block()}\n"
        f"{format_history(history)}\n\n"
        f"Propose {n} NEW config(s) that you predict will beat the best so far. "
        f"Favor configs that fit the budget with low latency and error. Do not "
        f"repeat configs already tried. "
        f"Return ONLY a JSON array of {n} config object(s), nothing else."
    )


def build_repair_prompt(task_desc: str, failed_config: dict[str, Any], error_msg: str) -> str:
    """Prompt asking the model to fix a config that failed to compile."""
    return (
        f"{task_desc}\n\n"
        f"{_schema_block()}\n"
        f"This config FAILED to compile:\n{json.dumps(failed_config, indent=2)}\n\n"
        f"Compiler/conversion error:\n{str(error_msg)[:400]}\n\n"
        f"Diagnose the likely cause (precision/overflow, reuse divisibility, or "
        f"strategy) and return ONE corrected config as a single JSON object. "
        f"Return ONLY the JSON object, nothing else."
    )


def _short(cfg: dict[str, Any]) -> str:
    try:
        return (
            f"w{cfg.get('weight_bits')}a{cfg.get('activation_bits')}"
            f"i{cfg.get('int_bits')}_r{cfg.get('reuse_dense_1')}-{cfg.get('reuse_dense_2')}"
            f"_{str(cfg.get('strategy'))[:3]}"
        )
    except Exception:  # noqa: BLE001
        return str(cfg)


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)
