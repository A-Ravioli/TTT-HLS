"""Parse GLM text output into validated :class:`BurnConfig` objects.

LLMs emit messy text. These helpers robustly pull JSON config objects out of a
model response and coerce them into the legal config space (clamping illegal
``int_bits`` rather than discarding an otherwise-good config), so a single
formatting slip doesn't waste an evaluation.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ttt.config_space import (
    BITWIDTHS,
    INT_BITS,
    REUSE_FACTORS,
    STRATEGIES,
    BurnConfig,
)

_JSON_OBJ = re.compile(r"\{[^{}]*\}", re.DOTALL)


def extract_json_dicts(text: str) -> list[dict[str, Any]]:
    """Pull every top-level JSON object out of ``text`` (arrays or loose objects)."""
    text = text.strip()
    # First, try to parse the whole thing (a clean array or object).
    for candidate in (text, _strip_code_fence(text)):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, list):
                return [d for d in obj if isinstance(d, dict)]
            if isinstance(obj, dict):
                return [obj]
        except (json.JSONDecodeError, TypeError):
            pass
    # Fall back to regex-scraping individual objects.
    out: list[dict[str, Any]] = []
    for m in _JSON_OBJ.finditer(text):
        try:
            d = json.loads(m.group(0))
            if isinstance(d, dict):
                out.append(d)
        except json.JSONDecodeError:
            continue
    return out


def _strip_code_fence(text: str) -> str:
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return fence.group(1) if fence else text


def _nearest(value: Any, choices: list[int], default: int) -> int:
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return min(choices, key=lambda c: abs(c - v))


def dict_to_config(d: dict[str, Any]) -> BurnConfig | None:
    """Coerce a (possibly imperfect) dict into a valid :class:`BurnConfig`.

    Returns ``None`` only if the dict is unusable. Otherwise snaps each field to
    the nearest legal value and guarantees ``int_bits < min(bitwidths)``.
    """
    if not isinstance(d, dict):
        return None
    try:
        wb = _nearest(d.get("weight_bits", d.get("bits", 16)), BITWIDTHS, 16)
        ab = _nearest(d.get("activation_bits", d.get("bits", wb)), BITWIDTHS, wb)
        max_int = min(wb, ab) - 1
        ib_choices = [b for b in INT_BITS if b <= max_int] or [min(max_int, INT_BITS[0])]
        ib = _nearest(d.get("int_bits", min(ib_choices)), ib_choices, min(ib_choices))
        ib = min(ib, max_int)
        if ib < 1:
            ib = 1
        r1 = _nearest(d.get("reuse_dense_1", d.get("reuse", 1)), REUSE_FACTORS, 1)
        r2 = _nearest(d.get("reuse_dense_2", d.get("reuse", r1)), REUSE_FACTORS, r1)
        strat = str(d.get("strategy", "Latency")).strip().capitalize()
        if strat not in STRATEGIES:
            strat = "Latency"
        return BurnConfig(wb, ab, ib, r1, r2, strat)
    except (ValueError, TypeError):
        return None


def parse_configs(text: str) -> list[BurnConfig]:
    """Extract all valid configs from a model response, in order, de-duplicated."""
    configs: list[BurnConfig] = []
    seen: set[str] = set()
    for d in extract_json_dicts(text):
        cfg = dict_to_config(d)
        if cfg is not None and cfg.short_name() not in seen:
            seen.add(cfg.short_name())
            configs.append(cfg)
    return configs
