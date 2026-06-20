"""Extract and validate HLS C++/H source files from GLM model output.

The GLM compiler agent emits code in markdown code fences. This module:
1. Extracts .cpp/.h content from fenced blocks.
2. Runs static checks (forbidden APIs, required interface pragmas).
3. Returns a validated dict of {filename: source_content}.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from paths import get_logger

logger = get_logger("burnttt.glm.parsing.hls")

# Patterns that indicate unsynthesizable code
FORBIDDEN_PATTERNS = [
    (r"\bnew\s+\w+", "dynamic allocation (new)"),
    (r"\bdelete\b", "dynamic deallocation (delete)"),
    (r"\bmalloc\s*\(", "malloc"),
    (r"\bfree\s*\(", "free"),
    (r"\bvirtual\b", "virtual functions"),
    (r"\bthrow\b", "exceptions (throw)"),
    (r"\btry\s*\{", "exceptions (try)"),
    (r"\bstd::vector\b", "std::vector (not synthesizable)"),
    (r"\bstd::string\b", "std::string (not synthesizable)"),
    (r"\bstd::map\b", "std::map (not synthesizable)"),
]

# Required patterns for a valid HLS kernel
REQUIRED_PATTERNS = [
    (r"#pragma\s+HLS\s+INTERFACE", "HLS INTERFACE pragma"),
    (r"void\s+kernel_top\s*\(", "kernel_top function definition"),
]


@dataclass
class HLSParseResult:
    """Result of parsing GLM output for HLS sources."""

    success: bool
    sources: dict[str, str]  # filename -> content
    warnings: list[str]
    errors: list[str]


def parse_hls_from_text(text: str) -> HLSParseResult:
    """Extract HLS source files from GLM output text.

    Looks for code fences with language markers (cpp, c, h) and infers
    filenames from comments or fence labels.
    """
    sources: dict[str, str] = {}
    warnings: list[str] = []
    errors: list[str] = []

    # Extract code blocks
    blocks = _extract_code_blocks(text)

    if not blocks:
        errors.append("No code blocks found in output")
        return HLSParseResult(success=False, sources={}, warnings=warnings, errors=errors)

    # Assign filenames based on content analysis
    for i, (lang, content) in enumerate(blocks):
        filename = _infer_filename(content, lang, i)
        sources[filename] = content

    # Validate
    all_content = "\n".join(sources.values())
    warnings.extend(_check_forbidden(all_content))
    missing = _check_required(all_content)
    if missing:
        # Only warn; the source might still compile
        warnings.extend(missing)

    success = len(sources) > 0 and len(errors) == 0
    if warnings:
        logger.debug("HLS parse warnings: %s", warnings)
    return HLSParseResult(success=success, sources=sources, warnings=warnings, errors=errors)


def validate_kernel_bundle(sources: dict[str, str]) -> list[str]:
    """Static validation of a kernel bundle's sources. Returns list of issues."""
    issues: list[str] = []
    all_content = "\n".join(sources.values())

    # Check for forbidden patterns
    for pattern, desc in FORBIDDEN_PATTERNS:
        if re.search(pattern, all_content):
            issues.append(f"Contains unsynthesizable construct: {desc}")

    # Check for kernel_top definition
    if not re.search(r"void\s+kernel_top\s*\(", all_content):
        issues.append("Missing kernel_top function definition")

    # Check header exists
    has_header = any(f.endswith(".h") for f in sources)
    if not has_header:
        issues.append("No header file (.h) found")

    # Check cpp exists
    has_cpp = any(f.endswith(".cpp") for f in sources)
    if not has_cpp:
        issues.append("No implementation file (.cpp) found")

    return issues


def _extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """Extract (language, content) pairs from markdown code fences."""
    # Match ```lang ... ``` blocks
    pattern = r"```(\w*)\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    blocks: list[tuple[str, str]] = []
    for lang, content in matches:
        content = content.strip()
        if not content:
            continue
        # Normalize language
        lang = lang.lower()
        if lang in ("cpp", "c++", "c", "h", "hpp", "hls"):
            blocks.append((lang, content))
        elif not lang and _looks_like_cpp(content):
            blocks.append(("cpp", content))

    # Fallback: if no fenced blocks, try to extract raw C++ from the text
    if not blocks:
        # Look for #include or void kernel_top as indicators
        if "#include" in text or "void kernel_top" in text:
            # Try to extract everything that looks like code
            code_start = text.find("#")
            if code_start >= 0:
                blocks.append(("cpp", text[code_start:].strip()))

    return blocks


def _infer_filename(content: str, lang: str, index: int) -> str:
    """Infer a filename from the content or fence language."""
    # Look for filename comments: // filename: kernel_top.h
    match = re.search(r"//\s*(?:file(?:name)?|File)\s*:\s*(\S+)", content)
    if match:
        return match.group(1)

    # Check if it's a header (has #ifndef guard or only declarations)
    if re.search(r"#ifndef\s+\w+_H", content) or lang in ("h", "hpp"):
        return "kernel_top.h"

    # Check for #include "kernel_top.h" -> it's the .cpp
    if '#include "kernel_top.h"' in content:
        return "kernel_top.cpp"

    # Default based on index
    if index == 0 and _looks_like_header(content):
        return "kernel_top.h"
    return "kernel_top.cpp"


def _looks_like_cpp(text: str) -> bool:
    """Heuristic: does this text look like C++ code?"""
    indicators = ["#include", "#pragma", "void ", "int ", "float ", "ap_fixed"]
    return any(ind in text for ind in indicators)


def _looks_like_header(content: str) -> bool:
    """Heuristic: does this look like a header file?"""
    return "#ifndef" in content or (
        "#define" in content and "void kernel_top" not in content.split("{")[0]
        if "{" in content else True
    )


def _check_forbidden(content: str) -> list[str]:
    """Check for forbidden/unsynthesizable patterns."""
    warnings = []
    for pattern, desc in FORBIDDEN_PATTERNS:
        if re.search(pattern, content):
            warnings.append(f"Potentially unsynthesizable: {desc}")
    return warnings


def _check_required(content: str) -> list[str]:
    """Check that required patterns are present."""
    missing = []
    for pattern, desc in REQUIRED_PATTERNS:
        if not re.search(pattern, content):
            missing.append(f"Missing required element: {desc}")
    return missing
