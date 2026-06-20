"""Tests for glm.parsing.hls — extracting HLS source from GLM output."""

import pytest

from glm.parsing.hls import (
    HLSParseResult,
    parse_hls_from_text,
    validate_kernel_bundle,
)


def test_parse_single_cpp_block():
    text = '''Here is the kernel:

```cpp
// file: kernel_top.cpp
#include "kernel_top.h"

void kernel_top(const float input[16], float output[16]) {
    #pragma HLS INTERFACE m_axi port=input
    #pragma HLS INTERFACE m_axi port=output
    for (int i = 0; i < 16; i++) {
        output[i] = input[i] * 2.0f;
    }
}
```
'''
    result = parse_hls_from_text(text)
    assert result.success
    assert "kernel_top.cpp" in result.sources
    assert "kernel_top" in result.sources["kernel_top.cpp"]


def test_parse_header_and_source():
    text = '''```cpp
#ifndef KERNEL_TOP_H
#define KERNEL_TOP_H
void kernel_top(const float input[16], float output[16]);
#endif
```

```cpp
#include "kernel_top.h"

void kernel_top(const float input[16], float output[16]) {
    #pragma HLS INTERFACE m_axi port=input
    #pragma HLS INTERFACE m_axi port=output
    for (int i = 0; i < 16; i++) {
        output[i] = input[i];
    }
}
```
'''
    result = parse_hls_from_text(text)
    assert result.success
    assert "kernel_top.h" in result.sources
    assert "kernel_top.cpp" in result.sources


def test_parse_no_code_blocks():
    result = parse_hls_from_text("This is just text with no code.")
    assert not result.success
    assert "No code blocks found" in result.errors[0]


def test_validate_good_bundle():
    sources = {
        "kernel_top.h": "#ifndef K_H\n#define K_H\nvoid kernel_top(const float* in, float* out);\n#endif\n",
        "kernel_top.cpp": (
            '#include "kernel_top.h"\n'
            "void kernel_top(const float* in, float* out) {\n"
            "    #pragma HLS INTERFACE m_axi port=in\n"
            "    #pragma HLS INTERFACE m_axi port=out\n"
            "}\n"
        ),
    }
    issues = validate_kernel_bundle(sources)
    assert len(issues) == 0


def test_validate_forbidden_patterns():
    sources = {
        "kernel_top.h": "void kernel_top();\n",
        "kernel_top.cpp": (
            "void kernel_top() {\n"
            "    #pragma HLS INTERFACE s_axilite port=return\n"
            "    int* p = new int[10];\n"
            "    delete[] p;\n"
            "}\n"
        ),
    }
    issues = validate_kernel_bundle(sources)
    assert any("dynamic allocation" in i for i in issues)
    assert any("deallocation" in i for i in issues)


def test_validate_missing_kernel_top():
    sources = {
        "kernel_top.h": "void some_function();\n",
        "kernel_top.cpp": "void some_function() {}\n",
    }
    issues = validate_kernel_bundle(sources)
    assert any("kernel_top" in i for i in issues)
