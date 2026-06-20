"""Qwen-on-F2-FPGA: HBM-streaming INT4 GEMV decoder accelerator.

This package is intentionally separate from the BurnTTT GLM-compiler code. It is
the *deployable* Qwen-inference-on-FPGA path described in the project brief:

    Host CPU : tokenize, orchestrate per-op FPGA calls, sample.
    FPGA     : INT4 HBM-streaming GEMV (the spine of decode), + glue ops.
    HBM      : quantized weights, KV cache, activation scratch.

The Python here (export + reference + golden vectors) is the correctness ground
truth. The C++/HLS under ``kernel/``, ``host/`` and ``hdk/`` is the artifact that
gets built on the AWS F2 instance via the AWS HDK (Vivado -> DCP -> AFI -> AGFI).
"""
