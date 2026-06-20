"""Qwen-2B ingestion and decomposition into FPGA-mappable blocks.

The north-star workload. :mod:`models.qwen.load_qwen` loads a Qwen2 model,
:mod:`models.qwen.decompose` slices it into sub-blocks (RMSNorm / attention / MLP),
and :mod:`models.qwen.blocks` exports a single sub-block (starting with the gated
MLP) into a frontend the feedback engine can ingest, plus golden I/O.

Full-model, multi-block orchestration is staged as research stubs (see
:mod:`models.qwen.decompose`); the first real milestone is one transformer block.
"""
