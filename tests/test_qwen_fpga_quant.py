"""Tests for the Qwen-on-F2 INT4 GEMV numeric contract.

These guard the *shared* quantization math that the Python reference, the C++
functional reference, and the HLS kernel all depend on. If they drift, the FPGA
result drifts.
"""
import numpy as np

from qwen_fpga.export.quant import (
    dequantize_weight,
    gemv_int4_quantized,
    pack_int4_rows,
    quantize_activation_int8,
    quantize_weight_int4,
    unpack_int4_rows,
)


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(0)
    q = rng.integers(-8, 8, size=(5, 16)).astype(np.int8)
    assert np.array_equal(unpack_int4_rows(pack_int4_rows(q)), q)


def test_dequant_close_to_original():
    rng = np.random.default_rng(1)
    w = (rng.standard_normal((32, 256)) * 0.05).astype(np.float32)
    qw = quantize_weight_int4(w, group_size=128)
    deq = dequantize_weight(qw)
    rel = np.abs(deq - w).max() / np.abs(w).max()
    assert rel < 0.15  # int4 groupwise


def test_quantized_gemv_matches_fp32_direction():
    rng = np.random.default_rng(2)
    w = (rng.standard_normal((64, 384)) * 0.05).astype(np.float32)
    x = rng.standard_normal(384).astype(np.float32)
    qw = quantize_weight_int4(w, group_size=128)
    y_q = gemv_int4_quantized(qw, x)
    y_fp = w @ x
    cos = np.dot(y_q, y_fp) / (np.linalg.norm(y_q) * np.linalg.norm(y_fp))
    assert cos > 0.97


def test_activation_int8_range():
    x = np.array([0.0, 1.0, -2.0, 0.5], dtype=np.float32)
    xq, scale = quantize_activation_int8(x)
    assert xq.dtype == np.int8
    assert np.abs(xq).max() <= 127
    assert abs(scale - 2.0 / 127) < 1e-9


def test_in_features_padded_even():
    w = np.ones((3, 7), dtype=np.float32)
    qw = quantize_weight_int4(w, group_size=128)
    assert qw.in_features % 2 == 0
    assert qw.packed.shape == (3, qw.in_features // 2)
