"""W8A8 GEMV contract for the PYNQ Z2 TinyStories accelerator.

The Qwen/F2 path uses groupwise INT4 weights because a 2 B model's *capacity*
(HBM footprint) is the binding constraint. On a Zynq-7020 driving a few-MB
TinyStories model out of 512 MB of DDR, capacity is free and INT4's ~10 %
per-matrix error compounds across the 8 layers into degenerate output. So the Z2
path uses the simpler, far more accurate **W8A8** scheme (measured ~0.5 % weight
error, fluent generation):

    y[m] = x_scale * w_scale[m] * sum_n ( wq[m, n] * xq[n] )

  * weights  wq : signed INT8 in [-127, 127], one symmetric scale per output row
  * acts     xq : signed INT8 in [-127, 127], one symmetric scale per call
  * inner accum : INT32
  * output      : FP32

This is the single source of truth for the numbers: the Python reference here,
the C++ GEMV reference, and the Z2 HLS kernel (Stage 2) must all match it. It is
deliberately byte-aligned (no nibble packing, no group index) so the HLS datapath
is as small and as fast as possible on the limited Z2 fabric.

On-disk layout (little-endian):

  <name>.int8.bin   : INT8 weights, row-major [M, N]
  <name>.scale.bin  : FP16 per-row scales [M]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INT8_QMAX = 127


@dataclass(frozen=True)
class QuantizedWeightInt8:
    """A per-row symmetric INT8 weight matrix, ready to serialize."""

    q: np.ndarray        # int8, shape [M, N]
    scales: np.ndarray   # float16, shape [M]
    out_features: int    # M
    in_features: int     # N


def quantize_weight_int8(weight: np.ndarray) -> QuantizedWeightInt8:
    """Per-row symmetric INT8 quantize a [M, N] (out_features, in_features) weight."""
    w = np.asarray(weight, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"expected 2-D weight, got shape {w.shape}")
    m, n = w.shape
    absmax = np.max(np.abs(w), axis=1, keepdims=True)  # [M, 1]
    scales = (absmax / INT8_QMAX).astype(np.float32)
    scales[scales == 0.0] = 1.0
    q = np.clip(np.round(w / scales), -INT8_QMAX, INT8_QMAX).astype(np.int8)
    return QuantizedWeightInt8(
        q=q, scales=scales[:, 0].astype(np.float16), out_features=m, in_features=n
    )


def quantize_activation_int8(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-vector symmetric INT8 quantize; returns (int8 codes, scale)."""
    x = np.asarray(x, dtype=np.float32).ravel()
    amax = float(np.max(np.abs(x))) if x.size else 0.0
    scale = amax / INT8_QMAX if amax > 0 else 1.0
    xq = np.clip(np.round(x / scale), -INT8_QMAX, INT8_QMAX).astype(np.int8)
    return xq, scale


def dequantize_weight(qw: QuantizedWeightInt8) -> np.ndarray:
    """Reconstruct the float32 weight matrix from its INT8 per-row form."""
    return qw.q.astype(np.float32) * qw.scales.astype(np.float32)[:, None]


def gemv_int8_quantized(qw: QuantizedWeightInt8, x: np.ndarray) -> np.ndarray:
    """Reference GEMV in the *quantized* domain -- exactly what the FPGA computes.

    INT8 weight x INT8 activation -> INT32 accumulate -> per-row fp scale.
    """
    xq, x_scale = quantize_activation_int8(x)
    n = qw.in_features
    if xq.shape[0] < n:
        xq = np.concatenate([xq, np.zeros(n - xq.shape[0], dtype=np.int8)])
    xq = xq[:n].astype(np.int32)
    acc = (qw.q.astype(np.int32) @ xq).astype(np.float64)  # [M], exact int32 path
    y = acc * qw.scales.astype(np.float64) * x_scale
    return y.astype(np.float32)
