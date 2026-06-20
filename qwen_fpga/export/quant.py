"""Groupwise symmetric INT4 weight quantization + INT8 activation quantization.

This module defines the *exact* numeric contract shared by three implementations:

  1. the Python quantized reference  (``qwen_fpga/reference/qref.py``),
  2. the C++ functional GEMV reference (``qwen_fpga/kernel/gemv_int4_ref.cpp``),
  3. the HLS GEMV kernel               (``qwen_fpga/kernel/gemv_int4_hls.cpp``).

If any of the three disagree with this file, the FPGA result is wrong. The whole
point of keeping the math in one place is to make that impossible to drift.

GEMV computed everywhere (W is a PyTorch ``nn.Linear`` weight, shape [M, N]):

    y[m] = x_scale * sum_g  w_scale[m, g] * ( sum_{n in group g} wq[m, n] * xq[n] )

  * weights wq         : signed INT4 in [-8, 7], symmetric, per (row, group) scale
  * activations xq     : signed INT8 in [-127, 127], symmetric, single per-call scale
  * inner accumulation : INT32
  * group size G       : contraction-dim tiling for the scales (default 128)

On-disk layout (little-endian):

  <name>.int4.bin    : packed weights, row-major over m; within a row the N int4
                       values are packed 2-per-byte, value at column n in the
                       low nibble when n is even, high nibble when n is odd.
                       size = M * (N // 2) bytes  (N is forced even by padding).
  <name>.scale.bin   : float16 scales, row-major [M, num_groups],
                       num_groups = ceil(N / G).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INT4_QMAX = 7  # symmetric int4 uses [-8, 7]; we clamp to +-7 so |q*scale| is balanced
INT4_QMIN = -8
INT8_QMAX = 127
DEFAULT_GROUP_SIZE = 128


@dataclass(frozen=True)
class QuantizedWeight:
    """A groupwise-INT4 quantized weight matrix, ready to serialize."""

    packed: np.ndarray   # uint8, shape [M, N//2]
    scales: np.ndarray   # float16, shape [M, num_groups]
    out_features: int    # M
    in_features: int     # N (padded to even)
    group_size: int

    @property
    def num_groups(self) -> int:
        return (self.in_features + self.group_size - 1) // self.group_size


def quantize_weight_int4(
    weight: np.ndarray, group_size: int = DEFAULT_GROUP_SIZE
) -> QuantizedWeight:
    """Groupwise symmetric INT4 quantize a [M, N] weight matrix.

    ``weight`` is a PyTorch Linear weight (out_features, in_features).
    """
    w = np.asarray(weight, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"expected 2-D weight, got shape {w.shape}")
    m, n = w.shape

    # Pad the contraction dim to an even length so int4 packs cleanly 2-per-byte.
    n_pad = n + (n & 1)
    if n_pad != n:
        w = np.concatenate([w, np.zeros((m, 1), dtype=np.float32)], axis=1)

    num_groups = (n_pad + group_size - 1) // group_size
    n_full = num_groups * group_size
    if n_full != n_pad:
        w = np.concatenate(
            [w, np.zeros((m, n_full - n_pad), dtype=np.float32)], axis=1
        )

    wg = w.reshape(m, num_groups, group_size)
    # symmetric scale per (row, group): max-abs / 7
    absmax = np.max(np.abs(wg), axis=2)  # [m, num_groups]
    scales = (absmax / INT4_QMAX).astype(np.float32)
    scales[scales == 0.0] = 1.0  # avoid div-by-zero on all-zero groups

    q = np.round(wg / scales[:, :, None])
    q = np.clip(q, INT4_QMIN, INT4_QMAX).astype(np.int8)
    q = q.reshape(m, n_full)[:, :n_pad]  # drop group padding, keep even N

    packed = pack_int4_rows(q)
    return QuantizedWeight(
        packed=packed,
        scales=scales.astype(np.float16),
        out_features=m,
        in_features=n_pad,
        group_size=group_size,
    )


def pack_int4_rows(q: np.ndarray) -> np.ndarray:
    """Pack signed int4 values (shape [M, N], N even) into uint8 [M, N//2].

    Column n goes in the low nibble when n is even, high nibble when n is odd.
    """
    m, n = q.shape
    if n & 1:
        raise ValueError("N must be even before packing")
    lo = (q[:, 0::2].astype(np.int32) & 0xF).astype(np.uint8)
    hi = (q[:, 1::2].astype(np.int32) & 0xF).astype(np.uint8)
    return (lo | (hi << 4)).astype(np.uint8)


def unpack_int4_rows(packed: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pack_int4_rows`; returns signed int8 in [-8, 7]."""
    lo = (packed & 0x0F).astype(np.int8)
    hi = ((packed >> 4) & 0x0F).astype(np.int8)
    # sign-extend 4-bit -> 8-bit
    lo = np.where(lo >= 8, lo - 16, lo)
    hi = np.where(hi >= 8, hi - 16, hi)
    m, half = packed.shape
    out = np.empty((m, half * 2), dtype=np.int8)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    return out


def dequantize_weight(qw: QuantizedWeight) -> np.ndarray:
    """Reconstruct the float32 weight matrix from its INT4 groupwise form."""
    q = unpack_int4_rows(qw.packed).astype(np.float32)  # [M, N]
    m, n = q.shape
    g = qw.group_size
    num_groups = qw.num_groups
    qg = q.reshape(m, num_groups, g) if n == num_groups * g else None
    if qg is None:
        # N is even but not a multiple of group_size for the last group
        n_full = num_groups * g
        qpad = np.concatenate([q, np.zeros((m, n_full - n), dtype=np.float32)], 1)
        qg = qpad.reshape(m, num_groups, g)
    deq = qg * qw.scales.astype(np.float32)[:, :, None]
    return deq.reshape(m, -1)[:, :n]


def quantize_activation_int8(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-vector symmetric INT8 quantize; returns (int8 codes, scale)."""
    x = np.asarray(x, dtype=np.float32).ravel()
    amax = float(np.max(np.abs(x))) if x.size else 0.0
    scale = amax / INT8_QMAX if amax > 0 else 1.0
    xq = np.clip(np.round(x / scale), -INT8_QMAX, INT8_QMAX).astype(np.int8)
    return xq, scale


def gemv_int4_quantized(qw: QuantizedWeight, x: np.ndarray) -> np.ndarray:
    """Reference GEMV in the *quantized* domain (what the FPGA actually computes).

    Returns float32 y of length M. This mirrors the C++/HLS datapath exactly:
    int4 * int8 -> int32 group accumulation -> fp scale -> accumulate.
    """
    xq, x_scale = quantize_activation_int8(x)
    wq = unpack_int4_rows(qw.packed).astype(np.int32)  # [M, N]
    m, n = wq.shape
    # pad activation to N (in_features padded to even on quantize)
    if xq.shape[0] < n:
        xq = np.concatenate([xq, np.zeros(n - xq.shape[0], dtype=np.int8)])
    xq = xq[:n].astype(np.int32)

    g = qw.group_size
    num_groups = qw.num_groups
    n_full = num_groups * g
    if n_full != n:
        wq = np.concatenate([wq, np.zeros((m, n_full - n), dtype=np.int32)], 1)
        xqf = np.concatenate([xq, np.zeros(n_full - n, dtype=np.int32)])
    else:
        xqf = xq
    wq_g = wq.reshape(m, num_groups, g)
    # int32 accumulate within each group, then scale per group
    acc = np.einsum("mgk,gk->mg", wq_g, xqf.reshape(num_groups, g)).astype(np.float64)
    y = (acc * qw.scales.astype(np.float64)).sum(axis=1) * x_scale
    return y.astype(np.float32)
