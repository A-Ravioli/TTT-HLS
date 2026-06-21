"""Pluggable W8A8 GEMV backends for the TinyStories host runtime.

The GPT-Neo op schedule (``tinystories_z2/model.py``) is identical regardless of
where the matrix-vector multiplies run. Only this dispatch changes:

    numpy : pure-numpy quantized reference (always available; FPGA-equivalent ints)
    cpp   : the compiled C++ datapath via ctypes -- the *same* arithmetic the HLS
            kernel synthesizes, on the CPU (bring-up / software fallback)
    pynq  : the real Zynq-7020 PL via the PYNQ runtime -- weights resident in
            PS-DDR, AXI-Lite register kick, result read back (Stage 3, on-board)

Each backend exposes ``gemv(qw: QuantizedWeightInt8, x: np.ndarray) -> np.ndarray``.
"""
from __future__ import annotations

import ctypes
import os
import time
from pathlib import Path

import numpy as np

from tinystories_z2.quant import (
    QuantizedWeightInt8,
    gemv_int8_quantized,
    quantize_activation_int8,
)


class GemvBackend:
    name = "base"

    def __init__(self) -> None:
        self.calls = 0
        self.total_s = 0.0

    def gemv(self, qw: QuantizedWeightInt8, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _tick(self, t0: float) -> None:
        self.calls += 1
        self.total_s += time.perf_counter() - t0

    @property
    def avg_ms(self) -> float:
        return 1e3 * self.total_s / self.calls if self.calls else 0.0


class NumpyGemv(GemvBackend):
    name = "numpy"

    def gemv(self, qw: QuantizedWeightInt8, x: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        y = gemv_int8_quantized(qw, x)
        self._tick(t0)
        return y


def _prep(qw: QuantizedWeightInt8, x: np.ndarray):
    """Quantize the activation and lay out contiguous buffers for a kernel call."""
    xq, x_scale = quantize_activation_int8(x)
    if xq.shape[0] < qw.in_features:
        xq = np.concatenate([xq, np.zeros(qw.in_features - xq.shape[0], np.int8)])
    xq = np.ascontiguousarray(xq[: qw.in_features], dtype=np.int8)
    w = np.ascontiguousarray(qw.q, dtype=np.int8)
    s = np.ascontiguousarray(qw.scales, dtype=np.float16)
    return w, s, xq, float(x_scale)


class CppGemv(GemvBackend):
    """Drives the compiled C++ datapath (libgemv_int8) -- same math as the FPGA."""

    name = "cpp"

    def __init__(self, lib_path: str | None = None) -> None:
        super().__init__()
        lib_path = lib_path or os.environ.get("TS_Z2_GEMV_LIB")
        if lib_path is None:
            root = Path(__file__).resolve().parents[1]
            for cand in ("libgemv_int8.so", "libgemv_int8.dylib"):
                p = root / "build" / cand
                if p.exists():
                    lib_path = str(p)
                    break
        if lib_path is None or not Path(lib_path).exists():
            raise FileNotFoundError("libgemv_int8 not built. Run `make -C tinystories_z2 lib`.")
        self.lib = ctypes.CDLL(lib_path)
        self.lib.gemv_int8_capi.restype = None
        self.lib.gemv_int8_capi.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_float, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ]

    def gemv(self, qw: QuantizedWeightInt8, x: np.ndarray) -> np.ndarray:
        w, s, xq, x_scale = _prep(qw, x)
        y = np.empty(qw.out_features, dtype=np.float32)
        t0 = time.perf_counter()
        self.lib.gemv_int8_capi(
            w.ctypes.data, s.ctypes.data, xq.ctypes.data,
            ctypes.c_float(x_scale), y.ctypes.data,
            int(qw.out_features), int(qw.in_features))
        self._tick(t0)
        return y


class PynqGemv(GemvBackend):
    """Real Zynq-7020 backend via the PYNQ runtime (Stage 3, on-board only).

    Loads the overlay once, allocates contiguous PS-DDR buffers, and caches each
    weight matrix in DDR across tokens (only the activation is re-written per
    call). Drives the kernel through the AXI-Lite register map in
    ``kernel/gemv_int8.hpp`` and reads the fp32 result back.

    Off-board the ``pynq`` import fails and the host falls back to ``cpp``/``numpy``.
    """

    name = "pynq"

    def __init__(self, bitfile: str | None = None) -> None:
        super().__init__()
        from pynq import Overlay, allocate  # noqa: F401  (only present on the board)

        self._allocate = allocate
        bitfile = bitfile or os.environ.get("TS_Z2_OVERLAY")
        if not bitfile:
            raise RuntimeError("set TS_Z2_OVERLAY to the .bit overlay path")
        self.overlay = Overlay(bitfile)
        # HLS exposes the kernel as an MMIO IP; name comes from the block design.
        self.ip = getattr(self.overlay, os.environ.get("TS_Z2_IP", "gemv_int8_0"))
        self._wbuf: dict[int, tuple] = {}  # id(qw) -> (w_buf, s_buf)

    # AXI-Lite offsets (see kernel/gemv_int8.hpp)
    R_CTRL, R_W_LO, R_S_LO, R_X_LO, R_Y_LO, R_XS, R_M, R_N = (
        0x000, 0x010, 0x018, 0x020, 0x028, 0x030, 0x034, 0x038)

    def _weight_buffers(self, qw: QuantizedWeightInt8):
        key = id(qw)
        if key in self._wbuf:
            return self._wbuf[key]
        w_buf = self._allocate(shape=(qw.out_features, qw.in_features), dtype=np.int8)
        s_buf = self._allocate(shape=(qw.out_features,), dtype=np.float16)
        w_buf[:] = np.ascontiguousarray(qw.q, dtype=np.int8)
        s_buf[:] = np.ascontiguousarray(qw.scales, dtype=np.float16)
        w_buf.flush(); s_buf.flush()
        self._wbuf[key] = (w_buf, s_buf)
        return w_buf, s_buf

    def gemv(self, qw: QuantizedWeightInt8, x: np.ndarray) -> np.ndarray:
        _, _, xq, x_scale = _prep(qw, x)
        w_buf, s_buf = self._weight_buffers(qw)
        x_buf = self._allocate(shape=(qw.in_features,), dtype=np.int8)
        y_buf = self._allocate(shape=(qw.out_features,), dtype=np.float32)
        x_buf[:] = xq
        x_buf.flush()

        t0 = time.perf_counter()
        self.ip.write(self.R_W_LO, w_buf.device_address)
        self.ip.write(self.R_S_LO, s_buf.device_address)
        self.ip.write(self.R_X_LO, x_buf.device_address)
        self.ip.write(self.R_Y_LO, y_buf.device_address)
        self.ip.write(self.R_XS, int(np.float32(x_scale).view(np.uint32)))
        self.ip.write(self.R_M, int(qw.out_features))
        self.ip.write(self.R_N, int(qw.in_features))
        self.ip.write(self.R_CTRL, 0x1)  # ap_start
        while (self.ip.read(self.R_CTRL) & 0x2) == 0:  # poll ap_done
            pass
        self._tick(t0)

        y_buf.invalidate()
        return np.array(y_buf, dtype=np.float32)


def make_gemv_backend(name: str | None = None) -> GemvBackend | None:
    name = (name or os.environ.get("TS_Z2_BACKEND", "numpy")).lower()
    if name == "numpy":
        return None  # NeoRunner uses gemv_int8_quantized directly
    if name == "cpp":
        try:
            return CppGemv()
        except FileNotFoundError:
            return NumpyGemv()
    if name in ("pynq", "fpga"):
        return PynqGemv()
    raise ValueError(f"unknown backend {name!r}")
