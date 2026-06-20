"""Pluggable GEMV backends for the Qwen host runtime.

The Qwen op schedule (``reference/qref.py``) is identical regardless of where the
matrix-vector multiplies run. Only this dispatch changes:

    numpy : pure-Python reference (correctness ground truth, always available)
    cpp   : the compiled C++ datapath via ctypes -- the *same* arithmetic as the
            HLS kernel, executed on CPU (bring-up / software fallback)
    xrt   : the real AWS F2 FPGA via XRT -- weights+activation in HBM, AXI-Lite
            register kick, result DMA'd back

Select with ``QWEN_FPGA_BACKEND={numpy,cpp,xrt}``. On the F2 instance you set
``xrt`` and point ``QWEN_FPGA_XCLBIN`` at the built AFI/xclbin; everywhere else
``cpp`` (or ``numpy``) gives bit-identical numerics so the whole pipeline -- and
the webapp -- runs and is testable off-board.
"""
from __future__ import annotations

import ctypes
import os
import time
from pathlib import Path

import numpy as np

from qwen_fpga.export.quant import QuantizedWeight, quantize_activation_int8, gemv_int4_quantized


class GemvBackend:
    name = "base"

    def gemv(self, qw: QuantizedWeight, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # rolling latency stats (seconds) of the matvec dispatch only
    def __init__(self) -> None:
        self.calls = 0
        self.total_s = 0.0

    def _tick(self, t0: float) -> None:
        self.calls += 1
        self.total_s += time.perf_counter() - t0

    @property
    def avg_ms(self) -> float:
        return 1e3 * self.total_s / self.calls if self.calls else 0.0


class NumpyGemv(GemvBackend):
    name = "numpy"

    def gemv(self, qw: QuantizedWeight, x: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        y = gemv_int4_quantized(qw, x)
        self._tick(t0)
        return y


class CppGemv(GemvBackend):
    """Drives the compiled C++ datapath (libgemv_int4) -- same math as the FPGA."""

    name = "cpp"

    def __init__(self, lib_path: str | None = None) -> None:
        super().__init__()
        lib_path = lib_path or os.environ.get("QWEN_FPGA_GEMV_LIB")
        if lib_path is None:
            root = Path(__file__).resolve().parents[1]
            for cand in ("libgemv_int4.so", "libgemv_int4.dylib"):
                p = root / "build" / cand
                if p.exists():
                    lib_path = str(p)
                    break
        if lib_path is None or not Path(lib_path).exists():
            raise FileNotFoundError(
                "libgemv_int4 not built. Run `make -C qwen_fpga lib` first.")
        self.lib = ctypes.CDLL(lib_path)
        self.lib.gemv_int4_capi.restype = None
        self.lib.gemv_int4_capi.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_float, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]

    def gemv(self, qw: QuantizedWeight, x: np.ndarray) -> np.ndarray:
        xq, x_scale = quantize_activation_int8(x)
        if xq.shape[0] < qw.in_features:
            xq = np.concatenate([xq, np.zeros(qw.in_features - xq.shape[0], np.int8)])
        xq = np.ascontiguousarray(xq[: qw.in_features], dtype=np.int8)
        packed = np.ascontiguousarray(qw.packed, dtype=np.uint8)
        scales = np.ascontiguousarray(qw.scales, dtype=np.float16)
        y = np.empty(qw.out_features, dtype=np.float32)
        t0 = time.perf_counter()
        self.lib.gemv_int4_capi(
            packed.ctypes.data, scales.ctypes.data, xq.ctypes.data,
            ctypes.c_float(float(x_scale)), y.ctypes.data,
            int(qw.out_features), int(qw.in_features), int(qw.group_size))
        self._tick(t0)
        return y


class XrtGemv(GemvBackend):
    """AWS F2 FPGA backend via XRT (pyxrt).

    Streams packed weights + the INT8 activation into HBM, kicks the GEMV kernel
    through the AXI-Lite register map (see hdk/register_map.md), and DMAs the
    fp32 result back. Weight buffers are cached in HBM across tokens so per-token
    decode only re-DMAs the activation (the brief's "weights live in HBM").

    Requires the F2 toolchain (pyxrt + a loaded AGFI/xclbin). Off-board this
    import fails and the host falls back to ``cpp``/``numpy``.
    """

    name = "xrt"

    def __init__(self, xclbin: str | None = None) -> None:
        super().__init__()
        import pyxrt  # noqa: F401  (only present on the F2 instance)

        self.pyxrt = pyxrt
        xclbin = xclbin or os.environ.get("QWEN_FPGA_XCLBIN")
        if not xclbin:
            raise RuntimeError("set QWEN_FPGA_XCLBIN to the built AFI/xclbin path")
        self.device = pyxrt.device(int(os.environ.get("QWEN_FPGA_DEVICE", "0")))
        self.uuid = self.device.load_xclbin(xclbin)
        self.kernel = pyxrt.kernel(self.device, self.uuid, "gemv_int4",
                                   pyxrt.kernel.cu_access_mode.exclusive)
        self._wbuf: dict[int, tuple] = {}  # id(qw) -> (w_bo, s_bo)

    def _weight_buffers(self, qw: QuantizedWeight):
        key = id(qw)
        if key in self._wbuf:
            return self._wbuf[key]
        pyxrt = self.pyxrt
        packed = np.ascontiguousarray(qw.packed, dtype=np.uint8).ravel()
        scales = np.ascontiguousarray(qw.scales, dtype=np.float16).ravel()
        w_bo = pyxrt.bo(self.device, packed.nbytes, pyxrt.bo.normal,
                        self.kernel.group_id(0))
        s_bo = pyxrt.bo(self.device, scales.nbytes, pyxrt.bo.normal,
                        self.kernel.group_id(1))
        w_bo.write(packed.tobytes(), 0); w_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        s_bo.write(scales.tobytes(), 0); s_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self._wbuf[key] = (w_bo, s_bo)
        return w_bo, s_bo

    def gemv(self, qw: QuantizedWeight, x: np.ndarray) -> np.ndarray:
        pyxrt = self.pyxrt
        xq, x_scale = quantize_activation_int8(x)
        if xq.shape[0] < qw.in_features:
            xq = np.concatenate([xq, np.zeros(qw.in_features - xq.shape[0], np.int8)])
        xq = np.ascontiguousarray(xq[: qw.in_features], dtype=np.int8)

        w_bo, s_bo = self._weight_buffers(qw)
        x_bo = pyxrt.bo(self.device, xq.nbytes, pyxrt.bo.normal, self.kernel.group_id(2))
        y_bo = pyxrt.bo(self.device, qw.out_features * 4, pyxrt.bo.normal,
                        self.kernel.group_id(3))
        x_bo.write(xq.tobytes(), 0)
        x_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        t0 = time.perf_counter()
        run = self.kernel(w_bo, s_bo, x_bo, y_bo, float(x_scale),
                          int(qw.out_features), int(qw.in_features), int(qw.group_size))
        run.wait()
        self._tick(t0)

        y_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        return np.frombuffer(y_bo.read(qw.out_features * 4, 0), dtype=np.float32).copy()


def make_gemv_backend(name: str | None = None) -> GemvBackend:
    name = (name or os.environ.get("QWEN_FPGA_BACKEND", "cpp")).lower()
    if name == "numpy":
        return NumpyGemv()
    if name == "cpp":
        try:
            return CppGemv()
        except FileNotFoundError:
            return NumpyGemv()
    if name in ("xrt", "fpga"):
        return XrtGemv()
    raise ValueError(f"unknown backend {name!r}")
