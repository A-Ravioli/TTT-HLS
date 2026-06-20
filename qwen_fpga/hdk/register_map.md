# GEMV kernel AXI-Lite register map

The `gemv_int4` kernel is controlled through an AXI-Lite slave. Offsets match the
`#define`s in [`kernel/gemv_int4.hpp`](../kernel/gemv_int4.hpp) and the argument
order the XRT host (`host/fpga_gemv.py::XrtGemv`) uses.

| Offset | Name            | Dir | Meaning                                                |
|--------|-----------------|-----|--------------------------------------------------------|
| 0x000  | CONTROL         | W   | bit0 `start`, bit1 `reset`                              |
| 0x004  | STATUS          | R   | bit0 `done`, bit1 `busy`, bit2 `error`                 |
| 0x010  | W_BASE_LO       | W   | HBM byte address of packed INT4 weights (low 32b)      |
| 0x014  | W_BASE_HI       | W   | weights base (high 32b)                                |
| 0x018  | X_BASE_LO       | W   | HBM address of INT8 activation vector (low)            |
| 0x01c  | X_BASE_HI       | W   | activation base (high)                                 |
| 0x020  | Y_BASE_LO       | W   | HBM address of FP32 output vector (low)                |
| 0x024  | Y_BASE_HI       | W   | output base (high)                                     |
| 0x028  | SCALE_BASE_LO   | W   | HBM address of FP16 group scales (low)                 |
| 0x02c  | SCALE_BASE_HI   | W   | scales base (high)                                     |
| 0x030  | M               | W   | output rows (out_features)                             |
| 0x034  | N               | W   | contraction dim (in_features, padded even)             |
| 0x038  | GROUP_SIZE      | W   | quantization group size (e.g. 128)                     |
| 0x03c  | FLAGS           | W   | bit0 `x_scale` present in XSCALE                        |
| 0x040  | XSCALE          | W   | IEEE-754 fp32 bits of the activation scale             |

## Datapath contract

```
y[m] = x_scale * sum_g  w_scale[m,g] * ( sum_{n in group g} wq[m,n] * xq[n] )
```

* `wq`  signed INT4 in [-8,7], packed 2/byte (col n -> low nibble if n even)
* `xq`  signed INT8, single per-call `x_scale`
* group accumulation INT32, output accumulation FP32
* identical to `qwen_fpga/export/quant.py` and verified by the golden test

## HBM residency

Weight + scale buffers are written into HBM **once** and reused across every
decoded token (the brief's "weights live in HBM, not LUTs"). Only the INT8
activation (a few KiB) and the FP32 result are transferred per GEMV. F2 exposes
16 GiB HBM over 32 AXI3 channels (~460 GB/s aggregate); the `gmem_w` master is
the bandwidth-critical port and should be spread across channels in P&R.
