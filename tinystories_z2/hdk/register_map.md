# gemv_int8 kernel AXI-Lite register map (PYNQ Z2)

The `gemv_int8` kernel is controlled through its AXI-Lite slave (`s_axi_control`),
reached from the PS via `M_AXI_GP0`. Offsets match the `#define`s in
[`kernel/gemv_int8.hpp`](../kernel/gemv_int8.hpp) and the writes the PYNQ host
(`host/gemv_backends.py::PynqGemv`) issues. On the Z2 the `*_BASE` registers hold
**PS-DDR physical addresses** from `pynq.allocate(...).device_address` (not HBM
offsets).

| Offset | Name            | Dir | Meaning                                            |
|--------|-----------------|-----|----------------------------------------------------|
| 0x000  | CONTROL         | R/W | ap_ctrl: bit0 `ap_start`, bit1 `ap_done`, bit2 `ap_idle`, bit3 `ap_ready` |
| 0x010  | W_BASE          | W   | DDR address of INT8 weights `[M,N]` (64-bit)       |
| 0x018  | SCALE_BASE      | W   | DDR address of fp16 per-row scales `[M]`           |
| 0x020  | X_BASE          | W   | DDR address of INT8 activation `[N]`               |
| 0x028  | Y_BASE          | W   | DDR address of fp32 output `[M]`                   |
| 0x030  | XSCALE          | W   | IEEE-754 fp32 bits of the activation scale         |
| 0x034  | M               | W   | output rows (out_features)                         |
| 0x038  | N               | W   | contraction dim (in_features)                      |

> Vitis HLS assigns the exact offsets in the generated
> `gemv_int8_prj/sol1/impl/ip/drivers/.../xgemv_int8_hw.h`. If they differ from
> the table above (HLS sometimes shifts 64-bit args), read them from that header
> and update `PynqGemv`'s `R_*` constants — they must match the synthesized core.

## Datapath contract

```
y[m] = x_scale * w_scale[m] * sum_n ( wq[m,n] * xq[n] )
```

* `wq` signed INT8 `[M,N]` row-major; `xq` signed INT8 `[N]`, single per-call `x_scale`
* `w_scale` fp16 per output row; inner accumulation INT32, output FP32
* identical to `tinystories_z2/quant.py` and verified by `make check` (49/49 golden)

## DDR residency

Each weight matrix + its scales is written into PS-DDR **once** (via
`pynq.allocate`) and reused across every decoded token. Only the INT8 activation
(a few hundred bytes) and the fp32 result move per GEMV. The two HP ports give
~1.2–1.5 GB/s each; `gmem_w` is the bandwidth-critical master, so it gets its own
HP port (HP0) in `build_bd.tcl`.
