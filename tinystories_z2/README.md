# TinyStories on a PYNQ Z2 — W8A8 GEMV-offload decoder

Run a small **GPT-Neo language model (TinyStories) on a Xilinx PYNQ Z2**
(Zynq-7020). Same host/FPGA split as `qwen_fpga`, retargeted from AWS F2 + HBM +
XRT down to a tiny Zynq SoC: the ARM core runs tokenizer + glue math, every large
linear is an INT8 GEMV, and the weights live in the PS-side 512 MB DDR3.

```
ARM (PS)                         FPGA fabric (PL)            DDR3 (512 MB)
  tokenize                         W8A8 GEMV datapath          INT8 weights [M,N]
  LayerNorm / GeLU / softmax ────► (AXI-HP master to DDR) ───► fp16 per-row scales
  pos-embed / sample         ◄──── AXI-Lite control regs       INT8 activation / fp32 out
```

## Why W8A8 (not the INT4 the Qwen path uses)

The Qwen/F2 path uses groupwise INT4 because a 2 B model's **HBM footprint** is the
binding constraint. On a Z2 the model is a few MB and DDR capacity is free, so the
constraint flips to **accuracy**: INT4 is ~10 % per-matrix error and compounds
across 8 layers into degenerate output. Measured here:

| scheme | lm_head rel. error | TinyStories-1M decode |
|--------|-------------------|------------------------|
| INT4 groupwise | ~0.097 | degenerate (`........`) |
| **INT8 per-row (W8A8)** | **~0.005** | **fluent** |

So the Z2 contract is W8A8: per-row symmetric INT8 weights, per-vector INT8
activations, INT32 accumulate, fp32 out (`quant.py`). Byte-aligned and tiny — the
smallest, fastest datapath for the limited fabric.

## Stage 1 — numeric foundation (done, off-board, tested)

Everything here runs and is verified on a laptop with **no FPGA and no Vivado**:

| Piece | File | Status |
|-------|------|--------|
| W8A8 quant contract | `quant.py` | the single source of truth |
| TinyStories → INT8 export | `export_tinystories.py` | runs on `roneneldan/TinyStories-1M`, 8 layers |
| GPT-Neo decode (pluggable linears) | `model.py` | matches HF fp32 (max \|Δlogit\| ≈ 8e-5; **identical greedy text**) |
| Quantized decode = FPGA math | `model.py` (`QuantWeights`) | **fluent** generation via the exact INT32 datapath |
| Generate CLI | `generate.py` | ~170 tok/s numpy reference on a laptop |
| Tests | `tests/test_tinystories_z2.py` | arch match, export round-trip, coherence, act-quant error |

```bash
python -m tinystories_z2.export_tinystories --model roneneldan/TinyStories-1M
python -m tinystories_z2.generate \
    --manifest tinystories_z2/weights/TinyStories-1M/manifest.json \
    --prompt "Once upon a time, there was a robot who" --max-new 70
```

The numpy backend is **bit-exact to the integer arithmetic the FPGA will run**, so
"coherent text here" means the hardware has a correct, matching target — exactly
the de-risking the `qwen_fpga` C++ datapath provides for F2.

## Stage 2 — the Z2 GEMV kernel + bitstream (needs Linux + Vivado)

A small Vitis-HLS W8A8 GEMV mirroring `quant.gemv_int8_quantized`:

```c
// y[m] = x_scale * w_scale[m] * sum_n ( wq[m,n] * xq[n] )
//   m_axi gmem_w  : INT8 weights  [M*N]   -> S_AXI_HP0 (PS DDR)
//   m_axi gmem_s  : fp16 scales   [M]     -> S_AXI_HP0
//   m_axi gmem_x  : INT8 acts     [N]     -> S_AXI_HP1
//   m_axi gmem_y  : fp32 out      [M]     -> S_AXI_HP1
//   s_axilite     : x_scale, M, N, ap_ctrl  -> M_AXI_GP0
```

It is a *simplification* of `qwen_fpga/kernel/gemv_int4_hls.cpp`: no nibble
unpacking, no group loop, and the activation cache binds to **BRAM** (the Z7020
has no URAM). DSP unroll ~32–64 wide fits the 220-DSP budget; decode stays
memory-bound on DDR bandwidth.

Build (on Linux x86 — **Vivado does not run on macOS**; free Vivado ML Standard
covers XC7Z020):

```
vitis_hls -f hdk/run_hls.tcl          # HLS C++ -> RTL IP
vivado    -source hdk/build_bd.tcl     # Zynq7 PS + gemv IP -> .bit + .hwh
```

Match Vivado to the board image: PYNQ-Z2 v2.7 → Vivado **2020.2**; PYNQ 3.0 →
**2022.1**. Drop the TUL pynq-z2 board files into `<Vivado>/data/boards/`.

## Stage 3 — on-board bring-up

A `host/pynq_gemv.py` `PynqGemv` backend (sibling of `qwen_fpga`'s `XrtGemv`):
`pynq.Overlay` + `pynq.allocate` contiguous DDR buffers (weights allocated once,
persisted across tokens; only the activation re-written per call) + MMIO writes of
`buf.device_address` into the AXI-Lite registers. Then `generate.py --backend pynq`
runs the same decode loop on hardware.

Milestone ladder (mirrors `qwen_fpga`): single GEMV vs golden → one layer → full
decode → webapp `/api/chat` with `backend="pynq"` + measured tok/s.

## Files

```
quant.py                W8A8 quant + reference GEMV (the contract)
model.py                GPT-Neo decode; HFWeights (fp32) | QuantWeights (W8A8)
export_tinystories.py   GPT-Neo checkpoint -> INT8 bins + manifest.json
generate.py             host decode loop / CLI (numpy now, pynq at Stage 3)
weights/<tag>/          exported artifacts (git-ignored)
```
