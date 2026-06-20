# Qwen on AWS F2 — HBM-streaming INT4 GEMV decoder accelerator

This package runs **Qwen transformer inference on an AWS F2 FPGA**, with the
quantized weights resident in HBM and the matrix-vector spine of decode executed
by a custom INT4 GEMV kernel. The CPU only tokenizes, runs the cheap glue
(RMSNorm / RoPE / softmax / SiLU), and samples; every large linear layer
(`q/k/v/o`, `gate/up/down`, `lm_head`) is an FPGA call.

```
Host CPU                         FPGA (custom logic on F2)         HBM (16 GiB)
  tokenize                         INT4 GEMV datapath                packed INT4 weights
  RMSNorm / RoPE / softmax  ─────► (gmem_w/scale/x/y AXI masters) ─► fp16 group scales
  sampling                  ◄───── AXI-Lite control regs            INT8 activations / FP32 out
```

## What is built and verified here (off-board, on this machine)

The hard, de-risking parts are done and tested locally:

| Stage | Artifact | Status |
|-------|----------|--------|
| INT4 groupwise quant contract | `export/quant.py` | unit-tested |
| Weight export (real Qwen → flat bins) | `export/export_weights.py` | ran on Qwen2.5-0.5B-Instruct, 24 layers |
| Quantized Python reference decode | `reference/qref.py` | generates coherent tokens ("Paris") |
| Golden GEMV vectors (every projection) | `reference/make_golden.py` | 168 cases |
| C++ GEMV datapath vs golden | `kernel/gemv_int4_ref` / `test_gemv_ref.cpp` | **168/168 match, rel ≤ 1.6e-7** |
| HLS kernel (same datapath + AXI/HBM) | `kernel/gemv_int4_hls.cpp` | csim-clean |
| Host runtime + pluggable GEMV backend | `host/fpga_gemv.py` | full decode via C++ datapath |
| Chat webapp `/api/chat` | `webapp/server.py` | serves, reports backend + latency |

The C++ datapath in `kernel/gemv_int4.hpp` is **byte-for-byte the arithmetic the
HLS kernel synthesizes**, so "works on CPU" here means "the FPGA has a correct,
matching target." Off-board the host uses `QWEN_FPGA_BACKEND=cpp`; on F2 you flip
it to `xrt` and the same calls hit the FPGA.

## What must be built on the F2 instance

Vivado synthesis, DCP→AFI generation, and `fpga-load-local-image` only run on the
F2 (or an F2-build) instance. See [`hdk/README.md`](hdk/README.md). Short version:

```bash
vitis_hls -f hdk/run_hls.tcl          # HLS C++ -> RTL IP
# instantiate gemv_int4 in a cl_qwen_gemv CL derived from CL_MEM_PERF
bash hdk/build_afi.sh                  # DCP -> AFI -> wait -> prints AGFI
sudo fpga-load-local-image -S 0 -I <AGFI>
```

We use the **HDK/Vivado path** (HLS only to emit RTL), because AWS does not
currently support Vitis AFI generation on F2.

## Quickstart (off-board)

```bash
# 1. export real Qwen weights to the FPGA binary layout
python -m qwen_fpga.export.export_weights --model Qwen/Qwen2.5-0.5B-Instruct

# 2. prove the quantized model decodes correctly
python -m qwen_fpga.reference.qref \
    --manifest qwen_fpga/weights/Qwen2.5-0.5B-Instruct/manifest.json \
    --prompt "What is the capital of France? Answer in one word."

# 3. golden vectors + C++ kernel verification (Milestone A for every shape)
python -m qwen_fpga.reference.make_golden \
    --manifest qwen_fpga/weights/Qwen2.5-0.5B-Instruct/manifest.json
make -C qwen_fpga check        # 168 cases vs golden

# 4. run the host runtime with the C++ datapath driving every GEMV
make -C qwen_fpga lib
QWEN_FPGA_BACKEND=cpp python -m qwen_fpga.webapp.server \
    --manifest qwen_fpga/weights/Qwen2.5-0.5B-Instruct/manifest.json
#   open http://localhost:8000
```

On F2, after loading the AGFI:

```bash
export QWEN_FPGA_BACKEND=xrt
export QWEN_FPGA_XCLBIN=/path/to/cl_qwen_gemv.awsxclbin
python -m qwen_fpga.webapp.server --manifest .../manifest.json   # backend=fpga
```

## Numeric contract

```
y[m] = x_scale * sum_g  w_scale[m,g] * ( sum_{n in group g} wq[m,n] * xq[n] )
```

* weights `wq`: symmetric INT4 in [-8,7], groupwise (default group=128), packed
  2/byte (column n in the low nibble when n is even);
* activations `xq`: symmetric INT8, one scale per call;
* group accumulation INT32, output accumulation FP32.

Defined once in `export/quant.py`, mirrored in `kernel/gemv_int4.hpp`, enforced by
`tests/test_qwen_fpga_quant.py` and the golden test.

## Milestone ladder

* **A** single projection on FPGA vs golden — proven in C++ here; same bins load on F2.
* **B** one full Qwen layer (projections on FPGA, glue on CPU) vs reference.
* **C** full token decode with `QWEN_FPGA_BACKEND=xrt`.
* **D** `/api/chat` answering with `backend="fpga"` + measured latency.

## Scaling up

`export_weights.py --model Qwen/Qwen2.5-1.5B-Instruct` produces the same layout at
larger dimensions; the kernel is dimension-generic (`M`, `N`, `group_size` are
runtime registers). INT4 weights for a ~2B model are ~1 GB — far under F2's 16 GiB
HBM, so capacity is never the constraint; HBM bandwidth and the GEMV schedule are.

## Files

```
export/quant.py            INT4/INT8 quant + GEMV reference math (the contract)
export/export_weights.py   Qwen checkpoint -> flat binaries + manifest.json
reference/qref.py          quantized Qwen decode (ground truth, pluggable backend)
reference/make_golden.py   real-activation golden vectors for the kernel
kernel/gemv_int4.hpp       shared datapath + AXI-Lite register map
kernel/gemv_int4_hls.cpp   Vitis HLS kernel (HBM AXI masters + AXI-Lite)
kernel/gemv_int4_capi.cpp  C ABI -> libgemv_int4 (host software backend)
kernel/test_gemv_ref.cpp   golden-vector verification harness
host/fpga_gemv.py          numpy / cpp / xrt GEMV backends
webapp/server.py           /api/chat -> FPGA-backed generation
hdk/                       run_hls.tcl, build_afi.sh, register_map.md, README
Makefile                   lib / test / check
```
