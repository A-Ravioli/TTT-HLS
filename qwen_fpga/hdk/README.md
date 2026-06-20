# AWS F2 HDK build for the Qwen GEMV accelerator

This directory turns the HLS GEMV datapath into a deployable Amazon FPGA Image
(AFI) for an F2 instance, using the **supported HDK/Vivado path** (not Vitis AFI
generation, which AWS does not currently support on F2).

```
kernel/gemv_int4_hls.cpp
      │  vitis_hls -f hdk/run_hls.tcl        (csynth + export_design -> RTL IP)
      ▼
gemv_int4 RTL IP
      │  instantiate inside CL custom logic (cl_qwen_gemv, derived from CL_MEM_PERF)
      ▼
aws_build_dcp_from_cl.py                      (Vivado synth + P&R -> DCP)
      ▼
aws ec2 create-fpga-image                     (DCP -> AFI/AGFI)
      ▼
sudo fpga-load-local-image -S 0 -I <AGFI>     (load onto slot 0)
      ▼
host/fpga_gemv.py  QWEN_FPGA_BACKEND=xrt      (drive it; weights live in HBM)
```

## Why CL_MEM_PERF as the base

`CL_MEM_PERF` is the AWS HDK example that already wires up all 32 HBM AXI3
channels and exercises ~460 GB/s. We keep its HBM/shell plumbing and replace the
traffic generator with the `gemv_int4` IP:

* the AXI-Lite slave (OCL/SDA) maps to the [register map](register_map.md);
* the IP's `gmem_w` master is attached to HBM so packed INT4 weights stream from
  HBM (resident across tokens), with `gmem_scale`, `gmem_x`, `gmem_y` as the
  scale/activation/result ports.

## Steps on the F2 instance

```bash
# 0. toolchain
git clone https://github.com/aws/aws-fpga.git ~/aws-fpga
cd ~/aws-fpga && git checkout f2
source hdk_setup.sh && source sdk_setup.sh

# 1. HLS -> RTL IP   (from this repo's qwen_fpga/ dir)
vitis_hls -f hdk/run_hls.tcl

# 2. drop the IP into a cl_qwen_gemv custom-logic dir cloned from cl_mem_perf,
#    instantiate gemv_int4, connect AXI-Lite regs + HBM masters (see above).

# 3. build DCP -> AFI -> wait -> load
export AWS_FPGA_REPO_DIR=~/aws-fpga
export CL_DIR=~/aws-fpga/hdk/cl/examples/cl_qwen_gemv
export S3_BUCKET=my-fpga-bucket
bash hdk/build_afi.sh
```

## Bring-up order (matches the milestones)

1. **Milestone A** — load AFI, run a single `gemv_int4` for `layers.0.attn.q_proj`
   with the golden INT8 activation; compare against `golden/.../layers_0_attn_q_proj.y.f32.bin`.
2. **Milestone B** — host runs one full Qwen layer (norm/RoPE/attn/SiLU on CPU,
   all projections on FPGA); compare against the Python reference layer output.
3. **Milestone C** — full token decode with `QWEN_FPGA_BACKEND=xrt`.
4. **Milestone D** — webapp `/api/chat` answers with `backend="fpga"` + latency.

Before the AFI exists, `QWEN_FPGA_BACKEND=cpp` runs the identical datapath on the
CPU so the entire host + webapp is testable, and the golden test proves the
numerics the FPGA must reproduce.
