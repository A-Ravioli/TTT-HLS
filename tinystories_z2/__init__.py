"""TinyStories (GPT-Neo) inference on a PYNQ Z2 (Zynq-7020) FPGA.

This is the small-LM sibling of ``qwen_fpga``: same host/FPGA split (glue math on
the CPU, every large linear as an INT4 GEMV call), retargeted from AWS F2 + HBM +
XRT down to a Zynq-7020 with weights resident in the PS-side 512 MB DDR3 and the
GEMV kernel driven over PYNQ MMIO.

It deliberately *reuses* the numeric contract from ``qwen_fpga.export.quant`` (the
groupwise-INT4 weight / INT8-activation / INT32-accum GEMV) so the FPGA datapath,
its C++ reference, and this Python reference cannot drift apart.

Stage 1 (this code) is the off-board numeric foundation:

  * ``export_tinystories.py`` -- a TinyStories GPT-Neo checkpoint -> flat INT4 bins
    + manifest, exactly the layout the on-board host will mmap.
  * ``model.py`` -- a from-scratch GPT-Neo decode whose linears dispatch to the
    *same* INT4 GEMV the FPGA runs (``cpp`` backend) or, for architecture
    validation, to exact fp32 weights pulled straight from Hugging Face.

It is fully testable on a laptop with no FPGA and no Vivado.
"""
