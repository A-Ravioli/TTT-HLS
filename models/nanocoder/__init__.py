"""NanoCoder: a hardware-hardened tiny coding model for the PYNQ-Z2.

Where TinyStories-1M proved a *story* block fits a $100 FPGA, NanoCoder is the
*code* analogue, co-designed from the fabric up. Two deliberate hardware choices
make it harden where an off-the-shelf coder never could:

  1. **Byte-level vocabulary (vocab=256).** Standard code tokenizers carry a
     ~50K-row embedding table. At hidden=128 that is 6.4M params (~6.4 MB) -- 10x
     the Zynq-7020's ~630 KB of BRAM, so it would have to stream from DRAM. Raw
     byte tokenization collapses the embedding to 256x128 = 33K params (~32 KB),
     freeing essentially all on-chip BRAM for the transformer logic. No tokenizer
     to train, no OOV, and Python/C are handled identically.

  2. **ReLU MLP (not GELU).** hls4ml's hardware activation library has no GELU, so
     a GPT-Neo gelu_new MLP will not synthesize. NanoCoder uses ReLU, which maps
     to a DSP-free comparator on fabric and compiles bit-accurately today.

Backbone: GPT-Neo, hidden=128, intermediate=512, 8 layers, 16 heads -- the
TinyStories-3M shape. The MLP block (128 -> 512 -> 128 = 131,072 MACs) targets the
PYNQ-Z2's 220 DSPs at ReuseFactor >= 596.

:mod:`models.nanocoder.model`   -- byte tokenizer + GPT-Neo config/model builders.
:mod:`models.nanocoder.harden`  -- the MLP sub-block as a compilable Keras model.
"""
