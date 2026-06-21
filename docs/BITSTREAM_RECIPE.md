# NanoCoder → PYNQ-Z2 Bitstream Recipe (Linux + Vivado)

Crossing the last wall: turning the hardened NanoCoder MLP block into a real
`.bit` running on the board. Everything up to here (fit, bit-accurate csim, trained
weights) runs on the Mac; **synthesis does not** — Vivado is Linux/Windows only.

```
 Keras MLP (trained NanoCoder block, 128→512→128, ReLU)
   └─ hls4ml VivadoAccelerator backend  (board=pynq-z2, axi_stream, io_stream)
        └─ Vivado HLS  csynth      → RTL
             └─ Vivado IPI + impl  → design_1_wrapper.bit  +  design_1.hwh  +  axi_stream_driver.py
                  └─ scp to board → pynq.Overlay / NeuralNetworkOverlay → DMA inference
```

Target part: **xc7z020clg400-1** (220 DSP, 53,200 LUT, 140 BRAM). Confirmed in
hls4ml's `supported_boards.json` as `pynq-z2`. The MLP fits at ReuseFactor 596
(~220/220 DSP); for routing headroom use reuse 768 (~171 DSP).

---

## Prerequisites (any path)

- **Vivado ML 2020.1** — last version with `vivado_hls`, the hls4ml-validated combo.
  xc7z020 is covered by the **free** Vivado ML Standard / WebPACK (no license file).
- The **`burnttt` Python stack** but on Linux: `numpy<2`, `tensorflow-cpu==2.14.1`,
  `hls4ml==1.0.0`, `onnx==1.15.0`. (CPU TF is fine — Vivado does the heavy lifting.)
- The trained block weights: `artifacts/nanocoder/mlp_layer0.npz` (extracted by
  scripts/22; commit it or scp it to the host). Synth also works without weights
  (random init) — the *fit* is weight-independent, only the numerics differ.

---

## Path A — bare-metal / VM Linux host

```bash
# 1. Vivado on PATH
source /tools/Xilinx/Vivado/2020.1/settings64.sh
vivado -version            # sanity

# 2. Python env
conda create -y -n burnttt python=3.11
conda activate burnttt
pip install "numpy<2.0" "tensorflow-cpu==2.14.1" "hls4ml==1.0.0" "onnx==1.15.0"

# 3. synthesise  (≈20–40 min: csynth + IPI + place&route + bitstream)
cd TTT-HLS
python scripts/25_synth_nanocoder.py
#   → build/nanocoder_pynq/**/design_1_wrapper.bit
#     build/nanocoder_pynq/**/design_1.hwh
#     build/nanocoder_pynq/axi_stream_driver.py
```

`scripts/25_synth_nanocoder.py` builds the Keras block (trained npz if present),
converts with `backend='VivadoAccelerator', board='pynq-z2', interface='axi_stream'`,
runs a csim sanity check, then `hls_model.build(synth=True, export=True, bitfile=True)`.
Use `--no-bitfile` for a quick csynth-only resource check (~3 min, no P&R).

---

## Path B — Modal cloud (gets Vivado off your Mac)

Vivado isn't pip-installable and its installer is AMD-account-gated (~100 GB), so
stage it into a Modal Volume **once**, then runs are turnkey.

```bash
# one-time: stage a Vivado 2020.1 install tree into a Volume
modal volume create vivado-2020-1
modal volume put vivado-2020-1 /tools/Xilinx/Vivado/2020.1 /Vivado/2020.1

# every run:
modal run infra/modal_synth_nanocoder.py
#   → downloads build/nanocoder_pynq/*.bit + *.hwh back to your Mac
```

The app (`infra/modal_synth_nanocoder.py`) mounts the repo, puts the Volume's
Vivado on PATH, and runs the same `scripts/25`. Auth is already set
(`modal token set` → profile `awd-60407`). Caveat: you still need a machine that can
*produce* the Vivado install to upload — Modal hosts it, it doesn't license it.

> Heads-up: the HF token and Modal token were pasted in chat — **rotate both** when
> done. The HF token is stored only as a Modal secret / gitignored `.env`, never in
> the repo.

---

## Board prep + deployment

1. **Boot the PYNQ image.** The board is currently in **JTAG mode** — it must boot
   the PYNQ-Z2 SD image (v2.7/3.0): set boot jumper to SD, flash the image, power on,
   connect Ethernet. Default login `xilinx`/`xilinx`, Jupyter on `http://<board-ip>:9090`.
2. **Copy the overlay:**
   ```bash
   scp build/nanocoder_pynq/**/design_1_wrapper.bit \
       build/nanocoder_pynq/**/design_1.hwh \
       xilinx@<board-ip>:~/nanocoder/
   ```
3. **Run on-board:** `python scripts/04_run_fpga_demo.py` drives
   `compiler.deploy_pynq.deploy_and_run`, which loads the overlay
   (`NeuralNetworkOverlay`) and DMAs an activation vector through the block.

---

## Hardware-in-the-loop (the architecture you chose)

The host runs NanoCoder; layer-0's MLP is delegated to the board:

```
host (NanoCoder, byte-level)  ──activations──▶  PYNQ-Z2  design_1_wrapper.bit
        ▲                                          │  (hardened c_fc→ReLU→c_proj)
        └──────────── MLP result ◀─────────────────┘
                 continue the forward pass on host
```

The `pynq` accelerator backend slots into the same interface as the `hls4ml`
bit-accurate backend (scripts/23) — flip the transport from `hls_model.predict()`
(emulated, on the Mac) to `NeuralNetworkOverlay.predict()` (real fabric, on the
board). Numerics are identical by construction; only where the matmul runs changes.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `No Vivado on PATH` from scripts/25 | `source .../settings64.sh` first |
| `nnet::gelu` compile error | a GELU block leaked in — NanoCoder must use **ReLU** (it does) |
| csynth DSP > 220 | raise `--reuse` (768 → ~171 DSP) |
| `vivado_hls: command not found` | use Vivado **2020.1** (2020.2+ dropped vivado_hls → set Vitis path / `backend='Vitis'`) |
| Overlay load fails on board | `.bit` and `.hwh` must be the **same** build, same basename |
| Timing not met (WNS<0) | relax `--clock` (e.g. 12–15 ns) and re-run |
