# Installing Vivado on the Prime Intellect pod (for the Z2 bitstream build)

Target: the existing pod (Ubuntu 22.04 x86-64, 26 cores, 221 GB RAM, 2.7 TB disk).
Goal: `vivado` + `vitis_hls` on PATH so `scripts/20_prime_fpga_build.py` can run
HLS → block design → `.bit`/`.hwh`.

**Edition / license:** the XC7Z020 (PYNQ Z2) is covered by the **free Vivado ML
Standard** edition — **no license file needed**. Match the version to your board
image: PYNQ-Z2 v2.7 → **2020.2**; PYNQ 3.0 → **2022.1** (recommended below).

SSH in first: `ssh ubuntu@192.222.52.183`

---

## 1. OS prerequisites (Ubuntu 22.04)

Vivado needs some legacy libs that 22.04 doesn't install by default:

```bash
sudo apt-get update
sudo apt-get install -y \
  libtinfo5 libncurses5 libx11-6 libxext6 libxrender1 libxtst6 libxi6 \
  libfontconfig1 libgtk2.0-0 ocl-icd-libopencl1 \
  build-essential default-jre unzip net-tools locales graphviz
sudo locale-gen en_US.UTF-8
```

(If you have no `sudo`, install Vivado under `$HOME/Xilinx` in step 3 — the build
bootstrap already searches there.)

## 2. Get the AMD Unified Installer onto the pod

The download requires an AMD account login, so grab the **Linux Self Extracting
Web Installer** in your browser, then copy it up. From your laptop:

```bash
# AMD downloads -> Vivado 2022.1 -> "Linux Self Extracting Web Installer" (~300 MB)
scp ~/Downloads/FPGAs_AdaptiveSoCs_Unified_*_Lin64.bin ubuntu@192.222.52.183:~/
```

## 3. Authenticate + install (batch, Zynq-7000 only to stay small)

On the pod:

```bash
chmod +x ~/FPGAs_AdaptiveSoCs_Unified_*_Lin64.bin
~/FPGAs_AdaptiveSoCs_Unified_*_Lin64.bin --keep --noexec --target ~/xsetup
cd ~/xsetup

# one-time: cache an auth token from your AMD account (prompts email/password)
./xsetup -b AuthTokenGen

# generate an install config, then trim it to just what we need
./xsetup -b ConfigGen
#   choose: 1) Vivado  -> 2) Vivado ML Standard
#   writes ~/.Xilinx/install_config.txt
```

Edit `~/.Xilinx/install_config.txt` to **cut size/time drastically** — keep only
Zynq-7000 and the tools we use:

```
Modules=Zynq-7000:1,Artix-7:0,Kintex-7:0,Virtex-7:0,Spartan-7:0, ... (set others to 0)
InstallOptions=...            # leave defaults
```

Then install (to `/tools/Xilinx`, or `$HOME/Xilinx` without sudo):

```bash
sudo mkdir -p /tools/Xilinx && sudo chown "$USER" /tools/Xilinx
./xsetup --agree XilinxEULA,3rdPartyEULA,WebTalkTerms \
         --batch Install \
         --config ~/.Xilinx/install_config.txt \
         --location /tools/Xilinx
```

Expect ~20–60 min (download + install) on this node.

## 4. Verify

```bash
source /tools/Xilinx/Vivado/2022.1/settings64.sh
vivado -version && vitis_hls -version
```

## 5. (Optional) PYNQ-Z2 board files

Not required — `hdk/build_bd.tcl` falls back to the raw part `xc7z020clg400-1`. For
the board preset, drop the TUL pynq-z2 board files into
`/tools/Xilinx/Vivado/2022.1/data/boards/board_files/` and set
`BOARD_PART=tul.com.tw:pynq-z2:part0:1.0`.

## 6. Build the overlay

From your laptop (the bootstrap auto-sources `/tools/Xilinx/Vivado/*` or
`$HOME/Xilinx/Vivado/*`):

```bash
python scripts/20_prime_fpga_build.py --probe   # should now show vivado + vitis_hls
python scripts/20_prime_fpga_build.py           # HLS -> Vivado -> pulls build/gemv_int8.{bit,hwh}
```

Then copy `tinystories_z2/build/gemv_int8.{bit,hwh}` to the PYNQ board and run
`python -m tinystories_z2.generate --backend pynq ...` (Stage 3).
