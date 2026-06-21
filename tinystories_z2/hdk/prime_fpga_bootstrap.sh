#!/usr/bin/env bash
# Run on a Prime Intellect (or any Linux x86) node to build the Z2 overlay.
#
# Builds the W8A8 GEMV: HLS C++ -> RTL IP (vitis_hls) -> Zynq7 block design ->
# .bit + .hwh (vivado). Emits tinystories_z2/build/gemv_int8.{bit,hwh}.
#
# Vivado/Vitis are NOT on stock Prime images and are license/account gated, so
# this expects an existing install. Point it at one of:
#   XILINX_VIVADO=/tools/Xilinx/Vivado/2022.1   (then settings64.sh is sourced)
# or have `vivado` + `vitis_hls` already on PATH. If neither is found it prints
# the exact install steps and exits non-zero (no silent fallback).
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/TTT-HLS}"
cd "$REPO_DIR/tinystories_z2"

# --- locate the Xilinx toolchain -------------------------------------------
if ! command -v vitis_hls &>/dev/null || ! command -v vivado &>/dev/null; then
  for base in "${XILINX_VIVADO:-}" /tools/Xilinx/Vivado/* /tools/Xilinx/*/Vivado \
              /opt/Xilinx/Vivado/* /opt/Xilinx/*/Vivado \
              "$HOME"/Xilinx/Vivado/* "$HOME"/Xilinx/*/Vivado \
              "$HOME"/tools/Xilinx/Vivado/* "$HOME"/tools/Xilinx/*/Vivado; do
    [ -n "$base" ] && [ -f "$base/settings64.sh" ] && source "$base/settings64.sh"
  done
  for base in /tools/Xilinx/Vitis/* /tools/Xilinx/*/Vitis /opt/Xilinx/Vitis/* \
              "$HOME"/Xilinx/Vitis/* "$HOME"/Xilinx/*/Vitis; do
    [ -f "$base/settings64.sh" ] && source "$base/settings64.sh"
  done
fi
if ! command -v vitis_hls &>/dev/null || ! command -v vivado &>/dev/null; then
  cat >&2 <<'EOF'
[bootstrap] Vivado/Vitis HLS not found on this node.

Vivado is required to build the bitstream and is gated behind a free AMD account
(the XC7Z020 is covered by the no-cost Vivado ML Standard edition). On the pod:

  1. Download the AMD Unified Installer (Linux) with your account, or upload the
     installer tarball to the pod.
  2. Install Vivado + Vitis HLS (2020.2 to match a PYNQ-Z2 v2.7 image, or 2022.1
     for PYNQ 3.0):
       ./xsetup --agree XilinxEULA,3rdPartyEULA --batch Install \
                --edition "Vivado ML Standard" --location /tools/Xilinx
  3. Re-run this script with XILINX_VIVADO=/tools/Xilinx/Vivado/<version>.

Nothing was built.
EOF
  exit 3
fi

echo "=== vitis_hls: $(command -v vitis_hls) ==="
echo "=== vivado:    $(command -v vivado)    ==="

JOBS="${JOBS:-16}"

# --- HLS C++ -> RTL IP ------------------------------------------------------
if [[ "${SKIP_HLS:-0}" == "1" ]]; then
  echo "=== [1/2] HLS synth skipped (SKIP_HLS=1; reusing gemv_int8_prj) ==="
  [[ -d gemv_int8_prj/sol1/impl/ip ]] || {
    echo "[bootstrap] SKIP_HLS set but gemv_int8_prj/sol1/impl/ip missing" >&2
    exit 4
  }
else
  echo "=== [1/2] HLS synth (W8A8 GEMV -> RTL IP) ==="
  vitis_hls -f hdk/run_hls.tcl
fi

# --- block design -> bitstream ---------------------------------------------
echo "=== [2/2] Vivado block design -> bitstream (JOBS=$JOBS) ==="
export JOBS
vivado -mode batch -source hdk/build_bd.tcl

echo "=== artifacts ==="
ls -la build/gemv_int8.bit build/gemv_int8.hwh
echo "Copy these to the PYNQ board and run:  generate.py --backend pynq"
