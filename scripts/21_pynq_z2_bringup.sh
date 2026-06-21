#!/usr/bin/env bash
# Deploy the Z2 overlay + TinyStories runtime to a PYNQ board and run Stage 3.
#
# Prereqs on the board: PYNQ image (2.7+), Python venv with numpy + transformers.
# Connect the board (USB-Ethernet default 192.168.2.99, user xilinx / password xilinx).
#
# Usage:
#   PYNQ_HOST=192.168.2.99 bash scripts/21_pynq_z2_bringup.sh
#   PYNQ_HOST=192.168.2.99 PYNQ_KEY=~/.ssh/id_ed25519 bash scripts/21_pynq_z2_bringup.sh
#
# Env:
#   PYNQ_HOST   board IP or hostname (required)
#   PYNQ_USER   SSH user (default: xilinx)
#   PYNQ_KEY    SSH private key (optional)
#   REMOTE_DIR  install root on board (default: ~/TTT-HLS)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYNQ_HOST="${PYNQ_HOST:?set PYNQ_HOST to the board IP, e.g. 192.168.2.99}"
PYNQ_USER="${PYNQ_USER:-xilinx}"
PYNQ_KEY="${PYNQ_KEY:-}"
REMOTE_DIR="${REMOTE_DIR:-~/TTT-HLS}"
MANIFEST="${MANIFEST:-tinystories_z2/weights/TinyStories-1M/manifest.json}"

SSH_OPTS=(-o StrictHostKeyChecking=no -o ConnectTimeout=10)
[[ -n "$PYNQ_KEY" ]] && SSH_OPTS+=(-i "$PYNQ_KEY)
RSYNC_SSH="ssh ${SSH_OPTS[*]}"

echo "=== probe ${PYNQ_USER}@${PYNQ_HOST} ==="
ssh "${SSH_OPTS[@]}" "${PYNQ_USER}@${PYNQ_HOST}" 'python3 -c "import pynq; print(\"pynq\", pynq.__version__)"'

echo "=== sync overlay (.bit + .hwh) ==="
ssh "${SSH_OPTS[@]}" "${PYNQ_USER}@${PYNQ_HOST}" "mkdir -p ${REMOTE_DIR}/tinystories_z2/build"
rsync -az -e "$RSYNC_SSH" \
  "$ROOT/tinystories_z2/build/gemv_int8.bit" \
  "$ROOT/tinystories_z2/build/gemv_int8.hwh" \
  "${PYNQ_USER}@${PYNQ_HOST}:${REMOTE_DIR}/tinystories_z2/build/"

echo "=== sync tinystories_z2 runtime (no golden/weights bulk) ==="
rsync -az -e "$RSYNC_SSH" \
  --exclude golden --exclude '__pycache__' --exclude '*.pyc' \
  "$ROOT/tinystories_z2/" \
  "${PYNQ_USER}@${PYNQ_HOST}:${REMOTE_DIR}/tinystories_z2/"

echo "=== sync repo root modules tinystories_z2 needs ==="
ssh "${SSH_OPTS[@]}" "${PYNQ_USER}@${PYNQ_HOST}" "mkdir -p ${REMOTE_DIR}"
rsync -az -e "$RSYNC_SSH" \
  "$ROOT/paths.py" \
  "${PYNQ_USER}@${PYNQ_HOST}:${REMOTE_DIR}/"

echo "=== run on-board generate (backend=pynq) ==="
ssh "${SSH_OPTS[@]}" "${PYNQ_USER}@${PYNQ_HOST}" bash -s <<REMOTE
set -euo pipefail
cd ${REMOTE_DIR}
export TS_Z2_OVERLAY=${REMOTE_DIR}/tinystories_z2/build/gemv_int8.bit
export TS_Z2_IP=gemv_int8_0
export PYTHONPATH=${REMOTE_DIR}:\${PYTHONPATH:-}
python3 -m tinystories_z2.generate \
  --manifest ${REMOTE_DIR}/${MANIFEST} \
  --prompt "Once upon a time, there was a robot who" \
  --max-new 40 \
  --backend pynq
REMOTE

echo "=== done ==="
