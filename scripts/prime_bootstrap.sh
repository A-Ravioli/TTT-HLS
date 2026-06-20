#!/usr/bin/env bash
# Run on a Prime GPU pod after rsync of the repo.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/TTT-HLS}"
cd "$REPO_DIR"

export BURN_GLM_BACKEND="${BURN_GLM_BACKEND:-hf}"
export BURN_GLM_MODEL="${BURN_GLM_MODEL:-THUDM/glm-4-9b-chat}"
export BURN_TARGET_PART="${BURN_TARGET_PART:-xc7z020clg400-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-burnttt}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-prime-pod-ttt-config-v1}"

PYTHON="${PYTHON:-python3.11}"
if ! command -v "$PYTHON" &>/dev/null; then
  PYTHON=python3
fi

"$PYTHON" -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt -r requirements-glm.txt

echo "=== Training toy model ==="
python scripts/00_train_model.py

echo "=== TTT finetune GLM (config mode) ==="
python scripts/07_ttt_finetune_glm.py --rounds "${ROUNDS:-8}" --candidates-per-round "${CANDIDATES:-3}"

echo "=== Dataset gating tests ==="
python -m pytest tests/test_ttt_dataset.py -q

echo "=== Done ==="
