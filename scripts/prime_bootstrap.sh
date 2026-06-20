#!/usr/bin/env bash
# Run on a Prime GPU pod after rsync of the repo.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/TTT-HLS}"
cd "$REPO_DIR"

export BURN_GLM_BACKEND="${BURN_GLM_BACKEND:-hf}"
export BURN_GLM_MODEL="${BURN_GLM_MODEL:-zai-org/GLM-5.2-FP8}"
export BURN_TARGET_PART="${BURN_TARGET_PART:-xc7z020clg400-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-burnttt}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-prime-glm52-ttt}"
export BURN_GLM_MAX_SEQ_LEN="${BURN_GLM_MAX_SEQ_LEN:-2048}"
export UNSLOTH_MOE_BACKEND="${UNSLOTH_MOE_BACKEND:-grouped_mm}"

PYTHON="${PYTHON:-python3.11}"
if ! command -v "$PYTHON" &>/dev/null; then
  PYTHON=python3
fi

"$PYTHON" -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt -r requirements-glm.txt

# Unsloth for GLM MoE (5.x, 4.7-Flash) LoRA (bf16 only; no QLoRA on MoE).
if [[ "$BURN_GLM_MODEL" == *[Gg][Ll][Mm]-5* ]] || [[ "$BURN_GLM_MODEL" == *"GLM-4.7"* ]]; then
  echo "=== Installing Unsloth for GLM-5 MoE ==="
  pip install "unsloth" || pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
fi

echo "=== GPUs ==="
nvidia-smi -L || true

echo "=== Training toy model ==="
python scripts/00_train_model.py

echo "=== TTT finetune GLM (config mode) ==="
python scripts/07_ttt_finetune_glm.py --rounds "${ROUNDS:-8}" --candidates-per-round "${CANDIDATES:-3}"

echo "=== Dataset gating tests ==="
python -m pytest tests/test_ttt_dataset.py -q

echo "=== Done ==="
