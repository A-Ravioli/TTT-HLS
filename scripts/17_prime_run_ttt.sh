#!/usr/bin/env bash
# Provision (optional) Prime pod, rsync repo, run TTT bootstrap on remote GPU.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KEEP_POD="${KEEP_POD:-1}"
ROUNDS="${ROUNDS:-8}"
CANDIDATES="${CANDIDATES:-3}"
# GLM-5.2-FP8: 744B MoE — needs 8 GPUs (Prime: 8×A100_80GB ~$22/hr; ideal: 8×H100)
GPU_TYPE="${GPU_TYPE:-A100_80GB}"
GPU_COUNT="${GPU_COUNT:-8}"
BURN_GLM_MODEL_POD="${BURN_GLM_MODEL_POD:-zai-org/GLM-5.2-FP8}"
REMOTE_DIR="${REMOTE_DIR:-~/TTT-HLS}"
SSH_USER="${SSH_USER:-ubuntu}"
SSH_HOST="${SSH_HOST:-}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
POD_ID="${PRIME_POD_ID:-}"
POD_ENV_FILE="${POD_ENV_FILE:-$REPO_ROOT/.env.pod}"

cd "$REPO_ROOT"

if [[ "${PREFLIGHT:-1}" == "1" && "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  echo "=== Preflight (free checks; set SKIP_PREFLIGHT=1 to skip) ==="
  python3 scripts/18_preflight_pod.py --model "$BURN_GLM_MODEL_POD" || exit 1
fi

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -f "$POD_ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$POD_ENV_FILE"
  set +a
  SSH_HOST="${SSH_HOST:-}"
  POD_ID="${PRIME_POD_ID:-${POD_ID:-}}"
fi

SSH_OPTS=(-o StrictHostKeyChecking=no -o ConnectTimeout=30)
if [[ -f "$SSH_KEY" ]]; then
  SSH_OPTS+=(-i "$SSH_KEY")
fi

ssh_cmd() {
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" "$@"
}

rsync_cmd() {
  rsync -az -e "ssh ${SSH_OPTS[*]}" \
    --exclude '.venv' --exclude 'build/' --exclude '__pycache__' --exclude '.git' \
    "$REPO_ROOT/" "${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}/"
}

if [[ -z "$SSH_HOST" && -z "$POD_ID" ]]; then
  echo "Ensuring Prime SSH key is registered..."
  python3 scripts/16_prime_pod.py ssh-upload --name cursor-burnttt 2>/dev/null || true
  echo "Creating Prime pod for ${BURN_GLM_MODEL_POD} (${GPU_COUNT}x ${GPU_TYPE})..."
  CREATE_OUT=$(python3 scripts/16_prime_pod.py create --name burnttt-glm52-ttt --model "$BURN_GLM_MODEL_POD" \
    --gpu-type "$GPU_TYPE" --gpu-count "$GPU_COUNT")
  echo "$CREATE_OUT"
  POD_ID=$(echo "$CREATE_OUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('id', d.get('podId', d.get('pod',{}).get('id',''))))")
  export PRIME_POD_ID="$POD_ID"
fi

if [[ -z "$SSH_HOST" && -n "${POD_ID:-}" ]]; then
  echo "Waiting for pod $POD_ID to become ACTIVE..."
  WAIT_OUT=$(python3 scripts/16_prime_pod.py wait "$POD_ID" --timeout 900)
  SSH_HOST=$(echo "$WAIT_OUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('ip',''))")
  SSH_USER="${SSH_USER:-ubuntu}"
fi

if [[ -z "$SSH_HOST" ]]; then
  echo "Set SSH_HOST or PRIME_POD_ID (or run create). See .env.pod.example" >&2
  exit 1
fi

if [[ -n "${POD_ID:-}" ]]; then
  python3 scripts/16_prime_pod.py write-env "$POD_ID" --host "$SSH_HOST" --user "$SSH_USER" --key "$SSH_KEY" \
    --out "$POD_ENV_FILE" 2>/dev/null || true
fi

echo "Pod: ${POD_ID:-unknown}  SSH: ${SSH_USER}@${SSH_HOST}  model: ${BURN_GLM_MODEL_POD}"
echo "Syncing repo..."
rsync_cmd

echo "Running TTT bootstrap (KEEP_POD=${KEEP_POD}, rounds=${ROUNDS}, model=${BURN_GLM_MODEL_POD})..."
ssh_cmd "cd ${REMOTE_DIR} && \
  export PRIME_API_KEY='${PRIME_API_KEY:-}' && \
  export WANDB_API_KEY='${WANDB_API_KEY:-}' && \
  export WANDB_PROJECT='${WANDB_PROJECT:-burnttt}' && \
  export WANDB_RUN_NAME='${WANDB_RUN_NAME:-prime-glm52-ttt}' && \
  export BURN_GLM_BACKEND='hf' && \
  export BURN_GLM_MODEL='${BURN_GLM_MODEL_POD}' && \
  export BURN_GLM_MAX_SEQ_LEN='${BURN_GLM_MAX_SEQ_LEN:-2048}' && \
  export BURN_TARGET_PART='${BURN_TARGET_PART:-xcvu47p-fsvh2892-2-e}' && \
  export BURN_TTT_USE_GRPO='${BURN_TTT_USE_GRPO:-0}' && \
  export BURN_TTT_USE_SFT='${BURN_TTT_USE_SFT:-1}' && \
  export BURN_TTT_USE_DPO='${BURN_TTT_USE_DPO:-0}' && \
  export BURN_TTT_STEPS_PER_ROUND='${BURN_TTT_STEPS_PER_ROUND:-4}' && \
  export BURN_TTT_LR='${BURN_TTT_LR:-1e-4}' && \
  export BURN_TTT_RUN_NAME='${BURN_TTT_RUN_NAME:-glm_ttt}' && \
  export UNSLOTH_MOE_BACKEND='grouped_mm' && \
  export ROUNDS='${ROUNDS}' && \
  export CANDIDATES='${CANDIDATES}' && \
  chmod +x scripts/prime_bootstrap.sh && \
  bash scripts/prime_bootstrap.sh"

if [[ "$KEEP_POD" != "1" && -n "${POD_ID:-}" ]]; then
  python3 scripts/16_prime_pod.py delete "$POD_ID"
  rm -f "$POD_ENV_FILE"
fi

echo "Remote TTT run complete."
