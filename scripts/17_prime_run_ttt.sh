#!/usr/bin/env bash
# Provision (optional) Prime pod, rsync repo, run TTT bootstrap on remote GPU.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KEEP_POD="${KEEP_POD:-1}"
ROUNDS="${ROUNDS:-8}"
CANDIDATES="${CANDIDATES:-3}"
GPU_TYPE="${GPU_TYPE:-H100_80GB}"
REMOTE_DIR="${REMOTE_DIR:-~/TTT-HLS}"
SSH_USER="${SSH_USER:-root}"
SSH_HOST="${SSH_HOST:-}"
POD_ID="${PRIME_POD_ID:-}"

cd "$REPO_ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "$SSH_HOST" && -z "$POD_ID" ]]; then
  echo "Creating Prime pod..."
  CREATE_OUT=$(python3 scripts/16_prime_pod.py create --gpu-type "$GPU_TYPE" --name burnttt-ttt)
  echo "$CREATE_OUT"
  POD_ID=$(echo "$CREATE_OUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('id', d.get('podId', d.get('pod',{}).get('id',''))))")
  export PRIME_POD_ID="$POD_ID"
fi

if [[ -z "$SSH_HOST" && -n "${POD_ID:-}" ]]; then
  echo "Waiting for pod $POD_ID SSH..."
  for i in $(seq 1 60); do
    STATUS=$(python3 scripts/16_prime_pod.py status "$POD_ID" 2>/dev/null || true)
    SSH_HOST=$(echo "$STATUS" | python3 -c "
import sys, json
try:
    p = json.load(sys.stdin)
    print(p.get('ip') or '')
except Exception:
    print('')
" 2>/dev/null || true)
    POD_STATUS=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || true)
    if [[ -n "$SSH_HOST" && "$POD_STATUS" == "ACTIVE" ]]; then
      SSH_USER="${SSH_USER:-ubuntu}"
      break
    fi
    sleep 10
  done
fi

if [[ -z "$SSH_HOST" ]]; then
  echo "Set SSH_HOST to the pod IP or PRIME_POD_ID with reachable SSH." >&2
  echo "Example: SSH_HOST=1.2.3.4 $0" >&2
  exit 1
fi

echo "Syncing repo to ${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}"
rsync -az --exclude '.venv' --exclude 'build/' --exclude '__pycache__' --exclude '.git' \
  "$REPO_ROOT/" "${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}/"

echo "Running bootstrap on pod..."
ssh "${SSH_USER}@${SSH_HOST}" "cd ${REMOTE_DIR} && \
  export PRIME_API_KEY='${PRIME_API_KEY:-}' && \
  export WANDB_API_KEY='${WANDB_API_KEY:-}' && \
  export WANDB_PROJECT='${WANDB_PROJECT:-burnttt}' && \
  export WANDB_RUN_NAME='${WANDB_RUN_NAME:-prime-pod-ttt-config-v1}' && \
  export BURN_GLM_BACKEND='${BURN_GLM_BACKEND:-hf}' && \
  export BURN_GLM_MODEL='${BURN_GLM_MODEL:-THUDM/glm-4-9b-chat}' && \
  export ROUNDS='${ROUNDS}' && \
  export CANDIDATES='${CANDIDATES}' && \
  chmod +x scripts/prime_bootstrap.sh && \
  bash scripts/prime_bootstrap.sh"

if [[ "$KEEP_POD" != "1" && -n "${POD_ID:-}" ]]; then
  python3 scripts/16_prime_pod.py delete "$POD_ID"
fi

echo "Remote TTT run complete."
