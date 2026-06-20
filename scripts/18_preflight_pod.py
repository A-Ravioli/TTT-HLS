#!/usr/bin/env python3
"""Preflight checks before provisioning an expensive Prime GPU pod.

Run locally (free):
  python scripts/18_preflight_pod.py

Optional: validate an *existing* pod without starting TTT (minimal $ if already up):
  python scripts/18_preflight_pod.py --remote

Full pipeline smoke without GPU weights (free, ~30s):
  python scripts/18_preflight_pod.py --local-pipeline
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import paths  # noqa: F401 — load .env

from glm.model_specs import DEFAULT_POD_GLM_MODEL, pod_spec_for_model
from infra import prime_pod

PASS = "ok"
WARN = "warn"
FAIL = "fail"


def _status(level: str, msg: str) -> tuple[str, str]:
    return level, msg


def check_env(model_id: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if os.environ.get("PRIME_API_KEY", "").strip():
        out.append(_status(PASS, "PRIME_API_KEY is set"))
    else:
        out.append(_status(WARN, "PRIME_API_KEY missing (needed for Prime Inference smoke)"))

    if os.environ.get("WANDB_API_KEY", "").strip():
        out.append(_status(PASS, "WANDB_API_KEY is set"))
    else:
        out.append(_status(WARN, "WANDB_API_KEY missing (wandb will be skipped)"))

    spec = pod_spec_for_model(model_id)
    out.append(
        _status(
            PASS,
            f"Model {model_id!r} -> {spec.gpu_count}x {spec.gpu_type} Unsloth={spec.use_unsloth} ({spec.notes})",
        )
    )
    return out


def check_hf_model(model_id: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    url = f"https://huggingface.co/api/models/{model_id}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            meta = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        out.append(_status(FAIL, f"HF model not found: {model_id} ({exc.code})"))
        return out
    except Exception as exc:  # noqa: BLE001
        out.append(_status(WARN, f"Could not reach Hugging Face: {exc}"))
        return out

    siblings = meta.get("siblings") or []
    total_bytes = sum(int(s.get("size") or 0) for s in siblings)
    gb = total_bytes / (1024**3)
    out.append(_status(PASS, f"HF repo exists: {meta.get('id', model_id)}"))
    out.append(_status(PASS, f"Weight files ~{gb:.0f} GB on disk (first run download)"))
    if gb > 500:
        out.append(
            _status(
                WARN,
                "Large download — budget 1–3+ hours on pod before TTT starts; ensure ~3TB disk",
            )
        )
    return out


def check_prime_offers(model_id: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    spec = pod_spec_for_model(model_id)
    offer = prime_pod.best_offer(spec.gpu_type, spec.gpu_count)
    if offer:
        price = offer.get("prices", {}).get("onDemand", "?")
        out.append(
            _status(
                PASS,
                f"Prime offer: {offer.get('gpuType')} x{offer.get('gpuCount')} "
                f"~${price}/hr ({offer.get('cloudId')})",
            )
        )
        est_8h = float(price) * 8 if isinstance(price, (int, float)) else None
        if est_8h:
            out.append(_status(PASS, f"Rough 8-round run estimate: ~${est_8h:.0f} (8hr) + download time"))
    else:
        out.append(
            _status(
                FAIL,
                f"No Prime offer for {spec.gpu_count}x {spec.gpu_type} — cannot run {model_id} on Prime today",
            )
        )
        smaller = prime_pod.best_offer(spec.gpu_type, 2) or prime_pod.best_offer("H100_80GB", 1)
        if smaller:
            p = smaller.get("prices", {}).get("onDemand")
            out.append(
                _status(
                    WARN,
                    f"Fallback available: {smaller.get('gpuType')} x{smaller.get('gpuCount')} ~${p}/hr "
                    f"(too small for GLM-5.2; use THUDM/glm-4-9b-chat-hf smoke instead)",
                )
            )
    return out


def check_ssh_key() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    pub = Path.home() / ".ssh" / "id_ed25519.pub"
    if not pub.is_file():
        out.append(_status(FAIL, f"No SSH public key at {pub}"))
        return out
    out.append(_status(PASS, f"Local SSH public key: {pub}"))
    try:
        keys = prime_pod.list_ssh_keys()
        pub_text = pub.read_text().strip()
        if any(k.get("publicKey", "").strip() == pub_text for k in keys):
            out.append(_status(PASS, f"Key registered with Prime ({len(keys)} key(s) on account)"))
        else:
            out.append(
                _status(
                    WARN,
                    "Public key not on Prime — run: python scripts/16_prime_pod.py ssh-upload",
                )
            )
    except Exception as exc:  # noqa: BLE001
        out.append(_status(WARN, f"Could not list Prime SSH keys: {exc}"))
    return out


def check_prime_inference_glm() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    key = os.environ.get("PRIME_API_KEY", "").strip()
    if not key:
        out.append(_status(WARN, "Skip Prime Inference check (no PRIME_API_KEY)"))
        return out
    try:
        req = urllib.request.Request(
            "https://api.pinference.ai/api/v1/models",
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
        models = [m.get("id", m) if isinstance(m, dict) else m for m in data.get("data", data)]
        glm = [m for m in models if "glm" in str(m).lower()]
        if "z-ai/glm-5.2" in glm:
            out.append(_status(PASS, "Prime Inference has z-ai/glm-5.2 (API authoring, not LoRA)"))
        else:
            out.append(_status(WARN, f"Prime GLM models: {', '.join(sorted(glm)[:6])}"))
    except Exception as exc:  # noqa: BLE001
        out.append(_status(WARN, f"Prime Inference API check failed: {exc}"))
    return out


def run_local_pipeline() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    env["BURN_GLM_BACKEND"] = "heuristic"
    env.pop("BURN_GLM_MODEL", None)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "07_ttt_finetune_glm.py"),
        "--rounds",
        "1",
        "--candidates-per-round",
        "1",
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if proc.returncode == 0:
            out.append(_status(PASS, "Local TTT pipeline (heuristic, 1 round) completed"))
        else:
            tail = (proc.stderr or proc.stdout or "")[-400:]
            out.append(_status(FAIL, f"Local pipeline failed (exit {proc.returncode}): {tail}"))
    except subprocess.TimeoutExpired:
        out.append(_status(FAIL, "Local pipeline timed out (>5min)"))
    except Exception as exc:  # noqa: BLE001
        out.append(_status(FAIL, f"Local pipeline error: {exc}"))
    return out


def check_remote(model_id: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    pod_env = ROOT / ".env.pod"
    if not pod_env.is_file():
        out.append(_status(WARN, "No .env.pod — skip remote check (or create pod first)"))
        return out

    spec = pod_spec_for_model(model_id)
    ssh_host = ssh_user = ssh_key = None
    for line in pod_env.read_text().splitlines():
        if line.startswith("SSH_HOST="):
            ssh_host = line.split("=", 1)[1].strip().strip("'\"")
        elif line.startswith("SSH_USER="):
            ssh_user = line.split("=", 1)[1].strip().strip("'\"")
        elif line.startswith("SSH_KEY="):
            ssh_key = os.path.expanduser(line.split("=", 1)[1].strip().strip("'\""))

    if not ssh_host:
        out.append(_status(WARN, ".env.pod has no SSH_HOST"))
        return out

    ssh_user = ssh_user or "ubuntu"
    ssh_key = ssh_key or str(Path.home() / ".ssh" / "id_ed25519")
    ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15"]
    if Path(ssh_key).is_file():
        ssh_cmd.extend(["-i", ssh_key])
    remote = (
        "echo SSH_OK; nvidia-smi -L; "
        "df -h ~ | tail -1; "
        "python3 -c \"import torch; print('cuda_devices', torch.cuda.device_count())\" 2>/dev/null || "
        "echo 'torch not installed yet'"
    )
    ssh_cmd.extend([f"{ssh_user}@{ssh_host}", remote])
    try:
        proc = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30, check=False)
        text = proc.stdout + proc.stderr
        if proc.returncode != 0:
            out.append(_status(FAIL, f"SSH failed: {text[-300:]}"))
            return out
        out.append(_status(PASS, f"SSH to {ssh_user}@{ssh_host} OK"))
        gpu_lines = [ln for ln in text.splitlines() if "GPU" in ln or "cuda_devices" in ln]
        n_gpu = len([ln for ln in gpu_lines if "GPU" in ln and "UUID" in ln])
        for ln in gpu_lines[:10]:
            out.append(_status(PASS, ln.strip()))
        if n_gpu < spec.gpu_count:
            out.append(
                _status(
                    WARN,
                    f"Pod has {n_gpu} GPU(s), model wants {spec.gpu_count} — GLM-5.2 load may OOM",
                )
            )
        else:
            out.append(_status(PASS, f"GPU count {n_gpu} >= required {spec.gpu_count}"))
    except Exception as exc:  # noqa: BLE001
        out.append(_status(FAIL, f"Remote check error: {exc}"))
    return out


def print_report(title: str, results: list[tuple[str, str]]) -> int:
    fails = sum(1 for lvl, _ in results if lvl == FAIL)
    warns = sum(1 for lvl, _ in results if lvl == WARN)
    print(f"\n=== {title} ===")
    for lvl, msg in results:
        tag = {"ok": "PASS", "warn": "WARN", "fail": "FAIL"}[lvl]
        print(f"  [{tag}] {msg}")
    return fails


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight before expensive Prime GLM TTT run")
    parser.add_argument("--model", default=os.environ.get("BURN_GLM_MODEL_POD", DEFAULT_POD_GLM_MODEL))
    parser.add_argument("--local-pipeline", action="store_true", help="Run 1-round heuristic TTT (~30s, free)")
    parser.add_argument("--remote", action="store_true", help="SSH checks on existing pod from .env.pod")
    args = parser.parse_args()

    print(f"Preflight for pod model: {args.model}")
    print("Tip: full GLM-5.2 weight load cannot be tested without GPU time; use tiers below.\n")

    total_fails = 0
    total_fails += print_report("Environment", check_env(args.model))
    total_fails += print_report("Hugging Face model", check_hf_model(args.model))
    total_fails += print_report("Prime GPU availability", check_prime_offers(args.model))
    total_fails += print_report("SSH", check_ssh_key())
    total_fails += print_report("Prime Inference (API)", check_prime_inference_glm())

    if args.local_pipeline:
        total_fails += print_report("Local TTT pipeline", run_local_pipeline())

    if args.remote:
        total_fails += print_report("Remote pod (existing)", check_remote(args.model))

    print("\n--- Free / cheap validation tiers ---")
    print("  1. This script (free)")
    print("  2. Local pipeline: --local-pipeline (free, tests compile/eval/TTT loop)")
    print("  3. Prime API GLM 5.2: BURN_GLM_BACKEND=prime python scripts/08_eval_glm_generator.py")
    print("     (pennies, not $22/hr — tests generation + parsing, not LoRA)")
    print("  4. 1×H100 smoke: BURN_GLM_MODEL_POD=THUDM/glm-4-9b-chat-hf GPU_COUNT=1 (~$4/hr)")
    print("     (real LoRA path, wrong model size but same code)")
    print("  5. 8×A100: only after 1–4 pass — bash scripts/17_prime_run_ttt.sh")

    if total_fails:
        print(f"\nPreflight: {total_fails} FAIL(s) — fix before provisioning pod.")
        return 1
    print("\nPreflight: no hard failures. Safe to proceed when you're ready to spend.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
