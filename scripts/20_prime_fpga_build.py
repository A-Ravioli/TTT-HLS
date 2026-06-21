#!/usr/bin/env python3
"""Build the PYNQ Z2 W8A8 overlay on a Prime Intellect node.

Reuses the repo's Prime pod plumbing (``infra/prime_pod.py`` + ``.env.pod``) but
for an FPGA *synthesis* job instead of GLM training: rsync the repo to a Linux
node, run ``tinystories_z2/hdk/prime_fpga_bootstrap.sh`` (HLS -> Vivado ->
.bit/.hwh), and rsync the overlay back to ``tinystories_z2/build/``.

Usage:
    # use the pod already in .env.pod
    python scripts/20_prime_fpga_build.py
    # or provision a fresh CPU-ish node first (cheapest GPU offer works as a box)
    python scripts/20_prime_fpga_build.py --create --gpu-type A4000 --name z2-fpga-build
    # just check whether the target node already has Vivado
    python scripts/20_prime_fpga_build.py --probe

Note: Vivado is license/account gated and not on stock Prime images. If the node
lacks it, the bootstrap prints exact install steps and exits; this script reports
that without pretending a bitstream was produced.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paths  # noqa: F401,E402  (loads .env)
from infra import prime_pod  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
POD_ENV = REPO / ".env.pod"
REMOTE_DIR = "~/TTT-HLS"


def _ssh_base(host: str, user: str, key: str | None) -> list[str]:
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no"]
    if key:
        cmd += ["-i", key]
    return cmd + [f"{user}@{host}"]


def _rsync_overlay(key: str | None, host: str, user: str) -> None:
    """Sync only tinystories_z2 sources needed for the FPGA build (~tens of MB)."""
    ssh = "ssh -o StrictHostKeyChecking=no" + (f" -i {key}" if key else "")
    src = f"{REPO}/tinystories_z2/"
    dst = f"{user}@{host}:{REMOTE_DIR}/tinystories_z2/"
    excludes = ["weights", "build", "golden", "gemv_int8_prj", "vivado_prj",
                "*.pyc", "__pycache__"]
    cmd = ["rsync", "-az", "-e", ssh]
    for e in excludes:
        cmd += ["--exclude", e]
    cmd += [src, dst]
    print(f"=== rsync overlay sources -> {user}@{host}:{REMOTE_DIR}/tinystories_z2/ ===")
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _resolve_target(args) -> tuple[str, str, str | None]:
    if args.create:
        prime_pod.ensure_ssh_key()
        offer = prime_pod.best_offer(args.gpu_type, 1)
        if offer is None:
            sys.exit(f"No Prime offer for {args.gpu_type}")
        pod = prime_pod.create_pod(offer, name=args.name)
        pod_id = pod.get("id") or pod.get("pod", {}).get("id")
        print(f"created pod {pod_id}; waiting for ACTIVE+IP ...")
        info = prime_pod.wait_for_pod(pod_id, timeout_sec=args.timeout)
        host = info["ip"]
        prime_pod.write_pod_env(POD_ENV, pod_id=pod_id, ssh_host=host,
                                ssh_user=args.user, ssh_key=args.key)
        return host, args.user, args.key
    env = prime_pod.read_pod_env(POD_ENV)
    if not env.get("SSH_HOST"):
        sys.exit("No .env.pod with SSH_HOST; pass --create to provision a node.")
    return env["SSH_HOST"], env.get("SSH_USER", "ubuntu"), env.get("SSH_KEY") or args.key


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--create", action="store_true", help="provision a fresh node")
    ap.add_argument("--gpu-type", default="A4000", help="node shape if --create")
    ap.add_argument("--name", default="z2-fpga-build")
    ap.add_argument("--user", default="ubuntu")
    ap.add_argument("--key", default=None, help="ssh private key path")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--probe", action="store_true",
                    help="only check Vivado availability on the node")
    ap.add_argument("--skip-rsync", action="store_true",
                    help="skip rsync; use sources already on the node")
    ap.add_argument("--vivado-only", action="store_true",
                    help="skip HLS; reuse gemv_int8_prj on the node")
    ap.add_argument("--jobs", type=int, default=16,
                    help="Vivado impl parallel jobs (default: 16)")
    args = ap.parse_args()

    host, user, key = _resolve_target(args)
    ssh = _ssh_base(host, user, key)

    if args.probe:
        print(f"=== probing {user}@{host} for Vivado/Vitis ===")
        subprocess.run(ssh + [
            "command -v vivado vitis_hls 2>/dev/null; "
            "ls -d /tools/Xilinx/Vivado/* /opt/Xilinx/Vivado/* 2>/dev/null; "
            "nproc; free -g | head -2; echo done"], check=False)
        return

    if not args.skip_rsync:
        _rsync_overlay(key, host, user)
    else:
        print("=== skipping rsync (--skip-rsync) ===")

    bootstrap = f"JOBS={args.jobs} "
    if args.vivado_only:
        bootstrap += "SKIP_HLS=1 "
    bootstrap += f"bash {REMOTE_DIR}/tinystories_z2/hdk/prime_fpga_bootstrap.sh"

    print("=== run FPGA build bootstrap on node ===")
    rc = subprocess.run(ssh + [bootstrap], check=False).returncode
    if rc != 0:
        sys.exit(f"\nbuild bootstrap exited {rc} (see output above; "
                 f"Vivado likely not installed on the node).")

    print("=== rsync overlay back -> tinystories_z2/build/ ===")
    (REPO / "tinystories_z2" / "build").mkdir(parents=True, exist_ok=True)
    sshopt = "ssh -o StrictHostKeyChecking=no" + (f" -i {key}" if key else "")
    subprocess.run(
        ["rsync", "-az", "-e", sshopt,
         f"{user}@{host}:{REMOTE_DIR}/tinystories_z2/build/gemv_int8.bit",
         f"{user}@{host}:{REMOTE_DIR}/tinystories_z2/build/gemv_int8.hwh",
         f"{REPO}/tinystories_z2/build/"], check=False)
    print("\nDone. Copy tinystories_z2/build/gemv_int8.{bit,hwh} to the PYNQ board.")


if __name__ == "__main__":
    main()
