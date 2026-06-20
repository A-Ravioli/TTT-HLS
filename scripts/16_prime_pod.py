#!/usr/bin/env python3
"""CLI for Prime Intellect GPU pod management."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paths  # noqa: F401 — load .env

from infra import prime_pod


def main():
    parser = argparse.ArgumentParser(description="Manage Prime Intellect GPU pods")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List GPU availability offers")
    p_list.add_argument("--gpu-type", default="H100_80GB")
    p_list.add_argument("--gpu-count", type=int, default=1)

    p_create = sub.add_parser("create", help="Create a pod from best offer")
    p_create.add_argument("--name", default="burnttt-ttt")
    p_create.add_argument("--gpu-type", default="H100_80GB")
    p_create.add_argument("--gpu-count", type=int, default=1)
    p_create.add_argument(
        "--model",
        default=None,
        help="HF model id — auto-picks GPU count (e.g. zai-org/GLM-5.2-FP8 -> 8x A100)",
    )

    p_status = sub.add_parser("status", help="Get pod status")
    p_status.add_argument("pod_id")

    p_wait = sub.add_parser("wait", help="Wait until pod is ACTIVE with an IP")
    p_wait.add_argument("pod_id")
    p_wait.add_argument("--timeout", type=int, default=600)

    p_pods = sub.add_parser("pods", help="List active pods")

    p_delete = sub.add_parser("delete", help="Delete a pod")
    p_delete.add_argument("pod_id")

    p_ssh = sub.add_parser("ssh-upload", help="Upload local SSH public key to Prime")
    p_ssh.add_argument("--name", default="cursor-burnttt")
    p_ssh.add_argument("--key", type=Path, default=None, help="Path to .pub (default: ~/.ssh/id_ed25519.pub)")

    p_env = sub.add_parser("write-env", help="Write .env.pod with pod SSH connection info")
    p_env.add_argument("pod_id")
    p_env.add_argument("--host", required=True)
    p_env.add_argument("--user", default="ubuntu")
    p_env.add_argument("--key", default=None)
    p_env.add_argument("--out", type=Path, default=Path(".env.pod"))

    args = parser.parse_args()

    if args.cmd == "list":
        offers = prime_pod.list_gpu_offers(args.gpu_type, args.gpu_count)
        print(json.dumps(offers[:5], indent=2))
    elif args.cmd == "create":
        if args.model:
            result = prime_pod.create_pod_for_model(args.model, name=args.name)
        else:
            offers = prime_pod.list_gpu_offers(args.gpu_type, args.gpu_count)
            if not offers:
                print("No offers available", file=sys.stderr)
                sys.exit(1)
            result = prime_pod.create_pod(offers[0], name=args.name)
        print(json.dumps(result, indent=2))
    elif args.cmd == "status":
        pod = prime_pod.get_pod(args.pod_id)
        print(json.dumps(pod, indent=2))
    elif args.cmd == "wait":
        pod = prime_pod.wait_for_pod(args.pod_id, timeout_sec=args.timeout)
        print(json.dumps(pod, indent=2))
    elif args.cmd == "pods":
        print(json.dumps(prime_pod.list_pods(), indent=2))
    elif args.cmd == "delete":
        print(json.dumps(prime_pod.delete_pod(args.pod_id), indent=2))
    elif args.cmd == "ssh-upload":
        pub_path = args.key or (Path.home() / ".ssh" / "id_ed25519.pub")
        if not pub_path.is_file():
            print(f"Missing public key: {pub_path}", file=sys.stderr)
            sys.exit(1)
        key_id = prime_pod.ensure_ssh_key(args.name, pub_path)
        print(json.dumps({"id": key_id, "public_key": str(pub_path)}, indent=2))
    elif args.cmd == "write-env":
        prime_pod.write_pod_env(
            args.out,
            pod_id=args.pod_id,
            ssh_host=args.host,
            ssh_user=args.user,
            ssh_key=args.key,
        )
        print(json.dumps({"written": str(args.out.resolve())}, indent=2))


if __name__ == "__main__":
    main()
