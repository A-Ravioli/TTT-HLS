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

    p_status = sub.add_parser("status", help="Get pod status")
    p_status.add_argument("pod_id")

    p_pods = sub.add_parser("pods", help="List active pods")

    p_delete = sub.add_parser("delete", help="Delete a pod")
    p_delete.add_argument("pod_id")

    args = parser.parse_args()

    if args.cmd == "list":
        offers = prime_pod.list_gpu_offers(args.gpu_type, args.gpu_count)
        print(json.dumps(offers[:5], indent=2))
    elif args.cmd == "create":
        offers = prime_pod.list_gpu_offers(args.gpu_type, args.gpu_count)
        if not offers:
            print("No offers available", file=sys.stderr)
            sys.exit(1)
        result = prime_pod.create_pod(offers[0], name=args.name)
        print(json.dumps(result, indent=2))
    elif args.cmd == "status":
        pod = prime_pod.get_pod(args.pod_id)
        print(json.dumps(pod, indent=2))
    elif args.cmd == "pods":
        print(json.dumps(prime_pod.list_pods(), indent=2))
    elif args.cmd == "delete":
        print(json.dumps(prime_pod.delete_pod(args.pod_id), indent=2))


if __name__ == "__main__":
    main()
