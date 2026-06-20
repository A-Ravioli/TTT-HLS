"""Generate text from the exported TinyStories model via the W8A8 datapath.

This is the host-side decode loop. Off-board it runs the numpy W8A8 reference
(bit-exact integer arithmetic the FPGA reproduces). On the PYNQ Z2 it will run
the same loop with ``--backend pynq`` once the overlay exists (Stage 3): only the
GEMV dispatch changes, not this schedule.

    python -m tinystories_z2.generate \
        --manifest tinystories_z2/weights/TinyStories-1M/manifest.json \
        --prompt "Once upon a time" --max-new 60
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tinystories_z2.model import NeoArch, NeoRunner, QuantWeights


def make_backend(name: str):
    """Select the GEMV backend. ``numpy`` is always available (FPGA-equivalent).

    ``pynq`` (Stage 3) drives the real Zynq-7020 overlay; importing it off-board
    fails, so we fall back to the numpy reference with a clear note.
    """
    name = name.lower()
    if name == "numpy":
        return None  # NeoRunner uses gemv_int8_quantized directly
    if name in ("pynq", "fpga"):
        try:
            from tinystories_z2.host.pynq_gemv import PynqGemv  # noqa: F401

            return PynqGemv()
        except Exception as exc:  # noqa: BLE001
            print(f"[generate] pynq backend unavailable ({exc}); using numpy reference")
            return None
    raise ValueError(f"unknown backend {name!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="numpy", choices=["numpy", "pynq", "fpga"])
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    arch = NeoArch.from_manifest(manifest)
    backend = make_backend(args.backend)
    runner = NeoRunner(arch, QuantWeights(args.manifest, backend=backend))

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(manifest["model_id"])
    ids = [int(t) for t in tok(args.prompt).input_ids]

    t0 = time.perf_counter()
    out = runner.generate(ids, max_new=args.max_new, temperature=args.temperature,
                          top_k=args.top_k, seed=args.seed)
    dt = time.perf_counter() - t0

    print(args.prompt + tok.decode(out))
    print(f"\n[{len(out)} tokens in {dt:.2f}s = {len(out)/dt:.1f} tok/s "
          f"on backend={args.backend}]")


if __name__ == "__main__":
    main()
