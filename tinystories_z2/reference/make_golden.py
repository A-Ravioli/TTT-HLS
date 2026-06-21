"""Dump golden W8A8 GEMV test cases from real TinyStories activations.

Runs one token through the quantized decode, capturing the *real* activation each
large linear sees, quantizes it to INT8, and records the expected output of the
FPGA-equivalent integer datapath. The C++ kernel test (and later HLS cosim) read
these and must match -- Milestone A for every projection shape.

Output (under ``--out``, default ``tinystories_z2/golden/<tag>/``):

    cases.json / cases.tsv     # index: per-case M, N, x_scale, files
    <case>.W.int8.bin          # INT8 weights [M, N]
    <case>.W.scale.bin         # fp16 per-row scales [M]
    <case>.x.int8.bin          # INT8 activation [N]
    <case>.y.f32.bin           # expected fp32 output [M]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tinystories_z2.model import NeoArch, NeoRunner, QuantWeights
from tinystories_z2.quant import gemv_int8_quantized, quantize_activation_int8


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--token", type=int, default=314, help="token id to drive activations")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    arch = NeoArch.from_manifest(manifest)
    provider = QuantWeights(args.manifest)
    runner = NeoRunner(arch, provider)

    tag = manifest["model_id"].split("/")[-1]
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "golden" / tag
    out.mkdir(parents=True, exist_ok=True)

    captured: list[tuple[str, np.ndarray]] = []
    orig_linear = provider.linear

    def recording_linear(key, x):
        captured.append((key, np.asarray(x, dtype=np.float32).copy()))
        return orig_linear(key, x)

    provider.linear = recording_linear  # type: ignore[assignment]
    _ = runner.forward_token(args.token, pos=0)  # one token through all layers + lm_head

    cases = []
    seen: set[str] = set()
    for key, x in captured:
        if key in seen:
            continue
        seen.add(key)
        name = key.replace(".", "_")
        qw = provider._qweight(key)

        xq, x_scale = quantize_activation_int8(x)
        if xq.shape[0] < qw.in_features:
            xq = np.concatenate([xq, np.zeros(qw.in_features - xq.shape[0], np.int8)])
        xq = xq[: qw.in_features]
        y = gemv_int8_quantized(qw, x)

        qw.q.astype(np.int8).tofile(out / f"{name}.W.int8.bin")
        qw.scales.astype(np.float16).tofile(out / f"{name}.W.scale.bin")
        xq.astype(np.int8).tofile(out / f"{name}.x.int8.bin")
        y.astype(np.float32).tofile(out / f"{name}.y.f32.bin")

        cases.append({"name": name, "M": int(qw.out_features), "N": int(qw.in_features),
                      "x_scale": float(x_scale)})
        print(f"  golden {name}: M={qw.out_features} N={qw.in_features}")

    (out / "cases.json").write_text(json.dumps(
        {"model_id": manifest["model_id"], "cases": cases}, indent=2))
    lines = ["# name\tM\tN\tx_scale"]
    for c in cases:
        lines.append(f"{c['name']}\t{c['M']}\t{c['N']}\t{c['x_scale']:.9g}")
    (out / "cases.tsv").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(cases)} golden cases to {out}")


if __name__ == "__main__":
    main()
