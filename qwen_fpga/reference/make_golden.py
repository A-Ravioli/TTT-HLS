"""Dump golden GEMV test cases from real Qwen activations.

For each large linear op in layer 0 (plus the final LM head) this captures the
*real* activation the op sees during a forward pass, quantizes it to INT8, and
records the expected output from the FPGA-equivalent integer datapath. The C++
functional reference and (later) the HLS cosim read these files and must match.

Output (under ``--out``, default ``qwen_fpga/golden/<tag>/``):

    cases.json                 # index: per-case M, N, group_size, x_scale, files
    <case>.W.int4.bin          # packed weights (copied from export)
    <case>.W.scale.bin         # fp16 group scales
    <case>.x.int8.bin          # INT8 activation codes (length N)
    <case>.y.f32.bin           # expected fp32 output (length M)

This realizes Milestone A (q_proj) and gives the C++ kernel coverage of every
projection shape it will run during real decode.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qwen_fpga.export.quant import gemv_int4_quantized, quantize_activation_int8
from qwen_fpga.reference.qref import QwenQuantRunner, load


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--token", type=int, default=9707, help="token id to drive activations")
    args = ap.parse_args()

    loaded = load(args.manifest)
    runner = QwenQuantRunner(loaded)
    tag = loaded.manifest["model_id"].split("/")[-1]
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "golden" / tag
    out.mkdir(parents=True, exist_ok=True)

    captured: list[tuple[str, dict, np.ndarray]] = []
    orig_gemv = runner._gemv

    def recording_gemv(spec, x, _name_box=[0]):
        captured.append((spec["weight"], spec, np.asarray(x, dtype=np.float32).copy()))
        return orig_gemv(spec, x)

    runner._gemv = recording_gemv  # type: ignore[assignment]

    # one full forward of a single token through all layers + lm_head
    _ = runner.forward_token(args.token, pos=0)

    cases = []
    seen = set()
    for weight_file, spec, x in captured:
        name = weight_file.replace(".int4.bin", "").replace(".", "_")
        if name in seen:
            continue
        seen.add(name)

        xq, x_scale = quantize_activation_int8(x)
        qw = runner.L.qweight(spec)
        # pad xq to in_features (export padded N to even / group multiple)
        if xq.shape[0] < qw.in_features:
            xq = np.concatenate([xq, np.zeros(qw.in_features - xq.shape[0], np.int8)])
        xq = xq[: qw.in_features]
        y = gemv_int4_quantized(qw, x)

        (out / f"{name}.W.int4.bin").write_bytes(qw.packed.tobytes())
        qw.scales.astype(np.float16).tofile(out / f"{name}.W.scale.bin")
        xq.astype(np.int8).tofile(out / f"{name}.x.int8.bin")
        y.astype(np.float32).tofile(out / f"{name}.y.f32.bin")

        cases.append({
            "name": name,
            "M": int(qw.out_features),
            "N": int(qw.in_features),
            "group_size": int(qw.group_size),
            "num_groups": int(qw.num_groups),
            "x_scale": float(x_scale),
            "files": {"W": f"{name}.W.int4.bin", "scale": f"{name}.W.scale.bin",
                      "x": f"{name}.x.int8.bin", "y": f"{name}.y.f32.bin"},
        })
        print(f"  golden {name}: M={qw.out_features} N={qw.in_features} groups={qw.num_groups}")

    (out / "cases.json").write_text(json.dumps({"model_id": loaded.manifest["model_id"],
                                                "cases": cases}, indent=2))
    # flat TSV index so the C++ test harness needs no JSON parser
    lines = ["# name\tM\tN\tgroup_size\tnum_groups\tx_scale"]
    for c in cases:
        lines.append(f"{c['name']}\t{c['M']}\t{c['N']}\t{c['group_size']}"
                     f"\t{c['num_groups']}\t{c['x_scale']:.9g}")
    (out / "cases.tsv").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(cases)} golden cases to {out}")


if __name__ == "__main__":
    main()
