"""Run the NanoCoder bitstream synthesis on Modal (cloud Linux + Vivado).

This is the cloud arm of docs/BITSTREAM_RECIPE.md. It gets Vivado off your Mac
(which can't run it) and onto a Modal Linux x86 container.

THE ONE PREREQUISITE -- Vivado is not pip-installable and its installer is
AMD-account-gated (~100 GB), so you must stage a Vivado install into a Modal Volume
ONCE. xc7z020 (PYNQ-Z2) is covered by the FREE Vivado ML Standard / WebPACK edition.

  # one-time, from a Linux box (or after downloading the installer):
  modal volume create vivado-2020-1
  #  install Vivado 2020.1 to a local dir, then upload the tree:
  modal volume put vivado-2020-1 /tools/Xilinx/Vivado/2020.1 /Vivado/2020.1

Then:
  modal run infra/modal_synth_nanocoder.py     # synthesises, writes build/nanocoder_pynq/*.bit locally

The HF token is read from a Modal secret (see docs); not needed for synth itself,
only if you re-train in the cloud.
"""

from __future__ import annotations

import modal

app = modal.App("nanocoder-synth")

# Vivado lives in a Volume you stage once (see module docstring).
vivado_vol = modal.Volume.from_name("vivado-2020-1", create_if_missing=True)

# Vivado 2020.1 needs these legacy libs on Debian; tensorflow-cpu + hls4ml drive the flow.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libtinfo5", "libncurses5", "libx11-6", "libxrender1",
                 "libxtst6", "libxi6", "g++", "make", "locales")
    .pip_install("numpy<2.0", "tensorflow-cpu==2.14.1", "hls4ml==1.0.0", "onnx==1.15.0")
    # ship the repo so models/nanocoder + scripts/25 are importable in the container
    .add_local_dir(".", remote_path="/root/ttt", copy=True,
                   ignore=["build", "node_modules", ".git", "web/node_modules", "artifacts/*.keras"])
)

VIVADO_BIN = "/vivado/Vivado/2020.1/bin"


@app.function(image=image, volumes={"/vivado": vivado_vol}, timeout=60 * 60 * 2,
              cpu=8.0, memory=32768)
def synth() -> dict:
    import os
    import subprocess

    env = dict(os.environ)
    env["PATH"] = f"{VIVADO_BIN}:" + env.get("PATH", "")
    env["XILINX_VIVADO"] = "/vivado/Vivado/2020.1"

    # run the same synth script we use on a bare-metal Linux host
    proc = subprocess.run(
        ["bash", "-lc", "cd /root/ttt && python scripts/25_synth_nanocoder.py"],
        env=env, capture_output=True, text=True,
    )
    print(proc.stdout[-4000:])
    if proc.returncode != 0:
        print("STDERR:", proc.stderr[-4000:])

    # collect the artifacts to ship back to the caller
    import glob

    out = {}
    for pat, key in [("**/*.bit", "bit"), ("**/*.hwh", "hwh")]:
        hits = sorted(glob.glob(f"/root/ttt/build/nanocoder_pynq/{pat}", recursive=True))
        if hits:
            with open(hits[0], "rb") as fh:
                out[key] = (os.path.basename(hits[0]), fh.read())
    out["rc"] = proc.returncode
    return out


@app.local_entrypoint()
def main():
    import pathlib

    res = synth.remote()
    dest = pathlib.Path("build/nanocoder_pynq")
    dest.mkdir(parents=True, exist_ok=True)
    for key in ("bit", "hwh"):
        if key in res:
            name, data = res[key]
            (dest / name).write_bytes(data)
            print(f"wrote {dest / name} ({len(data)} bytes)")
    if "bit" not in res:
        print(f"No bitstream produced (rc={res.get('rc')}). Check the Vivado log above / the Volume staging.")
    else:
        print("\nNext: scp build/nanocoder_pynq/*.bit *.hwh to the PYNQ-Z2 and run scripts/04_run_fpga_demo.py")
