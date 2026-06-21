"""Train NanoCoder properly on a Modal GPU with the tiny-codes dataset.

The local run (scripts/22) was MPS-bound and tiny (1500 steps / 8 MB) -> gibberish.
This runs a real GPU pretrain on the gated nampdn-ai/tiny-codes corpus (auth via the
`nanocoder-hf` Modal secret) and ships the trained checkpoint back to
artifacts/nanocoder/, where scripts/24 (serve) and scripts/23 (harden) pick it up.

    modal run infra/modal_train_nanocoder.py                      # defaults
    modal run infra/modal_train_nanocoder.py --steps 60000        # longer

NanoCoder stays vocab=256 / hidden=128 / ReLU (FPGA-hardenable) -- we are improving
the weights, not the architecture.
"""

from __future__ import annotations

import modal

app = modal.App("nanocoder-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "datasets", "numpy<2.0", "safetensors")
    .add_local_dir(
        ".", remote_path="/root/ttt", copy=True,
        ignore=["**/node_modules/**", ".git/**", "build/**", "web/**",
                "artifacts/**", "**/__pycache__/**", "*.keras"],
    )
)

vol = modal.Volume.from_name("nanocoder-ckpt", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60,
    volumes={"/ckpt": vol},
    secrets=[modal.Secret.from_name("nanocoder-hf")],
)
def train(steps: int = 6000, batch: int = 128, seq: int = 256, lr: float = 4e-4,
          warmup: int = 300, max_rows: int = 300000, cap_mb: int = 160) -> dict:
    import os
    import sys
    import time

    import numpy as np
    import torch

    sys.path.insert(0, "/root/ttt")
    from models.nanocoder.model import DEFAULT_ARCH, ByteTokenizer, build_model

    tok = ByteTokenizer()

    # --- corpus: stream an UNGATED code dataset into a byte buffer ---
    # tiny-codes is gated (needs a granted-access account, not just a token), so we
    # fall through a list of ungated code corpora and take the first that streams.
    # Generic field extraction (join all string values) handles raw-content vs
    # instruction/output schemas without per-dataset assumptions.
    from datasets import load_dataset

    DATASETS = [
        "codeparrot/codeparrot-clean-valid",          # raw GitHub Python ('content')
        "iamtarun/python_code_instructions_18k_alpaca",  # instruction/input/output
        "flytech/python-codes-25k",
    ]
    if os.environ.get("HF_TOKEN"):
        DATASETS.insert(0, "nampdn-ai/tiny-codes")  # used only if account has access

    def row_text(row) -> str:
        return "\n".join(v for v in row.values() if isinstance(v, str) and v.strip())

    cap = cap_mb * 1024 * 1024
    parts: list[bytes] = []
    total = 0
    used = None
    for name in DATASETS:
        try:
            print(f"Trying dataset: {name}")
            ds = load_dataset(name, split="train", streaming=True)
            for i, row in enumerate(ds):
                if i >= max_rows or total >= cap:
                    break
                b = row_text(row).encode("utf-8", "ignore")
                if b:
                    parts.append(b)
                    total += len(b)
            if total > 1_000_000:  # got a usable amount
                used = name
                break
        except Exception as e:  # noqa: BLE001
            print(f"  {name} unavailable ({repr(e)[:80]}); trying next")
            parts, total = [], 0
    if not used:
        raise RuntimeError("No code dataset could be streamed.")
    data = np.frombuffer(b"\n".join(parts), dtype=np.uint8).copy()
    n = data.shape[0]
    print(f"Corpus: {n/1e6:.1f} MB bytes from {len(parts)} docs (dataset={used})")

    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # GPT-Neo has no SDPA/Flash path in this transformers version, so attention is
    # eager. We make it cheap instead: short seq (O(seq^2)) + bf16 autocast (halves
    # the attention-score memory traffic on A10G tensor cores) + a big batch to
    # amortise per-step launch overhead (a 1.7M model is overhead-bound, not FLOP-bound).
    # Spot-GPU preemption-safety: resume from the Volume checkpoint if one exists, so a
    # preemption-restart continues instead of starting over.
    from transformers import GPTNeoForCausalLM

    ckpt_dir = "/ckpt/nanocoder"
    if os.path.exists(f"{ckpt_dir}/model.safetensors"):
        try:
            model = GPTNeoForCausalLM.from_pretrained(ckpt_dir).to(dev)
            print(f"Resumed from Volume checkpoint at {ckpt_dir} (preemption-safe).")
        except Exception as e:  # noqa: BLE001
            print("resume failed, fresh init:", repr(e)[:80])
            model = build_model(DEFAULT_ARCH).to(dev)
    else:
        model = build_model(DEFAULT_ARCH).to(dev)
    os.makedirs(ckpt_dir, exist_ok=True)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=warmup / steps, anneal_strategy="cos"
    )
    # Keep the whole corpus as uint8 ON the GPU (~corpus MB) and gather batches
    # there — no per-step CPU stacking, so the GPU stays fed.
    data_g = torch.from_numpy(data).to(dev)
    ar = torch.arange(seq, device=dev)

    def get_batch():
        ix = torch.randint(0, n - seq - 1, (batch,), device=dev)
        idx = ix[:, None] + ar[None, :]
        return data_g[idx].long(), data_g[idx + 1].long()

    print(f"Training {sum(p.numel() for p in model.parameters()):,} params on A10G: "
          f"{steps} steps x batch {batch} x seq {seq} = {steps*batch*seq/1e6:.0f}M tokens")
    t0 = time.time()
    for step in range(1, steps + 1):
        xb, yb = get_batch()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(input_ids=xb, labels=yb).loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == 1:
            el = time.time() - t0
            print(f"step {step:6d}/{steps}  loss={loss.item():.3f}  bits/byte={loss.item()/0.6931:.3f}  "
                  f"({step/el:.1f} st/s, {el/60:.1f} min)")
        if step % 1000 == 0:  # checkpoint to the Volume so preemption never wipes progress
            model.save_pretrained(ckpt_dir)
            vol.commit()
            print(f"  [checkpointed to Volume @ step {step}]")

    # --- sample ---
    model.eval()
    ids = torch.tensor([tok.encode("def ")], device=dev)
    with torch.no_grad():
        g = model.generate(ids, max_new_tokens=160, do_sample=True, top_k=40, temperature=0.7)
    sample = tok.decode(g[0].tolist())
    print("\n--- sample (prompt 'def ') ---\n" + sample + "\n--- end ---")

    # --- save to Volume + return bytes ---
    os.makedirs("/ckpt/nanocoder", exist_ok=True)
    model.save_pretrained("/ckpt/nanocoder")
    vol.commit()
    out = {}
    for fn in ("model.safetensors", "config.json", "generation_config.json"):
        p = f"/ckpt/nanocoder/{fn}"
        if os.path.exists(p):
            out[fn] = open(p, "rb").read()
    return {"files": out, "final_loss": float(loss.item()), "sample": sample}


@app.local_entrypoint()
def main(steps: int = 6000):
    import pathlib

    res = train.remote(steps=steps)
    dest = pathlib.Path("artifacts/nanocoder")
    dest.mkdir(parents=True, exist_ok=True)
    for fn, data in res["files"].items():
        (dest / fn).write_bytes(data)
        print(f"wrote {dest / fn} ({len(data)} bytes)")
    print(f"\nfinal loss={res['final_loss']:.3f} (bits/byte={res['final_loss']/0.6931:.3f})")
    print("sample:\n" + res["sample"][:300])
    print("\nNext: re-extract MLP weights + re-harden, restart scripts/24_serve_nanocoder.py")
