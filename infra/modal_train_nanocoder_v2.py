"""NanoCoder v2: small-BPE-vocab + hidden=256, for coherence (still hardens).

v1 was byte-level (vocab 256) / hidden 128 -> learned code *shape* but spelled every
token byte-by-byte, so it never got coherent (plateaued at 2.27 bits/byte). v2 fixes
the two binding limits:

  * a trained **BPE tokenizer (vocab ~4096)** on the code corpus -> parameters go to
    logic instead of spelling. This is the big coherence win, and it costs nothing on
    the FPGA: the hardened MLP block is vocab-independent.
  * **hidden 256 / intermediate 1024** (~7.5M params, ~4x v1) -> still fits the
    PYNQ-Z2: MLP block 256*1024*2 = 524,288 MACs -> ReuseFactor 2,384 on 220 DSP.

Self-contained (builds the config inline so it doesn't depend on the local module's
DEFAULT_ARCH). Trains the tokenizer + model on an A100, checkpoint/resume-safe against
spot preemption, and ships model.safetensors + config + tokenizer.json back to
artifacts/nanocoder/.

    modal run infra/modal_train_nanocoder_v2.py
"""

from __future__ import annotations

import modal

app = modal.App("nanocoder-v2-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "datasets", "tokenizers", "numpy<2.0", "safetensors")
    .add_local_dir(
        ".", remote_path="/root/ttt", copy=True,
        ignore=["**/node_modules/**", ".git/**", "build/**", "web/**",
                "artifacts/**", "**/__pycache__/**", "*.keras"],
    )
)

vol = modal.Volume.from_name("nanocoder-v2-ckpt", create_if_missing=True)


@app.function(image=image, gpu="A100", timeout=60 * 60 * 2,
              volumes={"/ckpt": vol}, secrets=[modal.Secret.from_name("nanocoder-hf")])
def train(steps: int = 6000, batch: int = 64, seq: int = 256, lr: float = 4e-4,
          warmup: int = 400, vocab_size: int = 4096, hidden: int = 256, inter: int = 1024,
          layers: int = 8, heads: int = 16, max_rows: int = 400000, cap_mb: int = 240) -> dict:
    import os
    import time

    import numpy as np
    import torch
    from datasets import load_dataset

    ckpt = "/ckpt/nanocoder"
    os.makedirs(ckpt, exist_ok=True)

    # --- 1. stream an ungated code corpus into a list of docs ---
    DATASETS = ["codeparrot/codeparrot-clean-valid",
                "iamtarun/python_code_instructions_18k_alpaca",
                "flytech/python-codes-25k"]
    docs: list[str] = []
    total = 0
    cap = cap_mb * 1024 * 1024
    used = None
    for name in DATASETS:
        try:
            print(f"Trying dataset: {name}")
            ds = load_dataset(name, split="train", streaming=True)
            for i, row in enumerate(ds):
                if i >= max_rows or total >= cap:
                    break
                t = "\n".join(v for v in row.values() if isinstance(v, str) and v.strip())
                if t:
                    docs.append(t)
                    total += len(t.encode("utf-8", "ignore"))
            if total > 1_000_000:
                used = name
                break
        except Exception as e:  # noqa: BLE001
            print(f"  {name} unavailable ({repr(e)[:70]}); next")
            docs, total = [], 0
    if not used:
        raise RuntimeError("no corpus")
    print(f"Corpus: {total/1e6:.1f} MB from {len(docs)} docs ({used})")

    # --- 2. BPE tokenizer (reuse from Volume on resume) ---
    from transformers import PreTrainedTokenizerFast

    tok_path = f"{ckpt}/tokenizer.json"
    if not os.path.exists(tok_path):
        from tokenizers import ByteLevelBPETokenizer

        bpe = ByteLevelBPETokenizer()
        bpe.train_from_iterator(docs, vocab_size=vocab_size, min_frequency=2,
                                special_tokens=["<|endoftext|>"])
        bpe.save(tok_path)
        print(f"Trained BPE tokenizer (vocab {vocab_size}) -> {tok_path}")
    else:
        print("Reusing tokenizer from Volume")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tok_path, eos_token="<|endoftext|>", bos_token="<|endoftext|>",
        pad_token="<|endoftext|>", unk_token="<|endoftext|>")
    eos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    real_vocab = tokenizer.vocab_size

    # --- 3. tokenize corpus -> one int32 stream ---
    ids: list[int] = []
    for d in docs:
        ids.extend(tokenizer.encode(d))
        ids.append(eos_id)
    data = np.asarray(ids, dtype=np.int32)
    n = data.shape[0]
    print(f"Tokens: {n/1e6:.1f}M  (vocab={real_vocab}, ~{total/max(1,n):.1f} bytes/token)")

    # --- 4. build v2 model (resume from Volume if present) ---
    from transformers import GPTNeoConfig, GPTNeoForCausalLM

    cfg = GPTNeoConfig(
        vocab_size=real_vocab, hidden_size=hidden, num_layers=layers, num_heads=heads,
        intermediate_size=inter, max_position_embeddings=512, activation_function="relu",
        window_size=256, attention_types=[[["global", "local"], layers // 2]],
        resid_dropout=0.0, embed_dropout=0.0, attention_dropout=0.0,
        bos_token_id=eos_id, eos_token_id=eos_id,
    )
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if os.path.exists(f"{ckpt}/model.safetensors"):
        try:
            model = GPTNeoForCausalLM.from_pretrained(ckpt).to(dev)
            print("Resumed model from Volume checkpoint.")
        except Exception as e:  # noqa: BLE001
            print("resume failed, fresh:", repr(e)[:70])
            model = GPTNeoForCausalLM(cfg).to(dev)
    else:
        model = GPTNeoForCausalLM(cfg).to(dev)
    model.train()
    nparam = sum(p.numel() for p in model.parameters())
    print(f"NanoCoder v2: {nparam:,} params (hidden={hidden} inter={inter} vocab={real_vocab}, relu)")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=warmup / steps, anneal_strategy="cos")
    data_g = torch.from_numpy(data).to(dev)
    ar = torch.arange(seq, device=dev)

    def get_batch():
        ix = torch.randint(0, n - seq - 1, (batch,), device=dev)
        idx = ix[:, None] + ar[None, :]
        return data_g[idx].long(), data_g[idx + 1].long()

    print(f"Training {steps} steps x batch {batch} x seq {seq} on A100")
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
            print(f"step {step:6d}/{steps}  loss={loss.item():.3f}  ({step/el:.1f} st/s, {el/60:.1f} min)")
        if step % 1000 == 0:
            model.save_pretrained(ckpt)
            vol.commit()
            print(f"  [checkpointed @ step {step}]")

    # --- 5. sample (proper code prompt via BPE) ---
    model.eval()
    prompt = "def fibonacci(n):\n"
    pin = torch.tensor([tokenizer.encode(prompt)], device=dev)
    with torch.no_grad():
        g = model.generate(pin, max_new_tokens=120, do_sample=True, top_k=40, temperature=0.6,
                           pad_token_id=eos_id)
    sample = tokenizer.decode(g[0].tolist(), skip_special_tokens=True)
    print("\n--- sample ---\n" + sample + "\n--- end ---")

    # --- 6. save + return ---
    model.save_pretrained(ckpt)
    vol.commit()
    out = {}
    for fn in ("model.safetensors", "config.json", "generation_config.json", "tokenizer.json"):
        p = f"{ckpt}/{fn}"
        if os.path.exists(p):
            out[fn] = open(p, "rb").read()
    return {"files": out, "final_loss": float(loss.item()), "sample": sample, "params": nparam}


@app.local_entrypoint()
def main(steps: int = 6000):
    import pathlib

    res = train.remote(steps=steps)
    dest = pathlib.Path("artifacts/nanocoder")
    dest.mkdir(parents=True, exist_ok=True)
    for fn, data in res["files"].items():
        (dest / fn).write_bytes(data)
        print(f"wrote {dest / fn} ({len(data)} bytes)")
    print(f"\nparams={res['params']:,}  final loss={res['final_loss']:.3f}")
    print("sample:\n" + res["sample"][:400])
    print("\nNext: re-extract MLP weights + scripts/23 re-harden, restart scripts/24 serve")
