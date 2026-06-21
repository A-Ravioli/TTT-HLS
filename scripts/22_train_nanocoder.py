#!/usr/bin/env python
"""Train NanoCoder byte-level on a code corpus.

NanoCoder (vocab=256, hidden=128, ReLU MLP -- see models/nanocoder) trained as a
plain next-byte language model. Corpus priority:
  1. nampdn-ai/tiny-codes  (if HF_TOKEN is set -- it is a gated dataset), else
  2. a local .py corpus globbed from the repo + this Python's stdlib (ungated,
     always available).

Runs on MPS/CUDA/CPU. This is a *bounded* run (``--steps``): on a laptop you get a
genuinely-trained-but-undertrained model -- enough to complete simple code-shaped
byte patterns and to harden the real trained MLP weights. Crank --steps on a GPU
for quality. Saves a HF checkpoint to artifacts/nanocoder/.

    python scripts/22_train_nanocoder.py --steps 1500
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from models.nanocoder.model import DEFAULT_ARCH, ByteTokenizer, build_model  # noqa: E402
from paths import ARTIFACTS_DIR, REPO_ROOT, ensure_dirs, get_logger  # noqa: E402

logger = get_logger("burnttt.script.train_nanocoder")

CKPT_DIR = ARTIFACTS_DIR / "nanocoder"
MAX_CORPUS_BYTES = 8 * 1024 * 1024  # 8 MB is plenty for a byte-level nano model


def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_corpus(tok: ByteTokenizer, max_docs: int = 4000) -> np.ndarray:
    """Return a 1-D int array of byte tokens. tiny-codes if authed, else local .py."""
    texts: list[str] = []
    if os.environ.get("HF_TOKEN"):
        try:
            from datasets import load_dataset

            ds = load_dataset("nampdn-ai/tiny-codes", split="train", streaming=True)
            for i, row in enumerate(ds):
                if i >= max_docs:
                    break
                texts.append(str(row.get("response") or row.get("prompt") or ""))
            logger.info("Loaded %d docs from nampdn-ai/tiny-codes.", len(texts))
        except Exception as exc:  # noqa: BLE001
            logger.warning("tiny-codes load failed (%s); falling back to local .py corpus.", exc)
            texts = []

    if not texts:
        import sysconfig

        roots = [str(REPO_ROOT), sysconfig.get_paths().get("stdlib", "")]
        seen = 0
        for root in roots:
            if not root:
                continue
            for path in glob.glob(os.path.join(root, "**", "*.py"), recursive=True):
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                        texts.append(fh.read())
                        seen += 1
                except OSError:
                    continue
                if seen >= max_docs:
                    break
            if seen >= max_docs:
                break
        logger.info("Loaded %d local .py files as the code corpus.", len(texts))

    blob = ("\n".join(texts)).encode("utf-8", errors="ignore")[:MAX_CORPUS_BYTES]
    if not blob:
        raise RuntimeError("Empty corpus.")
    logger.info("Corpus: %.1f MB of bytes.", len(blob) / 1e6)
    return np.frombuffer(blob, dtype=np.uint8).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NanoCoder byte-level on code")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    import torch

    ensure_dirs()
    dev = _device()
    tok = ByteTokenizer()
    data = _load_corpus(tok)
    data_t = torch.from_numpy(data)
    n = data_t.numel()

    model = build_model(DEFAULT_ARCH).to(dev)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(0)

    def batch():
        ix = rng.integers(0, n - args.seq - 1, size=args.batch)
        xb = torch.stack([data_t[i : i + args.seq] for i in ix]).to(dev)
        yb = torch.stack([data_t[i + 1 : i + 1 + args.seq] for i in ix]).to(dev)
        return xb, yb

    logger.info("Training NanoCoder on %s for %d steps (seq=%d batch=%d)...", dev, args.steps, args.seq, args.batch)
    for step in range(1, args.steps + 1):
        xb, yb = batch()
        out = model(input_ids=xb, labels=yb)
        loss = out.loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0 or step == 1:
            logger.info("step %4d/%d  loss=%.3f  (bits/byte=%.3f)", step, args.steps, loss.item(), loss.item() / 0.6931)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CKPT_DIR)
    logger.info("Saved NanoCoder checkpoint -> %s", CKPT_DIR)

    # Quick sample so we can eyeball that it learned code-shaped bytes.
    model.eval()
    prompt = "def "
    ids = torch.tensor([tok.encode(prompt)], device=dev)
    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=120, do_sample=True, top_k=40, temperature=0.8)
    print("\n--- NanoCoder sample (prompt 'def ') ---")
    print(tok.decode(gen[0].tolist()))
    print("--- end ---")


if __name__ == "__main__":
    main()
