#!/usr/bin/env python
"""Serve NanoCoder (the FPGA-hardened byte-level coder) for the Pinkdeer UI.

This replaces the Qwen backend: the chat brain is now the *same* model that
hardens onto the PYNQ-Z2 (models/nanocoder, 1.7M params, vocab=256). At this size
it is hyperfast -- hundreds of bytes/sec on CPU, more on MPS -- which is the whole
point: one tiny model, fast to run and small enough to put on a $100 board.

NanoCoder is a byte-level *completion* model (no chat template), so we flatten the
conversation to a byte prompt and stream the continuation token-by-token via a
manual sampling loop (with KV-cache) -- ByteTokenizer is not an HF tokenizer, so we
do not use TextIteratorStreamer.

Quality caveat: a 1.7M model trained briefly emits code-*shaped* bytes, not correct
programs. Train longer (scripts/22 --steps) for coherence.

    python scripts/24_serve_nanocoder.py            # :8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.nanocoder.model import load_tokenizer  # noqa: E402
from paths import ARTIFACTS_DIR, get_logger  # noqa: E402

logger = get_logger("burnttt.script.serve_nanocoder")

CKPT_DIR = ARTIFACTS_DIR / "nanocoder"
MODEL_NAME = "NanoCoder-1.7M"
CTX = 512

_LOCK = threading.Lock()
_TOK = None  # set in _load() from the checkpoint dir (BPE if present, else byte)
_MODEL = None
_DEVICE = "cpu"


def _pick_device() -> str:
    # NanoCoder is tiny (1.7M): single-token decode steps are dominated by
    # accelerator kernel-launch latency, so CPU is ~19x faster than MPS here
    # (921 vs 48 tok/s measured). Default to CPU; override with NANO_DEVICE.
    dev = os.environ.get("NANO_DEVICE", "cpu").strip()
    return dev or "cpu"


def _load() -> None:
    global _MODEL, _DEVICE, _TOK, MODEL_NAME
    import torch
    from transformers import GPTNeoForCausalLM

    if not (CKPT_DIR / "config.json").exists():
        raise SystemExit(f"No NanoCoder checkpoint at {CKPT_DIR}. Train it: modal run infra/modal_train_nanocoder_v2.py")
    _DEVICE = _pick_device()
    _MODEL = GPTNeoForCausalLM.from_pretrained(CKPT_DIR).to(_DEVICE).eval()
    _TOK = load_tokenizer(CKPT_DIR)
    n = sum(p.numel() for p in _MODEL.parameters())
    MODEL_NAME = f"NanoCoder-{n/1e6:.1f}M"
    logger.info("Loaded %s (%s params) on %s. Ready.", MODEL_NAME, f"{n:,}", _DEVICE)


def _prompt_from(messages: list[dict], system: str) -> str:
    """Flatten the conversation into a byte-completion prompt."""
    parts = [m.get("content") or m.get("text") or "" for m in messages]
    return "\n".join(p for p in parts if p).strip()


def _stream_iter(prompt: str, top_k: int, temperature: float, max_new_tokens: int):
    """Manual top-k sampling loop with KV-cache; yields decoded text incrementally.

    Tokenizer-agnostic: accumulates generated ids and emits the decode delta each
    step, so it works for the BPE tokenizer (v2) and the byte tokenizer (v1) alike.
    """
    import torch

    eos = getattr(_TOK, "eos_id", None)
    start = _TOK.encode(prompt)[-(CTX - 1):] or [eos if eos is not None else 0]
    ids = torch.tensor([start], device=_DEVICE)
    top_k = max(1, int(top_k))
    temp = max(0.05, float(temperature))
    gen: list[int] = []
    prev = ""
    t0 = time.time()
    with _LOCK, torch.no_grad():
        past = None
        cur = ids
        for _ in range(int(max_new_tokens)):
            out = _MODEL(input_ids=cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1, :] / temp
            topv, topi = torch.topk(logits, min(top_k, logits.shape[-1]))
            probs = torch.softmax(topv, dim=-1)
            nxt = int(topi[torch.multinomial(probs, 1)])
            cur = torch.tensor([[nxt]], device=_DEVICE)
            if eos is not None and nxt == eos:
                break
            gen.append(nxt)
            text = _TOK.decode(gen)
            if len(text) > len(prev):
                yield text[len(prev):]
                prev = text
    dt = time.time() - t0
    if dt > 0:
        logger.info("generated %d tokens in %.2fs = %.0f tok/s", len(gen), dt, len(gen) / dt)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._json(204, {})

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", "/api/health"):
            self._json(200, {"ok": True, "model": MODEL_NAME, "device": _DEVICE})
        else:
            self._json(404, {"error": "not found"})

    def _params(self, req):
        messages = req.get("messages") or ([{"role": "user", "content": req["prompt"]}] if req.get("prompt") else [])
        return (
            _prompt_from(messages, req.get("system", "")),
            req.get("top_k", 20),
            req.get("temperature", 0.8),
            int(req.get("max_new_tokens", 200)),
        )

    def do_POST(self):
        route = self.path.rstrip("/")
        streaming = route in ("/generate_stream", "/api/generate_stream")
        if route not in ("/generate", "/api/generate") and not streaming:
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
        except Exception as exc:  # noqa: BLE001
            self._json(400, {"error": f"bad request: {exc}"})
            return

        prompt, top_k, temperature, max_new_tokens = self._params(req)
        if not prompt:
            self._json(400, {"error": "empty prompt"})
            return

        if streaming:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            try:
                for piece in _stream_iter(prompt, top_k, temperature, max_new_tokens):
                    self.wfile.write(piece.encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:  # noqa: BLE001
                logger.exception("stream failed")
                try:
                    self.wfile.write(f"\n⚠ {exc}".encode())
                except Exception:  # noqa: BLE001
                    pass
            return

        text = "".join(_stream_iter(prompt, top_k, temperature, max_new_tokens))
        self._json(200, {"text": text, "model": MODEL_NAME})


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve NanoCoder for the Pinkdeer UI")
    parser.add_argument("--port", type=int, default=int(os.environ.get("CHAT_PORT", 8000)))
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    _load()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    logger.info("Serving %s on http://%s:%d  (POST /generate_stream, GET /health)", MODEL_NAME, args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
