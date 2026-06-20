"""Chat webapp for Qwen-on-F2: POST /api/chat -> FPGA-backed generation.

Loads the quantized Qwen runner once and serves a tiny chat UI plus a JSON API.
The transformer GEMVs run on whatever backend ``QWEN_FPGA_BACKEND`` selects
(``xrt`` = the F2 FPGA, ``cpp`` = the identical CPU datapath for off-board demo).

The /api/chat response always reports the backend and the *measured* GEMV (FPGA)
call latency, per the brief.

Run:
    QWEN_FPGA_BACKEND=xrt QWEN_FPGA_XCLBIN=/path/to.awsxclbin \
    python -m qwen_fpga.webapp.server \
        --manifest qwen_fpga/weights/Qwen2.5-0.5B-Instruct/manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from qwen_fpga.host.fpga_gemv import make_gemv_backend
from qwen_fpga.reference.qref import QwenQuantRunner, load

_PAGE = """<!doctype html><meta charset=utf-8>
<title>Qwen on F2 FPGA</title>
<style>
 body{font:15px system-ui;max-width:680px;margin:40px auto;padding:0 16px}
 #log{border:1px solid #ddd;border-radius:8px;padding:12px;min-height:220px;white-space:pre-wrap}
 .u{color:#0a58ca}.a{color:#198754}.m{color:#888;font-size:12px}
 input,button{font:15px system-ui;padding:8px}
 input{width:78%}button{width:18%}
</style>
<h2>Qwen2.5 &mdash; transformer math on AWS F2 FPGA</h2>
<div id=log></div>
<p><input id=q placeholder="Ask something..." autofocus>
<button onclick=send()>Send</button></p>
<script>
const log=document.getElementById('q');
async function send(){
 const q=document.getElementById('q'),L=document.getElementById('log');
 const text=q.value.trim(); if(!text)return; q.value='';
 L.innerHTML+='\\n<span class=u>you:</span> '+text;
 const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({prompt:text,max_new:48})});
 const j=await r.json();
 L.innerHTML+='\\n<span class=a>qwen:</span> '+j.text+
   '\\n<span class=m>backend='+j.backend+'  gemv_calls='+j.gemv_calls+
   '  gemv_avg='+j.gemv_avg_ms.toFixed(3)+'ms  fpga_total='+j.fpga_latency_ms.toFixed(1)+
   'ms  tok/s='+j.tokens_per_sec.toFixed(2)+'</span>\\n';
 L.scrollTop=L.scrollHeight;
}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key=='Enter')send();});
</script>
"""


class App:
    def __init__(self, manifest: str, backend_name: str | None):
        self.loaded = load(manifest)
        self.backend = make_gemv_backend(backend_name)
        self.runner = QwenQuantRunner(self.loaded, backend=self.backend)
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(self.loaded.manifest["model_id"])
        self.report_backend = "fpga" if self.backend.name in ("xrt", "fpga") else self.backend.name

    def chat(self, prompt: str, max_new: int, temperature: float) -> dict:
        enc = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=True, return_dict=True)
        ids = [int(t) for t in np.asarray(enc["input_ids"]).reshape(-1).tolist()]
        c0, s0 = self.backend.calls, self.backend.total_s
        t0 = time.perf_counter()
        out = self.runner.generate(ids, max_new=max_new, temperature=temperature)
        wall = time.perf_counter() - t0
        calls = self.backend.calls - c0
        fpga_s = self.backend.total_s - s0
        return {
            "text": self.tok.decode(out, skip_special_tokens=True),
            "backend": self.report_backend,
            "gemv_calls": calls,
            "gemv_avg_ms": 1e3 * fpga_s / calls if calls else 0.0,
            "fpga_latency_ms": 1e3 * fpga_s,
            "tokens": len(out),
            "tokens_per_sec": len(out) / wall if wall else 0.0,
        }


def make_handler(app: App):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quieter
            pass

        def _send(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, _PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/healthz":
                self._send(200, json.dumps({"ok": True, "backend": app.report_backend}).encode())
            else:
                self._send(404, b'{"error":"not found"}')

        def do_POST(self):
            if self.path != "/api/chat":
                self._send(404, b'{"error":"not found"}')
                return
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            try:
                res = app.chat(req.get("prompt", ""),
                               int(req.get("max_new", 32)),
                               float(req.get("temperature", 0.0)))
                self._send(200, json.dumps(res).encode())
            except Exception as exc:  # noqa: BLE001
                self._send(500, json.dumps({"error": str(exc)}).encode())

    return H


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--backend", default=None, help="numpy|cpp|xrt (default env or cpp)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    app = App(args.manifest, args.backend)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"serving on http://{args.host}:{args.port}  backend={app.report_backend}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
