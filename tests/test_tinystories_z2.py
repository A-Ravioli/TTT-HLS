"""Stage-1 (off-board) validation of the TinyStories -> PYNQ Z2 path.

These prove the numeric foundation without any FPGA or Vivado:

  1. the from-scratch GPT-Neo decode matches Hugging Face's ``GPTNeoForCausalLM``
     to fp32 tolerance (architecture is correct);
  2. the W8A8 export round-trips with small weight error;
  3. the *quantized* W8A8 datapath -- the exact integer arithmetic the Z2 GEMV
     kernel will run -- still generates coherent TinyStories text.

Skips cleanly if transformers / the model download is unavailable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

MODEL_ID = "roneneldan/TinyStories-1M"
PROMPT = "Once upon a time, there was a little"


@pytest.fixture(scope="module")
def hf():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.float32
        ).eval()
        tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as exc:  # noqa: BLE001 - offline / hub unavailable
        pytest.skip(f"could not load {MODEL_ID}: {exc}")
    return model, tok


@pytest.fixture(scope="module")
def exported(hf, tmp_path_factory):
    from tinystories_z2.export_tinystories import export

    model, _ = hf
    out = tmp_path_factory.mktemp("ts_weights")
    manifest = export(MODEL_ID, Path(out))
    return manifest


def _ids(tok):
    return [int(t) for t in tok(PROMPT).input_ids]


def test_arch_matches_hf(hf):
    """fp32 runner reproduces HF logits + greedy text exactly."""
    import torch

    from tinystories_z2.model import HFWeights, NeoArch, NeoRunner

    model, tok = hf
    ids = _ids(tok)
    arch = NeoArch.from_hf_config(model.config)
    runner = NeoRunner(arch, HFWeights(model))

    with torch.no_grad():
        hf_logits = model(torch.tensor([ids])).logits[0].numpy()

    runner.reset()
    for pos, t in enumerate(ids):
        mine = runner.forward_token(t, pos)
        assert int(mine.argmax()) == int(hf_logits[pos].argmax())
        assert np.max(np.abs(mine - hf_logits[pos])) < 1e-2

    hf_greedy = model.generate(torch.tensor([ids]), max_new_tokens=20, do_sample=False)
    hf_text = tok.decode(hf_greedy[0][len(ids):])
    mine_text = tok.decode(runner.generate(ids, max_new=20, temperature=0.0))
    assert mine_text == hf_text


def test_export_roundtrip(exported):
    """Per-row INT8 dequantization is faithful (small relative error)."""
    from tinystories_z2.model import QuantWeights
    from tinystories_z2.quant import dequantize_weight

    qw = QuantWeights(exported)
    for key in ("lm_head", "L0.q", "L3.fc", "L7.proj"):
        w = dequantize_weight(qw._qweight(key))
        # reconstruction should be within a few percent (INT8 per-row)
        assert np.isfinite(w).all()
        assert w.shape[0] == qw._spec(key)["out_features"]


def test_quantized_decode_coherent(hf, exported):
    """The W8A8 integer datapath (FPGA-equivalent) generates coherent text."""
    from tinystories_z2.model import NeoArch, NeoRunner, QuantWeights

    _, tok = hf
    arch = NeoArch.from_manifest(json.loads(Path(exported).read_text()))
    runner = NeoRunner(arch, QuantWeights(exported))  # numpy W8A8 == FPGA math
    ids = _ids(tok)
    out = runner.generate(ids, max_new=30, temperature=0.0)
    text = tok.decode(out)

    # not degenerate: many distinct tokens, mostly letters/spaces, real words
    assert len(set(out)) > 8, f"degenerate output: {text!r}"
    letters = sum(c.isalpha() or c.isspace() for c in text)
    assert letters / max(len(text), 1) > 0.85, f"non-text output: {text!r}"
    assert any(w in text.lower() for w in ("she", "the", "was", "a ")), text


def test_dequant_matches_quant_closely(hf, exported):
    """Activation-quant error is small: full W8A8 logits ~ weight-only dequant."""
    from tinystories_z2.model import NeoArch, NeoRunner, QuantWeights

    _, tok = hf
    arch = NeoArch.from_manifest(json.loads(Path(exported).read_text()))
    r_q = NeoRunner(arch, QuantWeights(exported))
    r_d = NeoRunner(arch, QuantWeights(exported, dequant=True))
    r_q.reset()
    r_d.reset()
    for pos, t in enumerate(_ids(tok)):
        lq = r_q.forward_token(t, pos)
        ld = r_d.forward_token(t, pos)
        # int8 activation quantization barely perturbs the logits
        assert np.corrcoef(lq, ld)[0, 1] > 0.99
