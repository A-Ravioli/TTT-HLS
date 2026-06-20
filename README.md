# 🔥 BurnTTT — a GLM that compiles models onto FPGAs, finetuned at test time

BurnTTT is a **test-time-trained LLM compiler**. An LLM (**GLM**) *authors* the
hardware-generation config that maps a neural-network block onto FPGA fabric via
[`hls4ml`](https://github.com/fastmachinelearning/hls4ml), and its **weights are
finetuned (LoRA) at test time** on real quantization-accuracy and resource
feedback from *this exact model block and FPGA target* — so it produces a hardware
configuration that beats the naive default, a random-forest surrogate baseline,
and random search.

The north-star workload is **Qwen-2B on FPGA**, reached by scaling from a toy
feed-forward block up through a single Qwen-2B transformer sub-block.

```
model block → GLM authors hls4ml config → HLS/RTL → bit-accurate sim + (synthesis) → reward
                     ▲                                                                    │
                     └──────────── LoRA test-time finetune on this task's feedback ◄──────┘
```

## 1. What BurnTTT is

Picking the hardware-generation config for an FPGA model is a search problem: too
much precision or too little reuse and the design **won't fit the board**; too
little precision and the **accuracy drifts**. The sweet spot depends on the
specific model block *and* the specific FPGA part.

BurnTTT treats each `(model-block, FPGA-part)` pair as a **fresh task** and runs a
GLM generator that:

1. **Authors** candidate hardware configs (quantization precision, per-layer reuse
   factors, HLS strategy/IO mode) from a structured prompt describing the block,
   the part, its resource budget, and the feedback seen so far.
2. **Evaluates** each config: hls4ml conversion → bit-accurate prediction error →
   (synthesis if available, else an analytical resource/throughput estimate) →
   scalar reward.
3. **Repairs** configs that fail to compile, by feeding the compiler error back to
   GLM (an agentic compiler loop).
4. **Finetunes its own weights** (LoRA) on the high-reward trajectories from this
   task, between rounds — the honest test-time training — then proposes again.

A **random-forest surrogate** ([baselines/random_forest_policy.py](baselines/random_forest_policy.py))
and **random search** run on an equal evaluation budget so the dashboard compares
`default hls4ml` vs `random search` vs `random forest` vs `GLM (frozen)` vs
`GLM (test-time finetuned)`.

## 2. The honest TTT framing

> **The deployed FPGA logic is fixed. The test-time training happens in the
> *generator*: GLM's own (LoRA) weights are updated during the run, on feedback
> from this specific model block and FPGA part, so it authors better hardware the
> longer it works on the task.**

This is genuine test-time *training* (weights change), not just selection — but of
the **compiler/generator**, not of the deployed network. The deployed accelerator
is a fixed bitstream.

### Runs anywhere (graceful degradation)

Every external dependency is optional and degrades gracefully:

- **No GLM weights / GPU / `transformers`?** A deterministic, history-driven
  **heuristic backend** ([glm/serving.py](glm/serving.py)) stands in for the LLM
  and is adapted analogously, so the full loop (including a measurable test-time
  improvement) still runs and is testable. Set `BURN_GLM_MODEL` to use a real GLM.
- **No Vivado/Vitis?** Resources/latency/throughput are **analytical estimates**;
  accuracy (`max_error`) is always real, bit-accurate hls4ml output. Pass
  `--synth` with a toolchain for real numbers.
- **No FPGA board?** A software bit-accurate equivalence demo runs instead.

## 3. Architecture

```
                  ┌──────────────────────────────────────────────┐
   FpgaTask ─────▶│ Prompt: block spec + part budget + feedback   │
 (block, part)    └───────────────────────┬──────────────────────┘
                                           ▼
                              ┌─────────────────────────┐
                              │ GLM generator (LoRA)     │  glm/agent.py
                              └────────────┬────────────┘
                                           ▼  hls4ml config (or per-layer BlockConfig)
                  ┌────────────────────────────────────────────────┐
                  │ Feedback engine: convert → C-sim error →        │  ttt/evaluate_config.py
                  │ synth/estimate → reward + throughput            │
                  └───────────────┬───────────────────┬────────────┘
                                  │ reward + feedback  │ trajectories
                       repair ◀───┘                    ▼  data/trajectories/*.jsonl
                                            ┌─────────────────────────┐
                                            │ Test-time finetune       │  glm/finetune/
                                            │ LoRA step on this task   │
                                            └────────────┬────────────┘
                                                         └─▶ better GLM next round
```

## 4. Install

Requires Python 3.10 or 3.11 (hls4ml 1.0.0 pins TensorFlow ≤ 2.14).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For a **real GLM + test-time LoRA finetuning**, also install the LLM extras and
point at GLM weights:

```bash
pip install -r requirements-glm.txt          # torch, transformers, peft, accelerate
export BURN_GLM_MODEL=THUDM/glm-4-9b-chat     # a local path or HF repo id
```

Without those, BurnTTT uses the heuristic backend automatically.

## 5. Run the pipeline

```bash
python scripts/00_train_model.py                 # train + export the toy FFN block
python scripts/01_baseline_compile.py            # compile the default hls4ml config
python scripts/08_eval_glm_generator.py          # default vs random vs RF vs GLM vs GLM+TTT
streamlit run dashboard/app.py                   # visualize the comparison
```

Test-time finetuning loop with reward-vs-step logging:

```bash
python scripts/06_collect_trajectories.py        # collect (config → feedback) trajectories
python scripts/07_ttt_finetune_glm.py            # finetune GLM at test time; chart reward-vs-step
```

Scale toward Qwen-2B:

```bash
python scripts/05_ingest_qwen.py                 # decompose Qwen + feasibility report
python scripts/05_ingest_qwen.py --export        # build a tiled, compilable Qwen MLP block
python scripts/09_block_to_fpga_demo.py --block qwen_mlp   # block demo + full-model plan
```

Hardware path (graceful stubs off-board):

```bash
python scripts/03_build_best_bitstream.py        # build the best config's HLS/bitstream project
python scripts/04_run_fpga_demo.py               # run on a PYNQ board, or software-equivalence demo
```

## 6. GLM backends and env vars

| Variable             | Effect                                                            |
|----------------------|------------------------------------------------------------------|
| `BURN_GLM_MODEL`     | Local path or HF repo id of the GLM to load (enables real LLM).   |
| `BURN_GLM_BACKEND`   | Force `heuristic` or `hf`.                                        |
| `BURN_TARGET_PART`   | FPGA part (default PYNQ-Z2 `xc7z020clg400-1`).                    |
| `BURN_BOARD`         | Board name for VivadoAccelerator bitstreams.                     |

Backend selection lives in `load_backend()` ([glm/serving.py](glm/serving.py)):
real `HFBackend` if `BURN_GLM_MODEL` is set and `transformers` is importable, else
the `HeuristicBackend`. The test-time trainer ([glm/finetune/trainer.py](glm/finetune/trainer.py))
runs real LoRA steps when `peft`/`torch` are present, and adapts the heuristic
otherwise — same interface either way.

## 7. Config space

`ttt/config_space.py` defines the artifact GLM authors. The compact `BurnConfig`
(uniform precision/reuse) over:

```
bits:   8, 10, 12, 14, 16
int:    3, 4, 5, 6           (must be < bitwidth)
reuse:  1, 2, 4, 8, 16
strat:  Latency, Resource
```

and the richer `BlockConfig` for multi-layer blocks (e.g. Qwen's SwiGLU MLP):
**per-layer** precision/reuse plus block-wide strategy and `io_parallel`/
`io_stream` dataflow. Precision maps to `ap_fixed<bits,int>`; fewer integer bits =
more fractional bits = lower quantization error (until integer overflow).

## 8. Reward function

Implemented in `ttt/reward.py`. A compile failure is the worst outcome; accuracy
drift past a threshold is heavily penalized; designs over the **part's** resource
budget are penalized; otherwise reward favors low latency and low resource use.
Budgets are per-part (`BOARD_BUDGETS`): a toy FFN fits a PYNQ-Z2, while a Qwen
block is judged against a large part (e.g. Alveo U250).

## 9. The dashboard

```bash
streamlit run dashboard/app.py
```

Shows headline metrics, the best config + resource utilization vs the board
budget, **best-reward-over-evaluations** for every method on an equal budget, a
method comparison table, the accuracy-vs-latency tradeoff scatter, the full
results table, and an explainer of what "test-time training" means here.

## 10. Repository layout

```
models/      tiny_ffn (toy block) + qwen/ (load, decompose, blocks, orchestrate)
compiler/    make_hls_config (+ per-layer), build_hls4ml_project, run_hls,
             parse_reports, estimate_resources (+ throughput), deploy_pynq
glm/         tasks, prompts/, parsing, serving (HF + heuristic), agent,
             trajectories, finetune/ (lora, dataset, trainer)   ← the GLM compiler
ttt/         config_space (BurnConfig + BlockConfig), reward (per-part),
             evaluate_config (feedback engine), search (GLM + baselines)
baselines/   random_forest_policy (the former headline, now a baseline)
infra/       staged_eval (sim→synth→bitstream funnel), launch (GPU placement)
dashboard/   app.py (Streamlit)
scripts/     00_train … 09_block_to_fpga_demo
tests/       config/reward/policy + GLM agent + Qwen decomposition tests
data/        trajectory store (jsonl) + LoRA adapters
paths.py     shared paths / logging / target-part helpers
```

Run the tests with:

```bash
python -m pytest tests -q
```

## 11. Roadmap / phases

- **Phase 1 (done):** GLM authors configs (replacing the random forest), with a
  compile-error repair loop, on the toy block.
- **Phase 2 (done):** Test-time LoRA finetuning of GLM on per-task feedback;
  reward-vs-step shows GLM+TTT climbing above frozen GLM and the RF baseline.
- **Phase 3 (done):** Qwen-2B ingestion + decomposition; a tiled, compilable
  SwiGLU MLP sub-block; per-layer `BlockConfig`; per-part budgets + throughput.
- **Phase 4 (stubs):** Full-model orchestration — tiling, weight streaming, KV
  cache, attention softmax/RoPE kernels, multi-FPGA partitioning, host code. See
  [models/qwen/orchestrate.py](models/qwen/orchestrate.py).

## 12. Known limitations

- **The honest target today is a single block + GLM test-time finetune.** Full
  Qwen-2B-on-FPGA is the north star, with the remaining work made explicit (not
  hidden) in `models/qwen/orchestrate.py`.
- **Attention is not hls4ml-native.** The q/k/v/o projections compile, but
  softmax/RoPE need FINN or hand-written HLS — flagged as a research stub. The MLP
  block is the first real milestone.
- **Estimates without Vivado.** Off-toolchain, latency/resources/throughput are
  analytical estimates; accuracy is always real, bit-accurate hls4ml output.
- **Heuristic stand-in off-GPU.** Without GLM weights the heuristic backend
  demonstrates the same loop; the real contribution is the LoRA test-time loop
  when a GLM is present.

## 13. What to say in a demo

1. "We treat each `(model block, FPGA part)` as a fresh task."
2. "**GLM authors** the hardware config; failed compiles are **repaired** by
   feeding the error back to the model."
3. "We **finetune GLM's own weights (LoRA) at test time** on this task's
   synthesis/simulation feedback — watch the reward-vs-step curve climb above the
   frozen model and the random-forest baseline."
4. "The deployed FPGA logic is fixed — the **test-time training is in the
   generator**, before deployment."
5. "Toy block today; here's the **Qwen-2B** decomposition and the path to running
   a real transformer block on the board."
