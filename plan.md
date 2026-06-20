# BurnTTT — realigned plan: a test-time-trained GLM compiler for model→FPGA

**Mission.** Finetune an LLM (**GLM**) *at test time* so it writes the optimal
compiler/generator that maps neural-network inference onto an FPGA, with
**Qwen-2B on FPGA at absolute peak performance** as the north-star workload.

This document is the realigned plan. The original hackathon plan ("BurnTTT MVP":
a random-forest surrogate over hls4ml knobs on a toy model) shipped first and now
survives as a *baseline* the GLM generator is measured against. Phases 1–3 prove
the TTT loop on a bounded action space; **Phases 4–7** pursue full custom
compiler/HLS where hls4ml's ceiling is the binding constraint.

**Budget assumption (north star).** Unlimited compute for GLM inference, LoRA
finetuning, HLS/Vivado synthesis, and on-board benchmarking. We optimize for
**peak deployed performance**, not eval-budget frugality. hls4ml config search
remains a fast baseline and warm-start; the production path is GLM-authored
custom HLS + host orchestration.

---

## 0. The honest framing

> **The deployed FPGA logic is fixed. The test-time training happens in the
> generator: GLM's own (LoRA) weights are updated during the run, on feedback
> from this specific `(model block, FPGA part)` task, so it authors better
> hardware the longer it works on the task.**

This is genuine test-time *training* of the compiler — weights change — as opposed
to the original project, where "TTT" meant refitting a random forest (no LLM, no
weight updates). The deployed accelerator remains a fixed bitstream.

**Two generator modes (same TTT machinery, different output artifacts):**

| Mode | GLM output | Backend | Performance ceiling |
|------|------------|---------|---------------------|
| **Config author** (Phases 1–3, done) | `BurnConfig` / `BlockConfig` JSON | Fixed hls4ml template | Good matmul blocks; hits hls4ml wall |
| **Compiler author** (Phases 4–7) | Vitis HLS C++, pragmas, tiling, host glue | GLM *is* the compiler | Absolute peak: streaming, fusion, attention |

Both modes use verifier-driven LoRA TTT (high-reward trajectories → gradient steps).
The north star requires **compiler author** mode.

---

## 1. What changed vs. the original MVP

| Dimension        | Original MVP (now baseline)        | Realigned system (Phases 1–3)                      | Peak-performance target (Phases 4–7)                |
|------------------|------------------------------------|----------------------------------------------------|-------------------------------------------------------|
| Policy           | `RandomForestRegressor` surrogate  | **GLM** authors configs (+ repairs failures)       | **GLM** authors + iterates on **custom HLS**          |
| "TTT"            | Refit a surrogate online           | **LoRA finetune GLM** on task feedback             | Same; richer supervision (HLS diffs, cosim, board)    |
| Compiler         | hls4ml (fixed)                     | hls4ml as feedback engine; GLM authors the config  | **Custom Vitis HLS**; hls4ml is baseline only         |
| Target workload  | 16→64→8 toy FFN                    | **Qwen-2B** sub-blocks                             | **Full Qwen-2B** decode on multi-FPGA / Alveo farm    |
| Config artifact  | 6-field `BurnConfig`               | `BurnConfig` + `BlockConfig`                       | **KernelBundle** (HLS sources + metadata + host)      |
| Eval funnel      | Synth sparingly                    | Staged sim→synth (`infra/staged_eval.py`)          | **Synth everything promising**; parallel Vivado pool  |

The random forest is retained at [baselines/random_forest_policy.py](baselines/random_forest_policy.py)
so the dashboard can show GLM beating it. At peak-performance scale we also
benchmark against **best hls4ml config TTT** and **hand-tuned vendor baselines**.

---

## 2. System architecture

### 2.1 Config-author loop (Phases 1–3, shipped)

```
FpgaTask(block, part, budget)
        │
        ▼  prompt: block spec + budget + feedback-so-far
   GLM generator (glm/agent.py, glm/serving.py)  ──► hls4ml config / BlockConfig
        │                                                   │
        │ repair on compile error                           ▼
        │                                   feedback engine (ttt/evaluate_config.py)
        │                                   convert → C-sim error → synth/estimate
        │                                                   │ reward + throughput
        └──────────── better config next round ◄── LoRA test-time finetune ◄── trajectories
                                                   (glm/finetune/)            (data/trajectories)
```

### 2.2 Compiler-author loop (Phases 4–7, north star)

```
FpgaTask(block | full_model, part(s), SLO)
        │
        ▼  prompt: IR spec + golden I/O + budget + HLS feedback history
   GLM compiler agent  ──► KernelBundle (HLS .cpp/.h, pragmas, tiling, host DMA)
        │                         │
        │ multi-turn repair       ▼
        │ (HLS errors, cosim      custom toolchain (compiler/hls/)
        │  mismatch, timing)     Vitis HLS → RTL → Vivado P&R → bitstream
        │                         │
        │                         ▼
        │              verifier stack (unlimited budget):
        │              • C-synthesis + C/RTL cosim vs golden Qwen block
        │              • post-route timing, resource, power
        │              • on-board tokens/sec @ target batch/seq
        │                         │ reward (latency, throughput, accuracy, $optional power)
        └──── edit / rewrite HLS ◄── LoRA + DPO on (prompt → good HLS) trajectories
              next iteration       preference pairs from synth-ranked candidates
                                   (data/trajectories/hls/)
```

Graceful degradation remains for CI: no GPU/GLM → heuristic backend; no Vivado →
analytical estimates. **North-star runs assume full toolchain + board farm.**

---

## 3. Components

### 3.1 Tasks — `glm/tasks.py`
`FpgaTask = (BlockSpec, target_part, budget, max_error)`. `BlockSpec` is a list of
matmul-bearing `LayerSpec`s. Built from the toy model or a Qwen sub-block.

**Phase 4+ extension:** `FullModelTask` — whole Qwen graph, multi-part placement,
decode SLO (tokens/sec @ seq_len, batch), KV-cache budget, weight-streaming
bandwidth. See [models/qwen/orchestrate.py](models/qwen/orchestrate.py).

### 3.2 GLM generator — `glm/`
- `serving.py`: `GLMBackend` ABC; `HFBackend` (real GLM via `transformers`,
  LoRA-adaptable, exposes `model`/`tokenizer`); `HeuristicBackend` (deterministic
  stand-in, similarity-weighted over feedback history, with an `adapt()` knob);
  `load_backend()` factory driven by `BURN_GLM_MODEL`/`BURN_GLM_BACKEND`.
- `prompts/`: system prompt + propose/repair templates including the legal config
  schema and the per-task feedback history.
- `parsing.py`: robustly extract + validate JSON configs (snap to the legal grid,
  clamp illegal `int_bits`).
- `agent.py`: `GLMGenerator` — propose/repair with validation, de-dup, and a
  random-exploration top-up so the loop never stalls.
- `trajectories.py`: append-only jsonl store under `data/trajectories/`.

**Phase 4+ extensions (`glm/`):**
- `prompts/hls_templates.py` — kernel spec, tiling constraints, pragma cookbook,
  golden-model I/O contract, multi-turn repair with line-level error context.
- `parsing/hls.py` — extract `.cpp`/`.h` from model output; static checks
  (forbidden APIs, interface pragmas present).
- `agent_hls.py` — `GLMCompilerAgent`: propose → compile → repair loop with
  **iterative edit** on the same artifact (not just fresh samples); branch on
  compile vs numerics vs timing failure.
- `finetune/dataset_hls.py` — SFT on top-synth kernels; DPO pairs from ranked
  trajectories; optional process supervision on repair steps.

### 3.3 Test-time finetuning — `glm/finetune/`
- `dataset.py`: trajectories → SFT examples (high-reward `prompt→config`) and
  preference pairs (higher-reward chosen over lower).
- `lora.py`: `peft` LoRA config targeting GLM/Qwen-style projections.
- `trainer.py`: `TestTimeTrainer.step()` — real LoRA gradient steps when
  `peft`/`torch` are present, else `HeuristicBackend.adapt()`. Same interface.

**Phase 4+ extensions:**
- `trainer_hls.py` — longer-context LoRA (8k–32k tokens for HLS sources);
  mixed SFT + DPO; task-specific adapter per `(model, part)` kept across rounds.
- Verifier-driven sample selection (VDS-TTT style): only backprop on kernels that
  pass cosim within ε or achieve best post-route Fmax.

### 3.4 Feedback engine — `ttt/`
- `evaluate_config.py`: convert → C-sim error → synth/estimate → reward (unchanged
  pipeline; now also surfaces throughput/accumulator-width estimates).
- `reward.py`: per-part budgets (`BOARD_BUDGETS`: PYNQ-Z2, KV260, Alveo U250),
  over-budget penalties judged against the task's part.
- `config_space.py`: `BurnConfig` (compact) + `BlockConfig`/`LayerKnobs` (per-layer
  precision/reuse + strategy + io_type).
- `search.py`: `run_glm_search` (frozen) and `run_glm_ttt_search` (test-time
  finetuned), plus the existing baseline/random/RF runs, all writing `runs.csv`.

**Phase 4+ extensions (`ttt/` + new `compiler/`):**
- `evaluate_hls.py` — full verifier pipeline: HLS compile → cosim vs golden
  tensors → Vivado synth/P&R → optional on-board benchmark.
- `reward_hls.py` — primary objective **tokens/sec** (or inverse latency per
  token); hard constraints on max_error, timing closure, DRAM bandwidth; soft
  penalties on power and BRAM.
- `config_space.py` → add `KernelBundle` dataclass (sources, tile sizes,
  precision typedefs, AXI width, double-buffer depth).

### 3.5 Custom compiler toolchain — `compiler/` (Phase 4+)

New package responsibilities (hls4ml helpers stay for baseline):

| Module | Role |
|--------|------|
| `compiler/ir.py` | Block/graph IR from Qwen decomposition; shapes, dtypes, fusion boundaries |
| `compiler/golden.py` | Reference outputs (PyTorch Qwen) for cosim and max_error |
| `compiler/hls_build.py` | Vitis HLS project scaffold, part selection, clock target |
| `compiler/cosim.py` | C/RTL cosim harness vs golden I/O |
| `compiler/vivado.py` | Batch P&R, timing reports, bitstream generation |
| `compiler/host.py` | XRT/OpenCL host: DMA, weight streaming, KV-cache read/write |
| `compiler/kernel_lib/` | Seed templates GLM edits (systolic GEMM, softmax, RoPE, layer norm) |

GLM does **not** emit hls4ml JSON in peak mode. It emits/edits files that drop
into this toolchain. Warm-start: translate best `BlockConfig` from Phase 3 into
initial HLS tile sizes and `ap_fixed` typedefs.

### 3.6 Qwen — `models/qwen/`
- `load_qwen.py`: architecture via `transformers.AutoConfig`, else a built-in spec
  table (works offline).
- `decompose.py`: decoder layer → sub-blocks (MLP = hls4ml-ready; attention =
  **custom HLS target** in Phase 5+).
- `blocks.py`: build a **tiled, compilable** SwiGLU MLP Keras model + golden I/O.
- `orchestrate.py`: full-model planning (capacity bin-pack, KV-cache and
  weight-streaming sizing); becomes **executable placement plan** in Phase 6.

### 3.7 Infra — `infra/`
- `staged_eval.py`: sim→synth funnel for **config mode** (keep for CI/baseline).
- `launch.py`: device/GPU report and placement description for finetuning.

**Phase 4+ extensions (`infra/`):**
- `synth_farm.py` — parallel Vivado/HLS job queue (unlimited workers); no
  candidate cap; priority queue by GLM confidence and partial cosim scores.
- `board_farm.py` — on-board benchmark orchestration; tokens/sec ground truth.
- `trace_store.py` — full HLS + log artifacts per trajectory for DPO mining.

---

## 4. Phases (status)

### Shipped (config-author TTT)

1. **Phase 1 — GLM replaces the random forest (done).** `GLMGenerator` authors
   configs on the toy block with a compile-error repair loop; benchmarked against
   the RF baseline.
2. **Phase 2 — Test-time finetuning (done).** LoRA (or heuristic) adaptation on
   per-task feedback; `scripts/07` charts reward-vs-step (GLM+TTT > frozen GLM >
   baseline).
3. **Phase 3 — Scale to a Qwen-2B block (done).** Ingest/decompose Qwen; tiled
   SwiGLU MLP block; per-layer `BlockConfig`; per-part budgets + throughput.

### Peak performance (compiler-author TTT, unlimited budget)

4. **Phase 4 — Custom HLS for one Qwen MLP sub-block.**
   - Build `compiler/` toolchain: IR, golden cosim, Vitis HLS batch build.
   - `GLMCompilerAgent` authors/edits SwiGLU MLP HLS with **fused gate/up/down**,
     explicit tiling, double-buffered weight DMA, `ap_fixed` or `ap_int` chosen
     by GLM.
   - TTT: LoRA + DPO on cosim-passing kernels; **full synth on all cosim winners**.
   - **Done when:** custom HLS beats best Phase-3 hls4ml+TTT on **post-route
     latency and tokens/sec** on Alveo U250, same accuracy ε.

5. **Phase 5 — Attention + non-hls4ml kernels.**
   - Hand-seed `kernel_lib/`: online softmax, RoPE, RMSNorm, GQA KV layout.
   - GLM iterates on fusion (QK^T → softmax → V) and numeric stability (max-sub,
     accumulator width).
   - Full decoder layer (attn + MLP) as one schedulable dataflow graph.
   - **Done when:** single Qwen decoder layer meets SLO on U250; cosim max_error
     ≤ threshold vs golden.

6. **Phase 6 — Full Qwen-2B multi-FPGA deployment.**
   - Execute [models/qwen/orchestrate.py](models/qwen/orchestrate.py) plan:
     weight streaming, off-chip KV cache, layer pipelining across chips/links.
   - GLM authors **host orchestration** (batching, prefetch, layer fusion across
     PCIe/NOC).
   - Per-layer bitstreams or partial reconfig where it wins latency.
   - **Done when:** end-to-end **Qwen-2B decode** runs on target cluster;
     report tokens/sec, power, accuracy vs PyTorch reference.

7. **Phase 7 — Absolute peak polish.**
   - Human-in-the-loop optional; GLM continues TTT from best hand-tuned baseline.
   - Architecture search: GLM proposes **microarchitecture variants** (systolic
     vs spatial, FIFO depths, clock targets) not expressible in hls4ml.
   - Autotune clock/floorplan with unlimited Vivado runs; Pareto frontier of
     latency vs power documented.
   - **Done when:** no remaining knob (HLS, host, placement) improves tokens/sec
     on board; documented gap vs theoretical roofline.

---

## 5. Scripts

### Shipped

```
00_train_model.py            train + export the toy FFN block
01_baseline_compile.py       default hls4ml compile (software-burned proof)
05_ingest_qwen.py            decompose Qwen; --export a tiled compilable MLP block
06_collect_trajectories.py   run frozen GLM; log (config → feedback) trajectories
07_ttt_finetune_glm.py       test-time finetune GLM; chart reward-vs-step
08_eval_glm_generator.py     default vs random vs RF vs GLM vs GLM+TTT → runs.csv
03_build_best_bitstream.py   build best config's HLS/bitstream project
04_run_fpga_demo.py          on-board run, or software-equivalence demo
09_block_to_fpga_demo.py     block demo + full-model orchestration plan
```

### Phase 4+ (compiler-author)

```
10_bootstrap_hls_from_config.py   warm-start HLS from best BlockConfig / hls4ml export
11_glm_author_hls.py              GLM propose/repair loop on custom HLS (single block)
12_cosim_golden.py                batch cosim all kernels in a run vs Qwen golden
13_synth_farm.py                  parallel Vivado P&R for cosim-passing kernels
14_ttt_finetune_glm_hls.py        LoRA+DPO on HLS trajectories; reward-vs-iteration
15_eval_hls_vs_hls4ml.py          Pareto: custom HLS vs hls4ml config TTT vs RF
16_full_layer_hls.py              Phase 5: fused decoder layer build + benchmark
17_full_model_deploy.py           Phase 6: multi-FPGA Qwen-2B host + bitstreams
18_roofline_report.py             Phase 7: latency/power vs theoretical bounds
```

---

## 6. Reward

### Config mode (Phases 1–3)

```python
def reward(result):
    if not result["compile_success"]:
        return -1000
    if result["max_error"] > 0.25:
        return -500 - 100 * result["max_error"]
    score = (1000
             - 0.05 * latency_cycles - 1.0 * dsp - 0.05 * lut
             - 0.1 * bram - 50.0 * max_error)
    # minus 300 per resource over 80% of THIS PART's budget
    return score
```

### HLS / peak mode (Phases 4–7)

Primary metric is **measured throughput**, not proxy estimates:

```python
def reward_hls(result):
    if not result["hls_compile_success"]:
        return -1000
    if not result["cosim_pass"]:
        return -800 - 100 * result["max_error"]  # numerics failure
    if not result["timing_met"]:
        return -600 - 10 * result["wns_violation_ns"]
    # tokens/sec from on-board or cycle-accurate sim @ target clock
    tps = result["tokens_per_sec"]
    score = 1000.0 * tps - 0.01 * result["power_w"] - 50.0 * result["max_error"]
    if result["max_error"] > result["max_error_threshold"]:
        return -500 - 100 * result["max_error"]
    return score
```

Preference pairs for DPO: kernel A ≻ kernel B iff `reward_hls(A) > reward_hls(B)`
after both complete cosim (incomparable failures ranked by failure mode severity).

---

## 7. Risks / open questions

### Config mode (mitigated in Phases 1–3)
- **Synthesis cost.** Staged funnel for CI; not a constraint on north-star runs.
- **Real GLM finetuning infra.** Heuristic backend for CI; production uses real GLM + GPU.

### Peak / custom HLS (Phase 4+)
- **LLM HLS correctness.** Mitigate with seed templates, incremental edit prompts,
  cosim gate before synth, and DPO on repair trajectories (not just final kernels).
- **Attention numerics.** Softmax/RoPE need explicit stability contracts in prompts
  and verifier thresholds; start from verified `kernel_lib/` seeds.
- **Search space explosion.** Unlimited synth farm + parallel GLM sampling (N≫1
  candidates per round); rank before LoRA step; keep task-specific LoRA adapters.
- **Multi-FPGA host complexity.** Phase 6 treats host code as first-class GLM
  output with the same repair/TTT loop; benchmark end-to-end, not per-kernel only.
- **Verification truth.** On-board tokens/sec is ground truth; cosim is filter;
  post-route timing is hard constraint before board runs.

---

## 8. What "done" means

### Milestone A (shipped) — config-author TTT
Single `(Qwen MLP block, large FPGA part)` task where **test-time-finetuned GLM**
authors a config that fits the part, keeps error under threshold, and beats frozen
GLM and the random-forest baseline on an equal evaluation budget — with
reward-vs-step curve and (where toolchain/board exist) a real bitstream PASS.

### Milestone B (north star) — absolute peak performance
**Full Qwen-2B decode** on the target FPGA cluster where:

1. GLM **authored and iteratively refined custom HLS + host code** (not hls4ml knobs).
2. Test-time LoRA/DPO adapted the generator on **synthesis, cosim, and board** feedback.
3. Deployed system **beats best hls4ml config TTT** on tokens/sec at equal accuracy.
4. Documented **Pareto frontier** (latency, power, resources) and gap to roofline.
5. No hls4ml-expressibility ceiling remains on the critical path (attention,
   streaming, KV cache handled in custom kernels).

hls4ml config search remains the fast regression baseline and Phase 1–3 demo path;
**Milestone B is the mission.**
