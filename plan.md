# BurnTTT — realigned plan: a test-time-trained GLM compiler for model→FPGA

**Mission.** Finetune an LLM (**GLM**) *at test time* so it writes the optimal
compiler/generator that maps neural-network inference onto an FPGA, with
**Qwen-2B on FPGA** as the north-star workload.

This document is the realigned plan. The original hackathon plan ("BurnTTT MVP":
a random-forest surrogate over hls4ml knobs on a toy model) shipped first and now
survives as a *baseline* the GLM generator is measured against. What follows is
the architecture we actually build toward the mission.

---

## 0. The honest framing

> **The deployed FPGA logic is fixed. The test-time training happens in the
> generator: GLM's own (LoRA) weights are updated during the run, on feedback
> from this specific `(model block, FPGA part)` task, so it authors better
> hardware the longer it works on the task.**

This is genuine test-time *training* of the compiler — weights change — as opposed
to the original project, where "TTT" meant refitting a random forest (no LLM, no
weight updates). The deployed accelerator remains a fixed bitstream.

---

## 1. What changed vs. the original MVP

| Dimension        | Original MVP (now baseline)        | Realigned system                                   |
|------------------|------------------------------------|----------------------------------------------------|
| Policy           | `RandomForestRegressor` surrogate  | **GLM** that *authors* configs (+ repairs failures)|
| "TTT"            | Refit a surrogate online           | **LoRA finetune GLM's weights** on task feedback   |
| Compiler         | hls4ml (fixed)                     | hls4ml as feedback engine; **GLM authors the config** |
| Target workload  | 16→64→8 toy FFN                    | **Qwen-2B**, via single transformer sub-blocks     |
| Config artifact  | 6-field `BurnConfig`               | `BurnConfig` **and** per-layer `BlockConfig`       |

The random forest is retained at [baselines/random_forest_policy.py](baselines/random_forest_policy.py)
so the dashboard can show GLM beating it on an equal budget.

---

## 2. System architecture

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

Graceful degradation is a first-class requirement: no GPU/GLM → heuristic backend;
no Vivado → analytical estimates; no board → software-equivalence demo. The loop,
and a measurable test-time improvement, run in all cases.

---

## 3. Components

### 3.1 Tasks — `glm/tasks.py`
`FpgaTask = (BlockSpec, target_part, budget, max_error)`. `BlockSpec` is a list of
matmul-bearing `LayerSpec`s. Built from the toy model or a Qwen sub-block.

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

### 3.3 Test-time finetuning — `glm/finetune/`
- `dataset.py`: trajectories → SFT examples (high-reward `prompt→config`) and
  preference pairs (higher-reward chosen over lower).
- `lora.py`: `peft` LoRA config targeting GLM/Qwen-style projections.
- `trainer.py`: `TestTimeTrainer.step()` — real LoRA gradient steps when
  `peft`/`torch` are present, else `HeuristicBackend.adapt()`. Same interface.

### 3.4 Feedback engine — `ttt/`
- `evaluate_config.py`: convert → C-sim error → synth/estimate → reward (unchanged
  pipeline; now also surfaces throughput/accumulator-width estimates).
- `reward.py`: per-part budgets (`BOARD_BUDGETS`: PYNQ-Z2, KV260, Alveo U250),
  over-budget penalties judged against the task's part.
- `config_space.py`: `BurnConfig` (compact) + `BlockConfig`/`LayerKnobs` (per-layer
  precision/reuse + strategy + io_type).
- `search.py`: `run_glm_search` (frozen) and `run_glm_ttt_search` (test-time
  finetuned), plus the existing baseline/random/RF runs, all writing `runs.csv`.

### 3.5 Qwen — `models/qwen/`
- `load_qwen.py`: architecture via `transformers.AutoConfig`, else a built-in spec
  table (works offline).
- `decompose.py`: decoder layer → sub-blocks (MLP = hls4ml-ready; attention =
  research stub) + feasibility report.
- `blocks.py`: build a **tiled, compilable** SwiGLU MLP Keras model + golden I/O.
- `orchestrate.py`: full-model planning stubs (capacity bin-pack, KV-cache and
  weight-streaming sizing, the explicit TODO list for a real full deployment).

### 3.6 Infra — `infra/`
- `staged_eval.py`: the sim→synth→bitstream funnel (don't synthesize everything
  GLM proposes).
- `launch.py`: device/GPU report and placement description for finetuning.

---

## 4. Phases (status)

1. **Phase 1 — GLM replaces the random forest (done).** `GLMGenerator` authors
   configs on the toy block with a compile-error repair loop; benchmarked against
   the RF baseline.
2. **Phase 2 — Test-time finetuning (done).** LoRA (or heuristic) adaptation on
   per-task feedback; `scripts/07` charts reward-vs-step (GLM+TTT > frozen GLM >
   baseline).
3. **Phase 3 — Scale to a Qwen-2B block (done).** Ingest/decompose Qwen; tiled
   SwiGLU MLP block; per-layer `BlockConfig`; per-part budgets + throughput.
4. **Phase 4 — Full Qwen-2B (research stubs).** Tiling, weight streaming, KV
   cache, attention softmax/RoPE kernels, multi-FPGA partitioning, host code.

---

## 5. Scripts

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

---

## 6. Reward (unchanged shape, per-part budgets)

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

---

## 7. Risks / open questions

- **Attention on FPGA.** hls4ml can't express softmax/RoPE; attention needs FINN or
  hand-written HLS. MLP-first de-risks the milestone.
- **Synthesis cost.** GLM proposes many candidates; the staged funnel
  (`infra/staged_eval.py`) keeps synthesis/bitstream counts small.
- **Full-model scale.** A single Qwen MLP exceeds a PYNQ-Z2 and needs tiling +
  streaming even on an Alveo; `orchestrate.py` quantifies this honestly.
- **Real GLM finetuning infra.** Assumes GLM weights + GPU for the LoRA path; the
  heuristic backend keeps everything runnable and testable without them.

---

## 8. What "done" means

- Single `(Qwen MLP block, large FPGA part)` task where the **test-time-finetuned
  GLM** authors a config that fits the part, keeps error under threshold, and beats
  the frozen GLM and the random-forest baseline on an equal evaluation budget —
  with the reward-vs-step curve and (where a toolchain/board exist) a real
  bitstream and on-board PASS. Full Qwen-2B remains the documented north star.
