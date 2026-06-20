# BurnTTT Project Context

This file is durable context for future coding agents working in this repo. Read
`README.md` for the user-facing overview and `plan.md` for the north-star plan;
use this file as the condensed memory of project intent, prior chat decisions, and
current implementation state.

## Mission

BurnTTT is a test-time-trained GLM compiler for neural-network inference on
FPGAs. The north star is Qwen-2B inference on FPGA at absolute peak performance,
with GLM adapting at test time so it writes better compiler artifacts for the
specific `(model block, FPGA part)` task.

The key framing is:

> The deployed FPGA logic is fixed. Test-time training updates the GLM
> compiler/generator, not Qwen, not the toy model, and not the deployed bitstream.

The workload model is fixed and only used as the thing being compiled. The final
accelerator is also fixed after build. All adaptation happens in the generator
that authors hardware configs or custom HLS.

## Corrected Mental Model

Several earlier chats corrected misunderstandings:

- We do not TTT or LoRA-finetune Qwen or the deployed neural model.
- We do TTT the GLM that writes the compiler/synthesizer artifact.
- Phase 1-3 config mode is legitimate verifier-driven TTT, but it is narrow: GLM
  authors `BurnConfig` / `BlockConfig` knobs and hls4ml remains the fixed backend.
- For absolute peak performance, the desired path is compiler-author mode: GLM
  writes and iteratively edits custom Vitis HLS C++ and host/orchestration code.
- Config TTT is still the baseline and CI-friendly wedge; custom HLS is the
  mission for full Qwen-2B performance.
- The user prefers continuous RL-style optimization over pure SFT. GRPO was added
  as the desired direction for pushing compiler performance.

## Two Modes

### Config-Author Mode

Artifact: `BurnConfig` or `BlockConfig`.

GLM proposes new complete hardware knob configs each round. It does not patch a
stored hls4ml YAML. The only edit-like path is repair after a failed compile.

Flow:

```text
FpgaTask -> GLMGenerator -> BurnConfig -> make_hls_config -> hls4ml
         -> sim/synth/estimate -> reward -> trajectories -> LoRA/GRPO TTT
```

This mode is implemented and used by `scripts/07_ttt_finetune_glm.py` and
`scripts/08_eval_glm_generator.py`. It compares default hls4ml, random search,
random forest/BurnTTT baseline, frozen GLM, and GLM+TTT.

### Compiler-Author Mode

Artifact: `KernelBundle`.

GLM authors HLS source bundles and iteratively repairs/improves them against
compiler, cosim, timing, resource, and throughput feedback.

Flow:

```text
FpgaTask -> GLMCompilerAgent -> KernelBundle -> Vitis HLS / Vivado / board
         -> cosim + timing + tokens/sec reward -> traces -> HLS TTT
```

This is Phase 4+ and the path to peak performance. It is scaffolded in the repo,
but real measured wins require toolchain/board runs.

## Current Implementation State

Config mode:

- `ttt/config_space.py` defines `BurnConfig`, `BlockConfig`, `LayerKnobs`, and
  `KernelBundle`.
- `compiler/make_hls_config.py` expands configs into hls4ml config dicts.
- `ttt/evaluate_config.py` converts, simulates, optionally synthesizes, estimates
  hardware, and computes reward.
- `ttt/reward.py` contains board budgets and the config reward. It includes AWS
  EC2 F2 / VU47P as `xcvu47p-fsvh2892-2-e`.
- `ttt/search.py` runs default, random, RF/BurnTTT, frozen GLM, and GLM+TTT.
- `glm/agent.py`, `glm/prompts/`, and `glm/parsing/` implement config propose and
  repair.
- `glm/finetune/trainer.py` supports SFT, DPO, and GRPO-controlled LoRA updates
  for HF backends. Prime API generation cannot update weights locally.
- `glm/finetune/grpo.py` implements group-relative advantages and policy-gradient
  loss for config completions.

Compiler-author mode:

- `compiler/ir.py` builds a small block IR from Qwen decomposition.
- `compiler/golden.py` produces golden I/O for cosim.
- `compiler/hls_build.py`, `compiler/cosim.py`, `compiler/vivado.py`, and
  `compiler/host.py` form the custom HLS evaluation scaffold.
- `compiler/kernel_lib/swiglu_mlp.py` provides a seed/customizable SwiGLU MLP
  kernel template.
- `ttt/evaluate_hls.py` evaluates a `KernelBundle`.
- `ttt/reward_hls.py` gates on HLS compile, cosim, timing, and accuracy; successful
  rewards are dominated by `tokens_per_sec`.
- `glm/agent_hls.py` implements propose, repair, and iterate for HLS bundles.
- `glm/prompts/hls_templates.py` and `glm/parsing/hls.py` support HLS prompting
  and extraction.
- `glm/finetune/trainer_hls.py` and `glm/finetune/dataset_hls.py` are the HLS TTT
  scaffolding.
- `scripts/11_glm_author_hls.py` is the Phase 4 single-block HLS author loop.

Prime / GPU operations:

- `glm/serving.py` has `HFBackend`, `HeuristicBackend`, and `PrimeBackend`.
- `PrimeBackend` uses the OpenAI-compatible Prime Inference API for generation
  only. It cannot do local LoRA/GRPO weight updates.
- For real TTT, use `HFBackend` on a GPU pod with `BURN_GLM_BACKEND=hf`.
- `infra/prime_pod.py`, `scripts/16_prime_pod.py`, `scripts/17_prime_run_ttt.sh`,
  and `scripts/18_preflight_pod.py` manage Prime GPU pods.
- `glm/model_specs.py` includes model-to-pod hints. `zai-org/GLM-4.7-Flash` is the
  single-H100 GRPO candidate used as the practical ~31B GLM. GLM-5.x / 5.2 are
  much larger MoE targets and need multi-GPU pods.

## Important Environment Variables

Do not commit real `.env` or `.env.pod` values.

Common:

```bash
BURN_TARGET_PART=xcvu47p-fsvh2892-2-e
BURN_GLM_BACKEND=hf        # real local/pod weights for LoRA/GRPO
BURN_GLM_BACKEND=prime     # Prime Inference API only, no local weight updates
BURN_GLM_MODEL=zai-org/GLM-4.7-Flash
BURN_GLM_MAX_SEQ_LEN=2048
BURN_GLM_MAX_TOKENS=8192
```

GRPO:

```bash
BURN_TTT_USE_GRPO=1
BURN_TTT_USE_SFT=0
BURN_TTT_USE_DPO=0
BURN_TTT_STEPS_PER_ROUND=8
BURN_TTT_LR=2e-5
BURN_TTT_RUN_NAME=glm_ttt_grpo
```

Prime pod:

```bash
BURN_GLM_MODEL_POD=zai-org/GLM-4.7-Flash
GPU_TYPE=H100_80GB
GPU_COUNT=1
ROUNDS=10
CANDIDATES=6
KEEP_POD=0
WANDB_PROJECT=burnttt
WANDB_RUN_NAME=grpo-glm47flash-f2-vu47p
```

## Recent Run Context

Previous chats attempted an aggressive GRPO run:

- Target: AWS EC2 F2 / AMD Virtex UltraScale+ HBM VU47P
  (`xcvu47p-fsvh2892-2-e`).
- Model: `zai-org/GLM-4.7-Flash`, chosen as the closest single-H100 practical GLM
  candidate around the user's requested 27B scale.
- Training intent: GRPO only, no SFT, 10 rounds x 6 candidates, 8 GRPO steps per
  round, learning rate `2e-5`.
- A missing `import os` in `ttt/search.py` caused one run to crash; it has been
  fixed.
- SSH/local-session drops interrupted a retry before GRPO started, so a detached
  `nohup` run was launched on the pod in the prior chat. Treat run status as
  potentially stale; check the pod or wandb before assuming completion.
- The run still optimized hls4ml config artifacts, not full HLS compiler bundles.
  Use `scripts/11_glm_author_hls.py` for the custom HLS path.

## Commands

Install:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-glm.txt
```

Core config pipeline:

```bash
python scripts/00_train_model.py
python scripts/01_baseline_compile.py
python scripts/08_eval_glm_generator.py
python scripts/07_ttt_finetune_glm.py
streamlit run dashboard/app.py
```

Qwen / block demos:

```bash
python scripts/05_ingest_qwen.py
python scripts/05_ingest_qwen.py --export
python scripts/09_block_to_fpga_demo.py --block qwen_mlp
```

HLS authoring:

```bash
python scripts/11_glm_author_hls.py --rounds 6 --part xcu250-figd2104-2l-e
python scripts/11_glm_author_hls.py --rounds 6 --part xcvu47p-fsvh2892-2-e --run-vivado
```

Prime pod run example:

```bash
SKIP_PREFLIGHT=1 \
BURN_GLM_MODEL_POD=zai-org/GLM-4.7-Flash \
BURN_TARGET_PART=xcvu47p-fsvh2892-2-e \
BURN_TTT_USE_GRPO=1 \
BURN_TTT_USE_SFT=0 \
BURN_TTT_STEPS_PER_ROUND=8 \
BURN_TTT_LR=2e-5 \
BURN_TTT_RUN_NAME=glm_ttt_grpo \
GPU_TYPE=H100_80GB \
GPU_COUNT=1 \
ROUNDS=10 \
CANDIDATES=6 \
KEEP_POD=0 \
WANDB_RUN_NAME=grpo-glm47flash-f2-vu47p \
bash scripts/17_prime_run_ttt.sh
```

Tests:

```bash
python -m pytest tests -q
python -m pytest tests/test_ttt_dataset.py tests/test_reward_hls.py tests/test_hls_parsing.py -q
```

## Engineering Rules For Future Agents

- Preserve config mode. It is the regression baseline and CI-friendly path.
- Do not pretend hls4ml config TTT is the final compiler-writing result. It is a
  useful baseline and warm start.
- For peak performance work, prioritize `KernelBundle`, HLS cosim, timing closure,
  throughput, and board/runtime validation.
- Accuracy cannot drop significantly. Config mode uses `MAX_ERROR_THRESHOLD=0.25`;
  HLS mode is tighter by default with `HLS_MAX_ERROR_THRESHOLD=0.01`.
- Real weight updates require local/pod HF weights and PEFT/torch. Prime API
  generation alone cannot perform LoRA/GRPO updates.
- Avoid leaking secrets. Never paste `.env`, `.env.pod`, API keys, SSH keys, or
  wandb keys into docs or commits.
- The repo may have local uncommitted work. Do not revert user changes unless
  explicitly asked.
- If launching long Prime jobs, prefer detached/nohup or robust logging so SSH
  session drops do not kill the run.

## Best Next Work

If asked to continue implementation, the highest-value path is the Phase 4
custom-HLS vertical slice:

1. Make `scripts/11_glm_author_hls.py` robust on the F2/VU47P target.
2. Ensure HLS traces contain full sources, prompts, errors, cosim metrics, timing,
   and reward for RL/DPO/GRPO training.
3. Add GRPO support for HLS `KernelBundle` completions, not just config JSON.
4. Run a small HLS loop without Vivado for CI, then a real Vitis/Vivado run where
   the toolchain exists.
5. Compare custom HLS against best hls4ml config TTT using `tokens_per_sec`,
   timing, resources, and max error.

