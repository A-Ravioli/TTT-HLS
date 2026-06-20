# 🔥 BurnTTT — a test-time-adaptive model-to-FPGA compiler

BurnTTT takes a tiny neural network, compiles it onto FPGA fabric with
[`hls4ml`](https://github.com/fastmachinelearning/hls4ml), and **adapts the
compiler at test time** — searching precision / reuse / strategy knobs using
real quantization-accuracy and resource feedback from *this exact model and FPGA
target* — to find a hardware configuration that beats the naive default and a
random-search baseline.

```
tiny Keras model → quantize → hls4ml → HLS/RTL project → (synthesis) → live inference
                          ▲
                          └── BurnTTT online policy adapts the compiler config
```

## 1. What BurnTTT is

Picking the hardware-generation config for an FPGA model is a search problem:
too much precision or too little reuse and the design **won't fit the board**;
too little precision and the **accuracy drifts**. The sweet spot depends on the
specific model *and* the specific FPGA part.

BurnTTT treats each `(model, FPGA part)` pair as a fresh task and runs a small
**online policy** that:

1. Starts from hand-picked seed configs spanning the accuracy/resource spectrum.
2. Evaluates each config: hls4ml conversion → bit-accurate prediction error →
   (synthesis if available, else an analytical resource estimate) → reward.
3. Trains a `RandomForestRegressor` surrogate on the `(config → reward)` pairs
   seen *so far*.
4. Proposes the next configs by scoring a candidate pool (random exploration +
   single-step neighbors of the best-known configs) and picking the predicted
   best — a surrogate-guided hill-climb.
5. Repeats for a configurable number of rounds and writes everything to
   `results/runs.csv`.

A **random-search baseline** runs on an equal evaluation budget so the dashboard
can compare `default hls4ml` vs `random search` vs `BurnTTT`.

## 2. The honest TTT framing

> **BurnTTT does not train the FPGA model after deployment. It adapts the compiler
> at test time before deployment, using simulation and synthesis feedback from the
> specific model and FPGA target.**

The deployed FPGA network is *fixed*. The "test-time training" is in the
**compiler loop**: the surrogate policy is (re)fit online, during the run, on
feedback measured from the real model + target. This is test-time adaptation of
the *generator*, not online learning of the deployed weights.

## 3. Install

Requires Python 3.10 or 3.11 (hls4ml 1.0.0 pins TensorFlow ≤ 2.14).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A C++ compiler (`g++`) is needed for hls4ml's bit-accurate C simulation (used to
measure quantization error). Vivado/Vitis HLS is **optional** — without it,
BurnTTT uses an analytical resource/latency estimator and still produces a full
comparison.

## 4. Run the full demo

```bash
python scripts/00_train_model.py                 # train + export tiny model
python scripts/01_baseline_compile.py            # compile the default hls4ml config
python scripts/02_run_burnttt_search.py          # baseline + BurnTTT + random search
streamlit run dashboard/app.py                   # visualize the results
```

Optional hardware path (graceful stubs off-board):

```bash
python scripts/03_build_best_bitstream.py        # build best config's HLS/bitstream project
python scripts/04_run_fpga_demo.py               # run on a PYNQ board, or software-equivalence demo
```

Tune the search:

```bash
python scripts/02_run_burnttt_search.py --rounds 4 --candidates-per-round 3
python scripts/02_run_burnttt_search.py --synth   # run real HLS synthesis per config (needs Vivado)
```

### Example result (PYNQ-Z2, no Vivado — analytical estimates)

| Method            | Best config            | Max error | DSP  | LUT  | Fits board? | Reward |
|-------------------|------------------------|-----------|------|------|-------------|--------|
| Default hls4ml    | `w16a16i6_r1-1_Lat`    | 0.058     | 1536 | 69520| **No**      | -4916  |
| Random search     | `w16a16i4_r16-16_Res`  | 0.016     | 59   | 4720 | Yes         | 699.6  |
| **BurnTTT**       | `w12a12i3_r16-16_Res`  | 0.117     | 59   | 3640 | Yes         | **748.6** |

The naive default **does not fit** the FPGA (1536 DSPs ≫ 220 available). BurnTTT
discovers that dropping to 12-bit with *more fractional bits* (`int_bits=3`) and
full layer reuse keeps the error well under threshold while using the least
fabric — beating even a lucky random hit on a 16-bit config.

## 5. Artifacts produced

```
artifacts/tiny_ffn.keras        # trained Keras model
artifacts/test_inputs.npy       # golden test inputs
artifacts/golden_outputs.npy    # golden float32 outputs
artifacts/best_config.json      # best config found (written by script 03)
build/baseline/...              # default-config HLS project
build/eval/...                  # per-config HLS projects from the search
build/best/...                  # best config's HLS/bitstream project
results/runs.csv                # every evaluation (method, config, error, resources, reward)
```

## 6. Targeting a specific FPGA part / board

The default part is the PYNQ-Z2 (`xc7z020clg400-1`). Override with env vars — no
part is hardcoded:

```bash
export BURN_TARGET_PART=xcu250-figd2104-2L-e   # e.g. an Alveo U250
export BURN_BOARD=pynq-z2                       # board name for VivadoAccelerator bitstreams
python scripts/02_run_burnttt_search.py
```

The on-board resource budget used for the "fits board?" check lives in
`ttt/reward.py` (`BOARD_BUDGET`) and reflects the PYNQ-Z2; adjust it for other
boards.

## 7. The dashboard

```bash
streamlit run dashboard/app.py
```

Shows: headline metrics, the best config + resource utilization vs the board
budget, **best-reward-over-evaluations** (random vs BurnTTT) on an equal budget,
a method comparison table, the accuracy-vs-latency tradeoff scatter, the full
results table, and an explainer of what "TTT" means here.

## 8. Reward function

Implemented in `ttt/reward.py` (see plan.md §9). In words: a compile failure is
the worst outcome; accuracy drift past a threshold is heavily penalized; designs
over the board's resource budget are penalized; otherwise reward favors low
latency and low resource usage.

```python
def reward(result):
    if not result["compile_success"]:
        return -1000
    if result["max_error"] is not None and result["max_error"] > 0.25:
        return -500 - 100 * result["max_error"]
    return (
        1000
        - 0.05 * latency_cycles
        - 1.0  * dsp
        - 0.05 * lut
        - 0.1  * bram
        - 50.0 * max_error
    )  # plus penalties for exceeding the board resource budget
```

## 9. Config space

`ttt/config_space.py` defines `BurnConfig(weight_bits, activation_bits, int_bits,
reuse_dense_1, reuse_dense_2, strategy)` over:

```
bits:   8, 10, 12, 14, 16
int:    3, 4, 5, 6           (must be < bitwidth)
reuse:  1, 2, 4, 8, 16
strat:  Latency, Resource
```

Precision maps to `ap_fixed<bits,int>`; `int_bits` is the integer part, so fewer
integer bits = more fractional bits = lower quantization error (until integer
overflow).

## 10. Repository layout

```
models/      tiny_ffn.py, train_toy_model.py, export_model.py
compiler/    make_hls_config, build_hls4ml_project, run_hls, parse_reports,
             estimate_resources, deploy_pynq
ttt/         config_space, reward, evaluate_config, online_policy, search
dashboard/   app.py (Streamlit)
scripts/     00_train_model … 04_run_fpga_demo
tests/       config-serialization + reward + policy smoke tests
paths.py     shared paths / logging / target-part helpers
```

Run the tests with:

```bash
python -m pytest tests -q
```

## 11. Known limitations

- **Tiny model.** A 16→64→8 feedforward block — intentionally small so
  conversion/synthesis is fast. Not a transformer or LLM.
- **Estimates without Vivado.** Off-toolchain, latency/resources are *analytical
  estimates* (`compiler/estimate_resources.py`); accuracy (`max_error`) is always
  real, bit-accurate hls4ml output. Install Vivado and pass `--synth` for real
  synthesis numbers parsed from the reports.
- **Surrogate is simple.** A random forest over a small discrete space; the value
  is the *online adaptation loop*, not a fancy optimizer.
- **FPGA deployment is optional.** Without a PYNQ board, script 04 runs a
  software bit-accurate equivalence demo and explains the on-board steps.

## 12. What to say in a hackathon demo

1. "Here's a tiny model. The **default** FPGA compile uses 1536 DSPs — it
   **doesn't fit** a PYNQ-Z2."
2. "BurnTTT runs an **online policy** that adapts the *compiler* at test time
   using accuracy + resource feedback from this exact model and board."
3. Show the dashboard: **BurnTTT's best-reward curve climbs above random** on an
   equal budget, and its best config **fits the board** with the lowest fabric
   cost.
4. "The deployed network is fixed — the **test-time training is in the
   compiler loop**, before deployment."
5. (If Vivado/PYNQ present) "And here it is running on real silicon."
