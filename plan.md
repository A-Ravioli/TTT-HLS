## Build this in 24 hours: **BurnTTT**

**Project:** a test-time-adaptive compiler that burns a tiny neural network onto a real FPGA, using synthesis/simulation feedback to improve the hardware config for that exact model and board.

The right MVP is:

> **Tiny PyTorch/Keras model → quantize → hls4ml → HLS/RTL → FPGA bitstream → live inference → TTT/autotuner shows better config than baseline.**

Use **hls4ml**, not FINN, for the 24-hour version. hls4ml directly targets ML inference on FPGAs using HLS and supports traditional ML model conversion; its PyTorch frontend parses models through `torch.fx`, so traceable models are the safe zone. ([FastML Foundation][1]) FINN is conceptually perfect for quantized neural networks and custom dataflow FPGA accelerators, but it is more likely to eat your hackathon time in tooling goblin bites. ([Xilinx][2])

---

# 0. The final demo you should aim for

The demo should say:

> “We take a small transformer-MLP-style model, burn it into FPGA fabric, and run real inference. Then we show BurnTTT improving the generated hardware configuration using feedback from hls4ml/Vivado reports.”

You show three things:

1. **Model burned onto FPGA**

   * Python model and FPGA output match.

2. **Hardware optimization**

   * Naive config: slower / more resources / fails timing / worse accuracy.
   * BurnTTT config: better latency-resource tradeoff.

3. **TTT relevance**

   * The generator trains/adapts **during the run** on feedback from this exact model + FPGA target.

The exact honest phrasing:

> “The deployed FPGA model is fixed. The TTT happens in the compiler/generator, which adapts to this specific model and FPGA using synthesis and simulation feedback.”

That’s the whole spellbook.

---

# 1. Brutal scope decision

## Build this model

Use a tiny “transformer FFN block”:

```text
input vector → Linear → ReLU/GELU-ish → Linear → output vector
```

Concrete size:

```text
input_dim = 16
hidden_dim = 32 or 64
output_dim = 8 or 16
precision = 8-bit or 6-bit fixed point
```

Do **not** build attention.
Do **not** build a whole TPU.
Do **not** run an LLM on FPGA.
Do **not** write raw Verilog from scratch.

This tiny model is still defensible because transformer inference is dominated by matrix/vector operations, and you are demonstrating the compiler loop on a toy block.

---

# 2. Tooling choice

## Main path: hls4ml

Use hls4ml because it gives you:

```text
model → HLS C++ → Vivado/Vitis project → RTL/bitstream path
```

hls4ml has tutorial material for bitstream generation and PYNQ-style FPGA deployment, which is exactly the kind of boring-but-essential trail you want in a 24-hour build. ([GitHub][3])

## FPGA target

Use whichever board someone physically has and has already used before.

Best targets:

1. **PYNQ-Z2 / PYNQ-ZU**

   * Best for demoing Python host code + FPGA overlay.

2. **Kria KV260**

   * More capable, but slightly more deployment complexity.

3. **AWS F2**

   * Only if someone already knows the F2 flow.
   * Otherwise avoid. Cloud FPGA deployment can become paperwork wearing a helmet.

---

# 3. What TTT actually is in this project

You are not doing mystical LLM self-improvement.

You are doing this:

```text
config candidate
   ↓
hls4ml conversion
   ↓
C simulation / synthesis / resource reports
   ↓
parse feedback
   ↓
train/update small policy or surrogate model
   ↓
propose better config
   ↓
repeat
```

The trainable test-time policy sees:

```text
model shape
layer sizes
bitwidths
reuse factors
HLS strategy
resource usage
latency estimate
accuracy drop
compile/pass/fail status
```

It outputs:

```text
next config to try
```

This is **TTT as online compiler adaptation**.

---

# 4. The optimization knobs

You genuinely have enough knobs to make this real:

```text
per-layer weight precision
per-layer activation precision
accumulator precision
reuse factor
latency vs resource strategy
activation lookup table size
IO mode
pipeline/unroll-ish HLS choices
layer folding / parallelism
```

For hls4ml specifically, the obvious knobs are:

```python
Precision
ReuseFactor
Strategy = "Latency" or "Resource"
table_size for activations
```

The basic tradeoff:

```text
lower reuse factor → more parallelism → lower latency → more resources
higher reuse factor → less parallelism → higher latency → fewer resources
```

That alone is enough for a strong search/demo.

---

# 5. Repo structure

Make this repo:

```text
burnttt/
  README.md
  requirements.txt

  models/
    tiny_ffn.py
    train_toy_model.py
    export_model.py

  compiler/
    make_hls_config.py
    build_hls4ml_project.py
    run_hls.py
    parse_reports.py
    deploy_pynq.py

  ttt/
    config_space.py
    evaluate_config.py
    online_policy.py
    search.py

  dashboard/
    app.py

  scripts/
    00_train_model.py
    01_baseline_compile.py
    02_run_burnttt_search.py
    03_build_best_bitstream.py
    04_run_fpga_demo.py
```

---

# 6. MVP architecture

```text
                    ┌──────────────────┐
                    │ Tiny Keras model │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ hls4ml config    │
                    │ precision/reuse  │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Generate HLS     │
                    └────────┬─────────┘
                             │
        ┌────────────────────▼────────────────────┐
        │ Evaluate                                │
        │ C sim correctness                       │
        │ HLS synth latency/resources             │
        │ Optional bitstream / board inference    │
        └────────────────────┬────────────────────┘
                             │
                    ┌────────▼─────────┐
                    │ Online TTT policy│
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Better config    │
                    └──────────────────┘
```

---

# 7. Hour-by-hour plan

## Hours 0-1: lock scope and hardware path

Decide:

```text
Board: PYNQ-Z2 / Kria / other
Tool: hls4ml
Model: 16 → 32 → 8 dense block
Demo target: Python input → FPGA output
```

Create the repo and README immediately.

README opening:

```text
BurnTTT is a test-time-adaptive model-to-FPGA compiler.
Given a small neural network and an FPGA target, it searches over quantization and HLS configuration choices, uses simulation and synthesis feedback as test-time training signal, and emits a working FPGA accelerator.
```

Do not write a grand manifesto yet. Build the skeleton.

---

## Hours 1-3: create the tiny model

Use Keras if you want minimum friction with hls4ml.

Model:

```python
import tensorflow as tf

def make_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(16,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(8, activation="linear"),
    ])
```

Use synthetic data:

```python
X = random vectors
y = teacher_model(X)
```

You do **not** need a real dataset. Your core proof is hardware equivalence, not Kaggle glory.

Train quickly:

```python
model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=5)
model.save("artifacts/tiny_ffn.keras")
```

Also export golden test vectors:

```text
artifacts/test_inputs.npy
artifacts/golden_outputs.npy
```

---

## Hours 3-5: get basic hls4ml conversion working

Goal: generate an HLS project with one fixed config.

Baseline config:

```python
import hls4ml

config = hls4ml.utils.config_from_keras_model(
    model,
    granularity="name",
    default_precision="ap_fixed<16,6>",
    default_reuse_factor=1,
)

hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir="build/baseline",
    part="xc7z020clg400-1",  # PYNQ-Z2 example
)

hls_model.compile()
```

Then compare:

```python
y_hls = hls_model.predict(X_test)
y_py = model.predict(X_test)
```

Pass condition:

```text
mean absolute error < threshold
```

At this point, you have a software-level “burned model” proof.

---

## Hours 5-7: generate several configs manually

Before building TTT, make sure the compiler can vary configs.

Try configs like:

```text
A: ap_fixed<16,6>, ReuseFactor=1, Strategy=Latency
B: ap_fixed<12,4>, ReuseFactor=2, Strategy=Latency
C: ap_fixed<10,4>, ReuseFactor=4, Strategy=Resource
D: ap_fixed<8,3>,  ReuseFactor=8, Strategy=Resource
```

For each config, record:

```text
accuracy/error vs Python
estimated latency
LUT usage
FF usage
DSP usage
BRAM usage
compile success
```

Save as:

```text
results/runs.csv
```

This CSV is your lifeblood. The dashboard, TTT policy, and demo all depend on it.

---

## Hours 7-9: report parser

Write `parse_reports.py`.

Extract from hls4ml/Vivado reports:

```text
Latency min/max
Initiation interval
BRAM
DSP
FF
LUT
Timing status if available
```

Output:

```json
{
  "latency_cycles": 123,
  "ii": 1,
  "bram": 4,
  "dsp": 32,
  "ff": 1024,
  "lut": 2048,
  "passed": true
}
```

Do this with ugly regex. This is not the time for cathedral architecture. This is a crowbar hour.

---

## Hours 9-12: build the online TTT/autotuner

Implement a small policy.

Simplest version:

```python
from sklearn.ensemble import RandomForestRegressor
```

The policy predicts reward from config.

Config vector:

```python
[
  weight_bits,
  activation_bits,
  int_bits,
  reuse_factor_layer1,
  reuse_factor_layer2,
  strategy_latency_bool,
]
```

Reward:

```python
reward = (
    1000 * passed
    - 5.0 * accuracy_error
    - 0.01 * latency_cycles
    - 0.1 * dsp
    - 0.01 * lut
    - 0.05 * bram
)
```

Online loop:

```python
history = []

# Seed with random/manual configs
for config in seed_configs:
    result = evaluate_config(config)
    history.append((config, result))

for round in range(5):
    policy.fit(X_configs, y_rewards)

    candidates = sample_random_configs(100)
    predicted_rewards = policy.predict(candidates)

    top = select_top_k(candidates, predicted_rewards, k=3)

    for config in top:
        result = evaluate_config(config)
        history.append((config, result))

save history
```

This is your TTT.

Call it:

> “At test time, BurnTTT trains a per-model, per-board surrogate policy on real synthesis/simulation feedback, then uses the adapted policy to propose better hardware configurations.”

That is honest and strong.

---

## Hours 12-14: make the baseline comparison

You need a clean chart.

Compare:

```text
Baseline:
  hls4ml default config

Random search:
  same number of trials, no learned policy

BurnTTT:
  online policy trained on feedback
```

Metrics:

```text
best valid reward found
best latency under resource budget
best accuracy/resource tradeoff
number of failed configs avoided
```

The demo chart can be simple:

```text
x-axis: config attempts
y-axis: best reward so far
lines: random vs BurnTTT
```

This is your “TTT did something” visual.

---

## Hours 14-18: real FPGA deployment

Now choose the best config and compile the bitstream.

Do only **one** real bitstream build if possible. FPGA implementation is slow and cranky.

Deployment goal:

```text
host Python sends input vector
FPGA returns output vector
compare against Python/hls4ml golden output
print PASS
```

Demo output:

```text
Input: [ ... ]
Python output: [ ... ]
FPGA output: [ ... ]
Max error: 0.03125
PASS
```

That’s enough. A physical FPGA saying “PASS” is worth 400 slides.

---

## Hours 18-20: dashboard

Build a tiny Streamlit app.

```bash
pip install streamlit
```

Dashboard panels:

1. **Current best config**
2. **Resource table**
3. **Latency/error chart**
4. **Random vs BurnTTT best reward**
5. **FPGA output match**

The dashboard should look like a cockpit, not a spreadsheet that got lost in a laundromat.

Minimum Streamlit:

```python
import streamlit as st
import pandas as pd

df = pd.read_csv("results/runs.csv")

st.title("BurnTTT: Test-Time Adaptive FPGA Model Burning")

st.subheader("Search Results")
st.dataframe(df)

st.subheader("Best Reward Over Time")
df["best_reward_so_far"] = df["reward"].cummax()
st.line_chart(df["best_reward_so_far"])

st.subheader("Best Config")
st.json(df.sort_values("reward", ascending=False).iloc[0].to_dict())
```

---

## Hours 20-22: polish the story

Your README should have:

```text
What it is
Why TTT matters
How the compiler adapts
What FPGA target we used
Results
How to reproduce
```

Add a diagram:

```text
Model → hls4ml → HLS reports → TTT policy → better config → FPGA
```

Add this line prominently:

> “BurnTTT does not train the FPGA model after deployment. It adapts the compiler at test time before deployment.”

That preempts the obvious objection.

---

## Hours 22-24: demo script and contingency

Create a 2-minute demo:

### Demo script

1. “Here is a tiny transformer-MLP-style model.”
2. “Naive hls4ml can compile it, but the hardware config is arbitrary.”
3. “BurnTTT tries configs, reads synthesis/simulation feedback, and trains a local policy for this exact model and FPGA.”
4. “The adapted generator finds a better config.”
5. “We burn that config to FPGA and run live inference.”
6. “FPGA output matches Python.”

### If FPGA deployment breaks

Say:

> “The full bitstream path is running, but here is the generated HLS/RTL, C simulation, synthesis reports, and board deployment script. The core contribution is the test-time adaptive compiler loop.”

But fight hard to get at least one board inference working.

---

# 8. Concrete implementation details

## Config space

Use this:

```python
BITWIDTHS = [8, 10, 12, 14, 16]
INT_BITS = [3, 4, 5, 6]
REUSE_FACTORS = [1, 2, 4, 8, 16]
STRATEGIES = ["Latency", "Resource"]
```

Config object:

```python
@dataclass
class BurnConfig:
    weight_bits: int
    activation_bits: int
    int_bits: int
    reuse_dense_1: int
    reuse_dense_2: int
    strategy: str
```

Precision string:

```python
f"ap_fixed<{bits},{int_bits}>"
```

Layer config:

```python
config["LayerName"]["dense"]["Precision"]["weight"] = precision
config["LayerName"]["dense"]["Precision"]["result"] = precision
config["LayerName"]["dense"]["ReuseFactor"] = reuse_dense_1
```

---

# 9. The reward function

Use this:

```python
def reward(result):
    if not result["compile_success"]:
        return -1000

    if result["max_error"] > 0.25:
        return -500 - 100 * result["max_error"]

    return (
        1000
        - 0.05 * result["latency_cycles"]
        - 1.0 * result["dsp"]
        - 0.05 * result["lut"]
        - 0.1 * result["bram"]
        - 50.0 * result["max_error"]
    )
```

Add resource limits if you know the board:

```python
if result["dsp_pct"] > 80:
    reward -= 300
if result["lut_pct"] > 80:
    reward -= 300
```

---

# 10. The “TTT” module

This is enough:

```python
class OnlineTTTPolicy:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=64)
        self.has_fit = False

    def fit(self, history):
        X = [config_to_vector(c) for c, r in history]
        y = [r["reward"] for c, r in history]
        if len(X) >= 5:
            self.model.fit(X, y)
            self.has_fit = True

    def propose(self, n=3):
        candidates = sample_random_configs(200)
        if not self.has_fit:
            return random.sample(candidates, n)

        preds = self.model.predict([config_to_vector(c) for c in candidates])
        ranked = sorted(zip(candidates, preds), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[:n]]
```

That is not fancy, but it is demoable.

If challenged, say:

> “We chose a small online policy instead of LoRA-updating the full LLM because FPGA tool feedback is structured and expensive. The principle is the same: update a trainable component at test time using self-supervised feedback from the test instance.”

---

# 11. What to do with the 8×H100s

Be honest: FPGA tools won’t use H100s much.

Use the H100s for:

```text
parallel quantization simulations
parallel model variants
many candidate evaluations before HLS
optional LoRA/LLM code-repair experiments
dashboard/backend serving
```

But the true bottleneck is HLS/Vivado. So stage your evaluation:

```text
Stage 1: cheap Python quantized simulation for hundreds of configs
Stage 2: hls4ml compile/C-sim for top 20
Stage 3: HLS synthesis for top 5
Stage 4: bitstream for top 1
```

Do not synthesize 100 configs. That is how you summon the Vivado kraken.

---

# 12. Stretch goals, only after MVP works

## Stretch 1: LLM patch repair

If hls4ml emits broken HLS or a config fails, ask an LLM to classify why:

```text
timing issue
resource issue
precision/accuracy issue
compile issue
```

Then propose a config mutation.

This makes it more agentic.

## Stretch 2: baked weights

Generate a small report showing:

```text
weights are compiled into the generated HLS/RTL constants / initialized arrays
```

Frame it as:

> “The network is physically specialized into the FPGA bitstream.”

## Stretch 3: transformer-ish block

Name the model:

```text
TinyFFNBlock
```

Then say it is the feedforward sub-block of a transformer. Keep dimensions tiny.

## Stretch 4: FINN backend

Only if hls4ml works early. FINN is the more “pure custom dataflow” path, but not the right first bite.

---

# 13. What your final results table should look like

Make this table in the README and dashboard:

| Method            | Precision | Reuse | Strategy | Max Error | Latency | DSP |  LUT | Fits?   |
| ----------------- | --------: | ----: | -------- | --------: | ------: | --: | ---: | ------- |
| Default hls4ml    |    16-bit |     1 | Latency  |      0.01 |     100 |  60 | 9000 | Yes     |
| Resource baseline |     8-bit |    16 | Resource |      0.18 |     850 |   8 | 3000 | Yes     |
| Random best       |    10-bit |     8 | Resource |      0.09 |     420 |  14 | 4100 | Yes     |
| **BurnTTT best**  |    12-bit |     4 | Latency  |      0.04 |     210 |  24 | 5000 | **Yes** |

The numbers can be whatever the tools actually produce. The shape of the story matters.

---

# 14. Judging pitch

Use this:

> “BurnTTT is a test-time-adaptive compiler for burning neural networks onto FPGAs. Instead of using a fixed compiler recipe, it treats each model and FPGA as a new test-time task. It generates candidate hardware configs, runs simulation and synthesis, parses real tool feedback, trains a small online policy on that feedback, and emits a better FPGA implementation. We demonstrate it by compiling a tiny transformer-MLP-style model to a real FPGA and showing the adapted compiler finds a better latency/resource/accuracy point than baseline search.”

Then the dagger:

> “The FPGA model itself is fixed after deployment. The intelligence is in the compiler loop before deployment.”

That makes you sound precise instead of hand-wavy.

---

# 15. Absolute minimum if everything goes wrong

You need these artifacts no matter what:

```text
1. tiny model
2. hls4ml-generated HLS project
3. C simulation showing model equivalence
4. parsed synthesis report
5. search CSV of configs
6. chart showing BurnTTT > random/default
7. deployment script for FPGA
```

Even without live FPGA, this is a coherent project. With live FPGA, it becomes nasty in the best way.

---

# Final recommendation

Build **BurnTTT with hls4ml**, not FINN, not raw RTL, not a whole TPU.

Your 24-hour MVP:

```text
Tiny transformer-MLP block
→ hls4ml FPGA implementation
→ online TTT/autotuner over precision + reuse + strategy
→ one best config deployed to real FPGA
→ dashboard proving improvement
```

This is the version that is:

* actually buildable,
* genuinely FPGA-based,
* TTT-relevant without overclaiming,
* and impressive enough to feel like a tiny Etched gremlin escaped into programmable logic.

[1]: https://fastmachinelearning.org/hls4ml/?utm_source=chatgpt.com "Welcome to hls4ml's documentation!"
[2]: https://xilinx.github.io/finn/?utm_source=chatgpt.com "FINN | finn"
[3]: https://github.com/fastmachinelearning/hls4ml-tutorial/blob/main/part7a_bitstream.ipynb?utm_source=chatgpt.com "hls4ml-tutorial/part7a_bitstream.ipynb at main"
