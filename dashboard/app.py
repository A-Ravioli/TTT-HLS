"""BurnTTT cockpit dashboard.

    streamlit run dashboard/app.py

Reads results/runs.csv and visualizes the search: best config, resource usage,
best-reward-over-attempts for random vs BurnTTT, and the accuracy/latency
tradeoff. Also explains what "TTT" means in this project.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paths import RUNS_CSV  # noqa: E402
from ttt.reward import get_board_budget  # noqa: E402

st.set_page_config(page_title="BurnTTT", page_icon="🔥", layout="wide")

METHOD_LABELS = {
    "default": "Default hls4ml",
    "random": "Random search",
    "burnttt": "Random forest (baseline)",
    "glm": "GLM generator (frozen)",
    "glm_ttt": "GLM generator (test-time finetuned)",
}

# Methods compared on the reward curve / tables, in plot order.
COMPARE_METHODS = ("random", "burnttt", "glm", "glm_ttt")


@st.cache_data
def load_runs(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def best_row(df: pd.DataFrame) -> pd.Series | None:
    valid = df[df["compile_success"] == True]  # noqa: E712
    if valid.empty:
        return None
    return valid.sort_values("reward", ascending=False).iloc[0]


def reward_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Best-reward-so-far per method, aligned on a shared attempt axis."""
    curves = {}
    for method in COMPARE_METHODS:
        sub = df[df["method"] == method].sort_values("attempt")
        if sub.empty:
            continue
        curves[METHOD_LABELS[method]] = sub["reward"].cummax().reset_index(drop=True)
    if not curves:
        return pd.DataFrame()
    out = pd.DataFrame(curves)
    out.index.name = "evaluation #"
    return out


def main() -> None:
    st.title("🔥 BurnTTT: a GLM that compiles models onto FPGAs, finetuned at test time")
    st.caption(
        "An LLM (GLM) **authors** the model-to-FPGA hardware config and is "
        "**finetuned at test time** on synthesis/simulation feedback from this exact "
        "model block + FPGA part. North-star target: Qwen-2B on FPGA."
    )

    with st.expander("What does \"test-time training\" mean here? (read me)", expanded=False):
        st.markdown(
            """
**The deployed FPGA logic is fixed. The test-time training happens in the
*generator*: GLM's own (LoRA) weights are updated during the run, on feedback
from this specific (model block, FPGA part) task, so it authors better hardware
the longer it works on the task.**

For each new task we (1) ask GLM to author a hardware-generation config, (2)
evaluate it with bit-accurate simulation + synthesis/resource estimation, (3)
append the feedback, and (4) take a LoRA gradient step on the high-reward
trajectories. Compare strategies below on an equal evaluation budget: the default
hls4ml config, random search, the random-forest **baseline**, the **frozen GLM**
generator, and the **test-time-finetuned GLM** generator.

*(Off-GPU, a heuristic backend stands in for GLM and is adapted analogously, so
the same climb is demonstrable without weights.)*
            """
        )

    if not Path(RUNS_CSV).exists():
        st.warning(
            f"No results found at `{RUNS_CSV}`.\n\n"
            "Run the pipeline first:\n"
            "```\npython scripts/00_train_model.py\n"
            "python scripts/02_run_burnttt_search.py --rounds 3 --candidates-per-round 3\n```"
        )
        st.stop()

    df = load_runs(str(RUNS_CSV), Path(RUNS_CSV).stat().st_mtime)
    df["method_label"] = df["method"].map(METHOD_LABELS).fillna(df["method"])

    # --- Headline metrics ---------------------------------------------------
    best = best_row(df)
    n_total = len(df)
    n_failed = int((df["compile_success"] != True).sum())  # noqa: E712
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Configs evaluated", n_total)
    c2.metric("Failed configs", n_failed)
    if best is not None:
        c3.metric("Best reward", f"{best['reward']:.1f}")
        c4.metric("Best max-error", f"{best['max_error']:.4f}" if pd.notna(best["max_error"]) else "n/a")

    estimated = bool(df.get("estimated_hw", pd.Series([False])).any())
    if estimated:
        st.info(
            "Latency/resource numbers are **analytical estimates** (no Vivado/Vitis "
            "toolchain detected). Accuracy (max_error) is real, bit-accurate hls4ml "
            "output. Install Vivado and rerun with `--synth` for real synthesis numbers."
        )

    # --- Best config + resources -------------------------------------------
    st.subheader("Best configuration found")
    if best is not None:
        left, right = st.columns([1, 1])
        with left:
            st.json(
                {
                    "method": METHOD_LABELS.get(best["method"], best["method"]),
                    "precision_bits": int(best["weight_bits"]),
                    "int_bits": int(best["int_bits"]),
                    "reuse_dense_1": int(best["reuse_dense_1"]),
                    "reuse_dense_2": int(best["reuse_dense_2"]),
                    "strategy": best["strategy"],
                    "max_error": float(best["max_error"]) if pd.notna(best["max_error"]) else None,
                    "latency_cycles": None if pd.isna(best["latency_cycles"]) else int(best["latency_cycles"]),
                    "reward": float(best["reward"]),
                    "fits_board": bool(best.get("fits_board", False)),
                }
            )
        with right:
            part = best.get("target_part") if "target_part" in best else None
            board_budget = get_board_budget(part)
            res_rows = []
            for field, budget in board_budget.items():
                used = best.get(field)
                used = None if pd.isna(used) else int(used)
                pct = f"{100 * used / budget:.1f}%" if used is not None else "n/a"
                res_rows.append({"resource": field.upper(), "used": used, "budget": budget, "utilization": pct})
            st.markdown(f"**Resource usage vs board budget** ({part or 'default part'})")
            st.dataframe(pd.DataFrame(res_rows), hide_index=True, use_container_width=True)

    # --- Reward over attempts ----------------------------------------------
    st.subheader("Best reward over evaluations: Random vs BurnTTT")
    curve = reward_curve(df)
    if not curve.empty:
        st.line_chart(curve)
        st.caption(
            "Each line is the best valid reward found so far, on an equal evaluation "
            "budget. BurnTTT's surrogate concentrates evaluations on promising configs."
        )
    else:
        st.write("Not enough data to plot the comparison yet.")

    # --- Method comparison table -------------------------------------------
    st.subheader("Method comparison (best valid config per method)")
    comp_rows = []
    for method in ("default", *COMPARE_METHODS):
        sub = df[(df["method"] == method) & (df["compile_success"] == True)]  # noqa: E712
        if sub.empty:
            continue
        row = sub.sort_values("reward", ascending=False).iloc[0]
        comp_rows.append(
            {
                "Method": METHOD_LABELS[method],
                "Precision": f"{int(row['weight_bits'])}-bit",
                "Reuse (d1/d2)": f"{int(row['reuse_dense_1'])}/{int(row['reuse_dense_2'])}",
                "Strategy": row["strategy"],
                "Max error": round(float(row["max_error"]), 4) if pd.notna(row["max_error"]) else None,
                "Latency": None if pd.isna(row["latency_cycles"]) else int(row["latency_cycles"]),
                "DSP": None if pd.isna(row["dsp"]) else int(row["dsp"]),
                "LUT": None if pd.isna(row["lut"]) else int(row["lut"]),
                "Fits?": "Yes" if row.get("fits_board", False) else "No",
                "Reward": round(float(row["reward"]), 1),
            }
        )
    if comp_rows:
        st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)

    # --- Tradeoff scatter ---------------------------------------------------
    st.subheader("Accuracy vs latency tradeoff")
    valid = df[df["compile_success"] == True].copy()  # noqa: E712
    if not valid.empty and valid["latency_cycles"].notna().any():
        chart_df = valid[["latency_cycles", "max_error", "method_label", "dsp"]].dropna(subset=["latency_cycles"])
        st.scatter_chart(
            chart_df,
            x="latency_cycles",
            y="max_error",
            color="method_label",
            size="dsp",
        )
        st.caption("Lower-left is better: low latency *and* low error. Point size = DSP usage.")

    # --- Raw table ----------------------------------------------------------
    st.subheader("All evaluations")
    st.dataframe(df, use_container_width=True, height=320)


if __name__ == "__main__":
    main()
