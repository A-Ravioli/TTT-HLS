"""Tests for the reward-variant sweep orchestration and registry."""

from __future__ import annotations

import os

from infra import reward_sweep as sweep
from ttt.reward_variants import (
    DEFAULT_VARIANT,
    active_weights,
    get_variant,
    list_variants,
)


def test_variant_registry_has_expected_members():
    names = list_variants()
    for expected in ("v2_balanced", "throughput_first", "efficiency_first", "accuracy_strict", "legacy"):
        assert expected in names


def test_active_weights_follows_env(monkeypatch):
    monkeypatch.setenv("BURN_REWARD_VARIANT", "throughput_first")
    assert active_weights().name == "throughput_first"
    monkeypatch.delenv("BURN_REWARD_VARIANT", raising=False)
    assert active_weights().name == DEFAULT_VARIANT


def test_get_variant_falls_back_to_default():
    assert get_variant("does-not-exist").name == DEFAULT_VARIANT


def test_sweep_variants_excludes_legacy_by_default():
    assert "legacy" not in sweep.sweep_variants()
    assert "legacy" in sweep.sweep_variants(include_legacy=True)


def test_reward_variant_env_restores(monkeypatch):
    monkeypatch.setenv("BURN_REWARD_VARIANT", "v2_balanced")
    with sweep.reward_variant_env("efficiency_first"):
        assert os.environ["BURN_REWARD_VARIANT"] == "efficiency_first"
    assert os.environ["BURN_REWARD_VARIANT"] == "v2_balanced"


def _row(reward, max_error=0.01, latency=200, dsp=20, compile_success=True, part="xcu250-figd2104-2l-e"):
    return {
        "reward": reward,
        "compile_success": compile_success,
        "max_error": max_error,
        "latency_cycles": latency,
        "dsp": dsp,
        "lut": 5000,
        "ff": 5000,
        "bram": 4,
        "target_part": part,
        "config_name": f"cfg_r{reward}",
    }


def test_summarize_run_picks_best_and_canonical():
    rows = [_row(0.2, latency=2000), _row(0.8, latency=100), _row(-1.0, compile_success=False)]
    res = sweep.summarize_run(0, "v2_balanced", rows, wall_seconds=1.0)
    assert res.n_evals == 3
    assert res.best_reward == 0.8
    assert res.best_canonical > 0  # the fast compiled design scores well canonically
    assert abs(res.compile_rate - 2 / 3) < 1e-9


def test_run_variant_sweep_sets_env_and_ranks(tmp_path):
    seen = {}

    def run_fn(variant):
        seen[variant] = os.environ.get("BURN_REWARD_VARIANT")
        # throughput_first "discovers" a faster design in this fake.
        latency = 80 if variant == "throughput_first" else 1500
        return [_row(0.5, latency=latency)]

    variants = ["v2_balanced", "throughput_first", "efficiency_first"]
    results = sweep.run_variant_sweep(run_fn, variants, iteration=0, csv_path=tmp_path / "sweep.csv")
    assert {r.variant for r in results} == set(variants)
    # The env was set to each variant during its run.
    assert seen == {v: v for v in variants}
    ranked = sweep.rank_variants(results)
    assert ranked[0].variant == "throughput_first"
    assert (tmp_path / "sweep.csv").exists()
    assert sweep.top_variants(results, 2)[0] == "throughput_first"
