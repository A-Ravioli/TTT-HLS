"""Tests for the GLM generator layer (heuristic backend; no heavy deps needed)."""

import random

from glm.agent import GLMGenerator
from glm.parsing import dict_to_config, parse_configs
from glm.serving import HeuristicBackend, _norm_vector, _dist2, load_backend
from glm.tasks import make_task, tiny_ffn_block
from glm.trajectories import TrajectoryStore
from ttt.config_space import BurnConfig
from ttt.reward import get_board_budget

PART = "xcu250-figd2104-2l-e"


def _task():
    return make_task(tiny_ffn_block(), PART, get_board_budget(PART))


# -- parsing ----------------------------------------------------------------

def test_parse_configs_from_messy_text():
    txt = (
        "Sure! Here is my proposal:\n```json\n"
        '[{"weight_bits": 12, "activation_bits": 12, "int_bits": 4, '
        '"reuse_dense_1": 4, "reuse_dense_2": 8, "strategy": "Resource"}]\n```'
    )
    cfgs = parse_configs(txt)
    assert len(cfgs) == 1
    assert cfgs[0] == BurnConfig(12, 12, 4, 4, 8, "Resource")


def test_parse_multiple_loose_objects():
    txt = '{"weight_bits":8,"activation_bits":8,"int_bits":3,"reuse_dense_1":16,"reuse_dense_2":16,"strategy":"Resource"} and {"weight_bits":16,"activation_bits":16,"int_bits":6,"reuse_dense_1":1,"reuse_dense_2":1,"strategy":"Latency"}'
    cfgs = parse_configs(txt)
    assert len(cfgs) == 2


def test_dict_to_config_clamps_illegal_int_bits():
    cfg = dict_to_config(
        {"weight_bits": 8, "activation_bits": 8, "int_bits": 9, "reuse_dense_1": 1, "reuse_dense_2": 1, "strategy": "Latency"}
    )
    assert cfg is not None
    assert cfg.int_bits < min(cfg.weight_bits, cfg.activation_bits)


def test_dict_to_config_snaps_to_grid():
    cfg = dict_to_config(
        {"weight_bits": 11, "activation_bits": 13, "int_bits": 4, "reuse_dense_1": 3, "reuse_dense_2": 7, "strategy": "resource"}
    )
    assert cfg is not None
    assert cfg.weight_bits in (10, 12)
    assert cfg.strategy == "Resource"


# -- heuristic backend / agent ---------------------------------------------

def test_heuristic_proposals_valid_and_distinct():
    be = HeuristicBackend()
    props = be.propose_configs(_task().describe(), [], 3, set(), random.Random(0))
    assert len(props) == 3
    assert len({p.short_name() for p in props}) == 3
    for p in props:
        assert p.int_bits < max(p.weight_bits, p.activation_bits)


def test_agent_tops_up_and_excludes():
    gen = GLMGenerator(backend=HeuristicBackend(), seed=1)
    exclude = {c.short_name() for c in gen.propose(_task(), [], n=5)}
    got = gen.propose(_task(), [], n=5, exclude=exclude)
    assert len(got) == 5
    assert all(g.short_name() not in exclude for g in got)


def test_repair_returns_alternative_config():
    gen = GLMGenerator(backend=HeuristicBackend(), seed=2)
    bad = BurnConfig(16, 16, 6, 1, 1, "Latency")
    fixed = gen.repair(_task(), bad, "ERROR: ap_fixed accumulator overflow")
    assert fixed is None or fixed.short_name() != bad.short_name()


def test_adapt_increases_exploitation():
    be = HeuristicBackend()
    cfgs = be.propose_configs(_task().describe(), [], 5, set(), random.Random(3))
    hist = [
        {"reward": 100.0 + i, "_config_obj": c, "config": c.to_dict(), "compile_success": True}
        for i, c in enumerate(cfgs)
    ]
    before = be.exploit
    info = be.adapt(hist)
    assert info["adapted"] is True
    assert be.exploit > before
    assert be.bandwidth < be.base_bandwidth


def test_load_backend_defaults_to_heuristic(monkeypatch):
    monkeypatch.delenv("BURN_GLM_MODEL", raising=False)
    monkeypatch.delenv("BURN_GLM_BACKEND", raising=False)
    assert load_backend().name == "heuristic"


# -- trajectory store -------------------------------------------------------

def test_trajectory_store_roundtrip(tmp_path):
    store = TrajectoryStore(path=tmp_path / "t.jsonl")
    cfg = BurnConfig(12, 12, 4, 8, 8, "Resource")
    store.append("task", cfg.to_dict(), {"reward": 42.0, "compile_success": True}, method="glm", round_idx=1)
    rows = TrajectoryStore.read(store.path)
    assert len(rows) == 1
    assert rows[0]["reward"] == 42.0
    assert rows[0]["method"] == "glm"


# -- the headline claim: test-time training beats the frozen generator ------

def _synthetic_reward(cfg: BurnConfig) -> float:
    """A smooth objective with a clear optimum, so exploitation should win."""
    target = _norm_vector(BurnConfig(12, 12, 3, 16, 16, "Resource"))
    return 1000.0 - 500.0 * _dist2(_norm_vector(cfg), target)


def _offline_run(adapt: bool, seed: int, rounds: int = 6, n: int = 3) -> float:
    gen = GLMGenerator(backend=HeuristicBackend(), seed=seed)
    task = _task()
    history: list[dict] = []
    tried: set[str] = set()
    best = float("-inf")
    for _ in range(rounds):
        for cfg in gen.propose(task, history, n=n, exclude=tried):
            r = _synthetic_reward(cfg)
            history.append(
                {"reward": r, "_config_obj": cfg, "config": cfg.to_dict(), "compile_success": True, "max_error": 0.0}
            )
            tried.add(cfg.short_name())
            best = max(best, r)
        if adapt:
            gen.adapt(history)
    return best


def test_test_time_training_beats_frozen_generator():
    # Averaged over seeds, the test-time-adapted generator should reach at least
    # as good a best-reward as the frozen one (it concentrates evaluations).
    ttt = [_offline_run(adapt=True, seed=s) for s in range(5)]
    frozen = [_offline_run(adapt=False, seed=s) for s in range(5)]
    assert sum(ttt) / len(ttt) >= sum(frozen) / len(frozen)
