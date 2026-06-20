import random

import pytest

from ttt.config_space import (
    BurnConfig,
    config_to_vector,
    neighbors,
    random_config,
    sample_random_configs,
    seed_configs,
)


def test_roundtrip_serialization():
    cfg = BurnConfig(12, 12, 4, 2, 8, "Latency")
    assert BurnConfig.from_dict(cfg.to_dict()) == cfg


def test_precision_strings():
    cfg = BurnConfig(10, 8, 4, 1, 1, "Resource")
    assert cfg.weight_precision() == "ap_fixed<10,4>"
    assert cfg.activation_precision() == "ap_fixed<8,4>"


def test_vector_shape_and_strategy_encoding():
    lat = BurnConfig(16, 16, 6, 1, 1, "Latency")
    res = BurnConfig(16, 16, 6, 1, 1, "Resource")
    assert len(config_to_vector(lat)) == 6
    assert lat.to_vector()[-1] == 1.0
    assert res.to_vector()[-1] == 0.0


def test_int_bits_must_fit_word():
    with pytest.raises(ValueError):
        BurnConfig(8, 8, 8, 1, 1, "Latency")
    with pytest.raises(ValueError):
        BurnConfig(8, 8, 9, 1, 1, "Latency")


def test_invalid_strategy_rejected():
    with pytest.raises(ValueError):
        BurnConfig(8, 8, 3, 1, 1, "Nonsense")


def test_random_configs_are_valid():
    rng = random.Random(0)
    for cfg in sample_random_configs(50, rng):
        assert cfg.int_bits < max(cfg.weight_bits, cfg.activation_bits)
        assert cfg.strategy in ("Latency", "Resource")


def test_seed_configs_cover_spectrum():
    configs = seed_configs()
    assert len(configs) >= 4
    bits = {c.weight_bits for c in configs}
    assert 8 in bits and 16 in bits


def test_short_name_stable():
    cfg = BurnConfig(12, 12, 4, 2, 8, "Latency")
    assert "w12a12i4" in cfg.short_name()
    assert "r2-8" in cfg.short_name()
    assert cfg.short_name() == BurnConfig(12, 12, 4, 2, 8, "Latency").short_name()


def test_random_config_is_deterministic_with_seed():
    a = random_config(random.Random(42))
    b = random_config(random.Random(42))
    assert a == b


def test_neighbors_are_valid_and_distinct():
    cfg = BurnConfig(12, 12, 4, 4, 4, "Resource")
    nbrs = neighbors(cfg)
    assert nbrs, "expected at least one neighbor"
    names = {n.short_name() for n in nbrs}
    assert cfg.short_name() not in names  # original excluded
    assert len(names) == len(nbrs)  # all distinct
    for n in nbrs:
        assert n.int_bits < max(n.weight_bits, n.activation_bits)
        assert n.strategy in ("Latency", "Resource")


def test_neighbors_include_strategy_flip():
    cfg = BurnConfig(12, 12, 4, 4, 4, "Latency")
    strategies = {n.strategy for n in neighbors(cfg)}
    assert "Resource" in strategies
