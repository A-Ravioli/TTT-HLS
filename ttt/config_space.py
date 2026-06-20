"""The BurnTTT hardware-configuration search space.

A :class:`BurnConfig` is the unit the autotuner searches over. It captures the
quantization and HLS knobs that hls4ml exposes and that materially change the
latency / resource / accuracy tradeoff of the generated accelerator.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Iterable

# Search space (see plan.md section 8).
BITWIDTHS = [8, 10, 12, 14, 16]
INT_BITS = [3, 4, 5, 6]
REUSE_FACTORS = [1, 2, 4, 8, 16]
STRATEGIES = ["Latency", "Resource"]

# Order matters: this defines the numeric feature vector fed to the policy.
VECTOR_FIELDS = (
    "weight_bits",
    "activation_bits",
    "int_bits",
    "reuse_dense_1",
    "reuse_dense_2",
    "strategy_latency",
)


@dataclass(frozen=True)
class BurnConfig:
    """A single candidate hardware generation config."""

    weight_bits: int
    activation_bits: int
    int_bits: int
    reuse_dense_1: int
    reuse_dense_2: int
    strategy: str

    def __post_init__(self) -> None:
        # int_bits must fit inside the total word length for ap_fixed<W, I>.
        max_word = max(self.weight_bits, self.activation_bits)
        if self.int_bits >= max_word:
            raise ValueError(
                f"int_bits ({self.int_bits}) must be < total bits ({max_word})"
            )
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {self.strategy}")

    # -- precision strings -------------------------------------------------
    def weight_precision(self) -> str:
        return f"ap_fixed<{self.weight_bits},{self.int_bits}>"

    def activation_precision(self) -> str:
        return f"ap_fixed<{self.activation_bits},{self.int_bits}>"

    # -- serialization -----------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BurnConfig":
        return cls(
            weight_bits=int(d["weight_bits"]),
            activation_bits=int(d["activation_bits"]),
            int_bits=int(d["int_bits"]),
            reuse_dense_1=int(d["reuse_dense_1"]),
            reuse_dense_2=int(d["reuse_dense_2"]),
            strategy=str(d["strategy"]),
        )

    def to_vector(self) -> list[float]:
        """Numeric feature vector consumed by the online policy."""
        return [
            float(self.weight_bits),
            float(self.activation_bits),
            float(self.int_bits),
            float(self.reuse_dense_1),
            float(self.reuse_dense_2),
            1.0 if self.strategy == "Latency" else 0.0,
        ]

    def short_name(self) -> str:
        return (
            f"w{self.weight_bits}a{self.activation_bits}i{self.int_bits}"
            f"_r{self.reuse_dense_1}-{self.reuse_dense_2}_{self.strategy[:3]}"
        )


def config_to_vector(config: BurnConfig) -> list[float]:
    return config.to_vector()


def random_config(rng: random.Random | None = None) -> BurnConfig:
    """Sample one valid random config (retries until int_bits constraint holds)."""
    r = rng or random
    while True:
        bits = r.choice(BITWIDTHS)
        int_bits = r.choice([b for b in INT_BITS if b < bits] or [3])
        try:
            return BurnConfig(
                weight_bits=bits,
                activation_bits=bits,
                int_bits=int_bits,
                reuse_dense_1=r.choice(REUSE_FACTORS),
                reuse_dense_2=r.choice(REUSE_FACTORS),
                strategy=r.choice(STRATEGIES),
            )
        except ValueError:
            continue


def sample_random_configs(n: int, rng: random.Random | None = None) -> list[BurnConfig]:
    """Sample ``n`` distinct-ish random configs."""
    return [random_config(rng) for _ in range(n)]


def seed_configs() -> list[BurnConfig]:
    """Hand-picked seed configs spanning the accuracy/resource spectrum.

    These mirror the manual A/B/C/D configs in plan.md section 5 and give the
    surrogate policy a diverse warm start. Six points (>= the fit threshold) let
    the policy learn the resource/accuracy boundary from round 1.
    """
    return [
        BurnConfig(16, 16, 6, 1, 1, "Latency"),  # high precision, fast, over-budget cliff
        BurnConfig(14, 14, 4, 2, 2, "Latency"),  # over-budget
        BurnConfig(12, 12, 4, 8, 8, "Resource"),  # fits, mid region (good basin start)
        BurnConfig(10, 10, 3, 16, 8, "Resource"),  # fits, lower precision
        BurnConfig(8, 8, 3, 16, 16, "Resource"),  # low precision, light, fits easily
        BurnConfig(14, 14, 5, 4, 4, "Resource"),  # borderline resources
    ]


def _adjacent(grid: list[int], value: int) -> list[int]:
    """Neighboring values one step away in a discrete grid."""
    if value not in grid:
        return grid[:1]
    i = grid.index(value)
    out = []
    if i > 0:
        out.append(grid[i - 1])
    if i < len(grid) - 1:
        out.append(grid[i + 1])
    return out


def neighbors(config: BurnConfig) -> list[BurnConfig]:
    """Single-step mutations of ``config`` for local exploitation.

    Varies one knob at a time (bit level, int bits, each reuse factor, strategy)
    while keeping weight/activation bitwidths tied, as in :func:`random_config`.
    """
    out: list[BurnConfig] = []
    bits = config.weight_bits

    def _try(**overrides) -> None:
        params = config.to_dict()
        params.update(overrides)
        try:
            out.append(BurnConfig.from_dict(params))
        except ValueError:
            pass

    for nb in _adjacent(BITWIDTHS, bits):
        ib = config.int_bits if config.int_bits < nb else nb - 1
        _try(weight_bits=nb, activation_bits=nb, int_bits=ib)
    for nib in _adjacent(INT_BITS, config.int_bits):
        if nib < bits:
            _try(int_bits=nib)
    for nr in _adjacent(REUSE_FACTORS, config.reuse_dense_1):
        _try(reuse_dense_1=nr)
    for nr in _adjacent(REUSE_FACTORS, config.reuse_dense_2):
        _try(reuse_dense_2=nr)
    _try(strategy="Resource" if config.strategy == "Latency" else "Latency")

    # Dedupe, dropping the original.
    seen = {config.short_name()}
    unique = []
    for c in out:
        if c.short_name() not in seen:
            seen.add(c.short_name())
            unique.append(c)
    return unique


def all_field_names() -> Iterable[str]:
    return VECTOR_FIELDS
