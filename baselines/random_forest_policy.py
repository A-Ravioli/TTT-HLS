"""Random-forest surrogate policy (the former 'BurnTTT' headline, now a baseline).

This is the original online surrogate from the hackathon MVP: a
:class:`~sklearn.ensemble.RandomForestRegressor` refit online during a search run
that scores a candidate pool of configs and proposes the predicted-best. It does
*not* author hardware configs and it does *not* update an LLM's weights; it picks
points on a fixed, hand-defined knob grid.

It is retained here as the comparison baseline for the GLM generator (see
:mod:`glm`). ``ttt.online_policy`` re-exports these symbols for backward
compatibility.
"""

from __future__ import annotations

import random
from typing import Sequence

from sklearn.ensemble import RandomForestRegressor

from paths import get_logger
from ttt.config_space import BurnConfig, neighbors, sample_random_configs

logger = get_logger("burnttt.baseline.random_forest")

MIN_SAMPLES_TO_FIT = 5


class OnlineTTTPolicy:
    """Surrogate reward model refit online during a search run (baseline)."""

    def __init__(self, n_estimators: int = 128, random_state: int = 0):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self.has_fit = False
        self._rng = random.Random(random_state)
        self.n_fits = 0
        self._configs: list[BurnConfig] = []
        self._rewards: list[float] = []

    def fit(self, configs: Sequence[BurnConfig], rewards: Sequence[float]) -> bool:
        """Fit the surrogate on (config, reward) pairs. Returns whether it fit."""
        self._configs = list(configs)
        self._rewards = list(rewards)
        if len(configs) < MIN_SAMPLES_TO_FIT:
            logger.info("Policy not fit yet: %d/%d samples", len(configs), MIN_SAMPLES_TO_FIT)
            return False
        x = [c.to_vector() for c in configs]
        self.model.fit(x, list(rewards))
        self.has_fit = True
        self.n_fits += 1
        logger.info("Policy refit #%d on %d samples", self.n_fits, len(configs))
        return True

    def _best_known(self, k: int = 5) -> list[BurnConfig]:
        order = sorted(zip(self._configs, self._rewards), key=lambda cr: cr[1], reverse=True)
        return [c for c, _ in order[:k]]

    def predict(self, configs: Sequence[BurnConfig]) -> list[float]:
        if not self.has_fit:
            raise RuntimeError("Policy has not been fit yet")
        return list(self.model.predict([c.to_vector() for c in configs]))

    def propose(
        self,
        n: int = 3,
        n_candidates: int = 200,
        exclude: set[str] | None = None,
    ) -> list[BurnConfig]:
        """Propose ``n`` configs.

        Before the surrogate is fit, this samples randomly (exploration). After,
        it scores a candidate pool with the surrogate and returns the
        predicted-best, skipping already-tried configs. The candidate pool mixes
        random samples (exploration) with single-step neighbors of the
        best-known configs (local exploitation), which makes the search reliably
        refine the best region rather than relying on lucky random draws.
        """
        exclude = exclude or set()
        candidates = list(sample_random_configs(n_candidates, self._rng))

        if self.has_fit and self._configs:
            for best in self._best_known(k=5):
                candidates.extend(neighbors(best))

        candidates = [c for c in self._dedupe(candidates) if c.short_name() not in exclude]
        if not candidates:
            candidates = self._dedupe(sample_random_configs(n_candidates, self._rng))

        if not self.has_fit:
            return candidates[:n]

        preds = self.predict(candidates)
        ranked = [c for c, _ in sorted(zip(candidates, preds), key=lambda cp: cp[1], reverse=True)]

        # Reserve the first slot for a guaranteed greedy refinement of the current
        # best config: its surrogate-best unevaluated neighbor. This makes the
        # search a monotonic hill-climb that does not stall if the surrogate's
        # global ranking is noisy.
        picks: list[BurnConfig] = []
        best_known = self._best_known(k=1)
        if best_known:
            nbrs = [c for c in neighbors(best_known[0]) if c.short_name() not in exclude]
            if nbrs:
                nbr_pred = self.predict(nbrs)
                best_nbr = max(zip(nbrs, nbr_pred), key=lambda cp: cp[1])[0]
                picks.append(best_nbr)

        for c in ranked:
            if len(picks) >= n:
                break
            if c.short_name() not in {p.short_name() for p in picks}:
                picks.append(c)
        return picks[:n]

    @staticmethod
    def _dedupe(configs: list[BurnConfig]) -> list[BurnConfig]:
        seen: set[str] = set()
        out: list[BurnConfig] = []
        for c in configs:
            key = c.short_name()
            if key not in seen:
                seen.add(key)
                out.append(c)
        return out


# Clearer alias for the baseline's role in the realigned project.
RandomForestPolicy = OnlineTTTPolicy
