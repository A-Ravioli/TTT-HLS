"""Backward-compatibility shim for the random-forest surrogate policy.

The random forest was the original 'BurnTTT' headline. After the realignment it is
a *baseline* the GLM generator is compared against, so its implementation now
lives in :mod:`baselines.random_forest_policy`. This module re-exports it so old
imports (``from ttt.online_policy import OnlineTTTPolicy``) keep working.
"""

from __future__ import annotations

from baselines.random_forest_policy import (
    MIN_SAMPLES_TO_FIT,
    OnlineTTTPolicy,
    RandomForestPolicy,
)

__all__ = ["MIN_SAMPLES_TO_FIT", "OnlineTTTPolicy", "RandomForestPolicy"]
