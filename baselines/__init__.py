"""Baseline search policies that the GLM generator is benchmarked against.

These are intentionally *not* the headline method anymore. The headline method is
the GLM generator under test-time finetuning (see :mod:`glm`). The policies here
(random search lives in :mod:`ttt.search`; the surrogate lives in
:mod:`baselines.random_forest_policy`) exist so the dashboard can show that an LLM
that *authors* hardware configs and adapts its weights at test time beats a
classic surrogate-guided hill-climb on an equal evaluation budget.
"""
