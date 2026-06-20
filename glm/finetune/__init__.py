"""Test-time finetuning of the GLM generator.

This is the realigned, honest "TTT": the *generator's weights* (LoRA adapters)
are updated during a run, on feedback from the specific (block, part) task, so it
authors better hardware the longer it works on that task. Off-GPU, the
:class:`~glm.serving.HeuristicBackend` is adapted instead so the loop still
demonstrates the same climb.
"""
