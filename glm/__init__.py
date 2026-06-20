"""GLM: the LLM that authors model-to-FPGA compiler artifacts and is
test-time-finetuned on per-task synthesis/simulation feedback.

This package is the heart of the realigned project. Instead of a random-forest
surrogate picking points on a fixed knob grid, an LLM (GLM) *authors* the
hardware-generation artifact (an hls4ml config, and -- behind a flag -- raw HLS),
gets structured feedback from the feedback engine (:mod:`ttt.evaluate_config`),
and has its weights adapted (LoRA) to the specific ``(model-block, FPGA-part)``
task at test time.

Everything here degrades gracefully: if no GPU / GLM weights / ``transformers``
are available, a deterministic heuristic backend stands in for the LLM so the
whole loop still runs and is testable. See :mod:`glm.serving`.
"""
