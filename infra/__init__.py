"""Infrastructure: staged evaluation and finetuning launch helpers.

The GLM generates many candidate configs, but HLS synthesis and bitstream builds
are expensive. :mod:`infra.staged_eval` enforces the staged funnel from
``plan.md`` (cheap sim for many -> synth for a few -> bitstream for one), and
:mod:`infra.launch` handles GPU placement for test-time finetuning.
"""
