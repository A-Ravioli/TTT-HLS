"""Seed HLS kernel templates that GLM edits rather than generating from scratch.

These templates are verified starting points — correct, synthesizable, and
passing cosim — that the GLM compiler agent iteratively modifies to optimize
for a specific (model, FPGA) task.
"""
