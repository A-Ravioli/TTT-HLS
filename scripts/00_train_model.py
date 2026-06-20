#!/usr/bin/env python
"""Train and export the TinyFFNBlock + golden test vectors.

Produces:
    artifacts/tiny_ffn.keras
    artifacts/test_inputs.npy
    artifacts/golden_outputs.npy
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.export_model import export  # noqa: E402
from models.train_toy_model import train  # noqa: E402
from paths import get_logger  # noqa: E402

logger = get_logger("burnttt.script.train")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train + export TinyFFNBlock")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-test", type=int, default=256)
    args = parser.parse_args()

    logger.info("Training TinyFFNBlock (epochs=%d, seed=%d)", args.epochs, args.seed)
    artifacts = train(epochs=args.epochs, seed=args.seed, n_test=args.n_test)
    export(artifacts)
    logger.info("Done. Next: python scripts/01_baseline_compile.py")


if __name__ == "__main__":
    main()
