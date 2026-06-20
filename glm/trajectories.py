"""Append-only trajectory store: (task, prompt, generated artifact, feedback).

Every config the GLM authors -- plus the feedback it earned -- is logged as a JSON
line under ``data/trajectories/``. This is the training data for test-time
finetuning (:mod:`glm.finetune.dataset`) and the audit trail for the dashboard.
Pure stdlib so it works everywhere.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from paths import REPO_ROOT

DATA_DIR = REPO_ROOT / "data"
TRAJ_DIR = DATA_DIR / "trajectories"


class TrajectoryStore:
    """Append (task, config, feedback, reward) rows to a per-run jsonl file."""

    def __init__(self, run_name: str | None = None, path: Path | None = None):
        TRAJ_DIR.mkdir(parents=True, exist_ok=True)
        if path is not None:
            self.path = Path(path)
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.path = TRAJ_DIR / f"{run_name or 'run'}_{stamp}.jsonl"

    def append(
        self,
        task_name: str,
        config: dict[str, Any],
        result: dict[str, Any],
        method: str,
        round_idx: int | None = None,
        prompt: str | None = None,
    ) -> None:
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "task": task_name,
            "method": method,
            "round": round_idx,
            "config": config,
            "prompt": prompt,
            "compile_success": result.get("compile_success"),
            "max_error": result.get("max_error"),
            "latency_cycles": result.get("latency_cycles"),
            "dsp": result.get("dsp"),
            "lut": result.get("lut"),
            "fits_board": result.get("fits_board"),
            "reward": result.get("reward"),
            "error_msg": result.get("error_msg"),
        }
        with self.path.open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")

    @staticmethod
    def read(path: str | Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        p = Path(path)
        if not p.exists():
            return rows
        with p.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    @staticmethod
    def read_all(directory: str | Path | None = None) -> Iterator[dict[str, Any]]:
        d = Path(directory or TRAJ_DIR)
        if not d.exists():
            return
        for jf in sorted(d.glob("*.jsonl")):
            for row in TrajectoryStore.read(jf):
                yield row
