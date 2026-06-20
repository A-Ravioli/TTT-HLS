"""Trace store: persist full HLS + logs for every trajectory (DPO mining).

Stores complete kernel sources, synthesis logs, cosim results, and metadata
for each attempt so that preference pairs can be mined after the fact.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from paths import REPO_ROOT, get_logger

logger = get_logger("burnttt.infra.trace_store")

DEFAULT_STORE_DIR = REPO_ROOT / "data" / "trajectories" / "hls"


@dataclass
class HLSTrace:
    """A single HLS trajectory entry."""

    trace_id: str
    kernel_name: str
    task_name: str
    round_idx: int
    method: str  # "glm_hls", "glm_hls_ttt", etc.
    sources: dict[str, str]  # filename -> content
    result: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "kernel_name": self.kernel_name,
            "task_name": self.task_name,
            "round_idx": self.round_idx,
            "method": self.method,
            "sources": self.sources,
            "result": self.result,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class HLSTraceStore:
    """Append-only JSONL store for HLS trajectories.

    Each run gets its own file under ``data/trajectories/hls/{run_name}.jsonl``.
    Traces include full source code so DPO pairs can be mined post-hoc.
    """

    def __init__(self, run_name: str = "default", store_dir: Path | None = None):
        self.run_name = run_name
        self.store_dir = store_dir or DEFAULT_STORE_DIR
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.store_dir / f"{run_name}.jsonl"
        self._count = 0

    def append(
        self,
        task_name: str,
        kernel_name: str,
        sources: dict[str, str],
        result: dict[str, Any],
        round_idx: int = 0,
        method: str = "glm_hls",
        metadata: dict[str, Any] | None = None,
    ) -> HLSTrace:
        """Append a trace entry to the store."""
        self._count += 1
        trace = HLSTrace(
            trace_id=f"{self.run_name}_{self._count:04d}",
            kernel_name=kernel_name,
            task_name=task_name,
            round_idx=round_idx,
            method=method,
            sources=sources,
            result=_sanitize_result(result),
            metadata=metadata or {},
        )

        with open(self._path, "a") as f:
            f.write(json.dumps(trace.to_dict()) + "\n")

        logger.debug("Stored trace %s (reward=%.1f)", trace.trace_id, result.get("reward", -1e9))
        return trace

    def load_all(self) -> list[HLSTrace]:
        """Load all traces from the store file."""
        if not self._path.exists():
            return []
        traces = []
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    traces.append(HLSTrace(**d))
        return traces

    def load_top_k(self, k: int = 10) -> list[HLSTrace]:
        """Load the top-k traces by reward."""
        all_traces = self.load_all()
        return sorted(
            all_traces,
            key=lambda t: t.result.get("reward", -1e9),
            reverse=True,
        )[:k]

    def load_for_dpo(self, min_reward_gap: float = 10.0) -> list[tuple[HLSTrace, HLSTrace]]:
        """Load preference pairs for DPO: (chosen, rejected) where reward gap > threshold."""
        traces = sorted(
            [t for t in self.load_all() if t.result.get("cosim_pass")],
            key=lambda t: t.result.get("reward", -1e9),
            reverse=True,
        )
        pairs = []
        n = len(traces)
        for i in range(n):
            for j in range(n - 1, i, -1):
                r_i = traces[i].result.get("reward", -1e9)
                r_j = traces[j].result.get("reward", -1e9)
                if r_i - r_j >= min_reward_gap:
                    pairs.append((traces[i], traces[j]))
                    if len(pairs) >= 64:
                        return pairs
        return pairs

    @property
    def count(self) -> int:
        """Number of traces appended this session."""
        return self._count


def _sanitize_result(result: dict[str, Any]) -> dict[str, Any]:
    """Remove non-serializable entries from a result dict."""
    clean = {}
    for k, v in result.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            clean[k] = v
        elif isinstance(v, dict):
            clean[k] = _sanitize_result(v)
        elif isinstance(v, (list, tuple)):
            clean[k] = [x for x in v if isinstance(x, (str, int, float, bool, type(None)))]
        else:
            clean[k] = str(v)
    return clean
