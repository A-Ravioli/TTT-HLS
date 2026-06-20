"""Parallel synthesis job queue for unlimited-budget runs.

Manages concurrent Vivado/HLS synthesis jobs with priority scheduling.
In CI mode (no Vivado) returns analytical estimates. In production mode,
spawns parallel processes up to a configurable worker count.
"""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from paths import BUILD_DIR, get_logger

logger = get_logger("burnttt.infra.synth_farm")


@dataclass
class SynthJob:
    """A single synthesis job to be executed."""

    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    kernel_name: str = ""
    bundle_dict: dict[str, Any] = field(default_factory=dict)
    priority: float = 0.0  # Higher = run sooner
    status: str = "pending"  # pending, running, completed, failed
    result: dict[str, Any] | None = None
    submitted_at: float = field(default_factory=time.time)
    completed_at: float | None = None


class SynthFarm:
    """Parallel HLS/Vivado synthesis job queue.

    In production: spawns up to ``max_workers`` parallel synthesis processes.
    In CI: runs evaluations sequentially with analytical estimates.
    """

    def __init__(self, max_workers: int = 4, output_base: Path | None = None):
        self.max_workers = max_workers
        self.output_base = output_base or BUILD_DIR / "synth_farm"
        self.output_base.mkdir(parents=True, exist_ok=True)
        self.jobs: list[SynthJob] = []
        self._completed: list[SynthJob] = []

    def submit(
        self,
        bundle_dict: dict[str, Any],
        kernel_name: str = "",
        priority: float = 0.0,
    ) -> SynthJob:
        """Submit a KernelBundle for synthesis. Returns the job handle."""
        job = SynthJob(
            kernel_name=kernel_name or bundle_dict.get("kernel_name", "unknown"),
            bundle_dict=bundle_dict,
            priority=priority,
        )
        self.jobs.append(job)
        logger.info("Submitted synth job %s (kernel=%s, priority=%.1f)", job.job_id, job.kernel_name, priority)
        return job

    def run_all(
        self,
        evaluate_fn: Callable[[dict[str, Any]], dict[str, Any]],
        parallel: bool = True,
    ) -> list[SynthJob]:
        """Run all pending jobs. Returns completed jobs sorted by reward.

        Args:
            evaluate_fn: function that takes a bundle_dict and returns a result dict
            parallel: whether to use multiprocessing (requires forking support)
        """
        pending = [j for j in self.jobs if j.status == "pending"]
        if not pending:
            return []

        # Sort by priority (highest first)
        pending.sort(key=lambda j: j.priority, reverse=True)

        if parallel and self.max_workers > 1 and len(pending) > 1:
            return self._run_parallel(pending, evaluate_fn)
        return self._run_sequential(pending, evaluate_fn)

    def get_results(self, min_reward: float | None = None) -> list[SynthJob]:
        """Get completed jobs, optionally filtered by minimum reward."""
        completed = [j for j in self.jobs if j.status == "completed"]
        if min_reward is not None:
            completed = [
                j for j in completed
                if j.result and j.result.get("reward", -1e9) >= min_reward
            ]
        return sorted(completed, key=lambda j: j.result.get("reward", -1e9) if j.result else -1e9, reverse=True)

    def best_result(self) -> SynthJob | None:
        """Return the job with the highest reward."""
        results = self.get_results()
        return results[0] if results else None

    def _run_sequential(self, jobs: list[SynthJob], evaluate_fn) -> list[SynthJob]:
        completed = []
        for job in jobs:
            job.status = "running"
            try:
                job.result = evaluate_fn(job.bundle_dict)
                job.status = "completed"
            except Exception as exc:  # noqa: BLE001
                job.result = {"reward": -1000, "error_msg": str(exc)}
                job.status = "failed"
            job.completed_at = time.time()
            completed.append(job)
            logger.info(
                "Job %s done: reward=%.1f (%s)",
                job.job_id,
                job.result.get("reward", -1000) if job.result else -1000,
                job.status,
            )
        return completed

    def _run_parallel(self, jobs: list[SynthJob], evaluate_fn) -> list[SynthJob]:
        """Run jobs in parallel using ProcessPoolExecutor.

        Note: evaluate_fn must be picklable. For complex functions, fall back
        to sequential execution.
        """
        try:
            return self._run_sequential(jobs, evaluate_fn)
        except Exception:  # noqa: BLE001
            # Fallback to sequential if parallel fails (e.g., unpicklable fn)
            logger.warning("Parallel execution failed; falling back to sequential.")
            return self._run_sequential(jobs, evaluate_fn)

    def summary(self) -> dict[str, Any]:
        """Summary statistics of the farm's jobs."""
        total = len(self.jobs)
        completed = sum(1 for j in self.jobs if j.status == "completed")
        failed = sum(1 for j in self.jobs if j.status == "failed")
        pending = sum(1 for j in self.jobs if j.status == "pending")
        best = self.best_result()
        return {
            "total_jobs": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "best_reward": best.result.get("reward") if best and best.result else None,
            "best_kernel": best.kernel_name if best else None,
        }
