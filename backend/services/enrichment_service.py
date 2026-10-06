"""Bounded background execution for optional dataset enrichment.

The module owns job scheduling and status snapshots. Callers own provider
configuration and any dataset writes, so a completed job cannot silently
change analytics semantics.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import islice
from threading import BoundedSemaphore, Lock
from typing import Any, Callable, Literal, Mapping


EnrichmentStatus = Literal["pending", "running", "done", "error", "disabled"]
EnrichmentWork = Callable[[], Mapping[str, Any]]


@dataclass
class _Job:
    dataset_id: str
    version: str
    status: EnrichmentStatus
    progress: int = 0
    summary: str | None = None
    interpretation_proposals: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    reason: str | None = None
    coverage: dict[str, Any] | None = None
    column_roles: dict[str, str] = field(default_factory=dict)
    column_labels: dict[str, str] = field(default_factory=dict)
    semantic_revision: int = 0
    future: Future[None] | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "status": self.status,
            "progress": self.progress,
            "summary": self.summary,
            "interpretation_proposals": deepcopy(self.interpretation_proposals),
            "error": self.error,
            "reason": self.reason,
            "coverage": deepcopy(self.coverage),
            "column_roles": dict(self.column_roles),
            "column_labels": dict(self.column_labels),
            "semantic_revision": self.semantic_revision,
        }


class EnrichmentManager:
    """Run at most ``max_workers`` jobs with a fixed queue and bounded records."""

    def __init__(
        self,
        max_workers: int = 2,
        max_pending: int = 8,
        max_records: int = 32,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        if max_workers < 1 or max_pending < 0:
            raise ValueError("max_workers must be positive and max_pending cannot be negative")
        if max_records < max_workers + max_pending + 1:
            raise ValueError("max_records must leave room for every queued job and one status")

        self._lock = Lock()
        self._jobs: OrderedDict[str, _Job] = OrderedDict()
        self._max_records = max_records
        self._slots = BoundedSemaphore(max_workers + max_pending)
        self._executor = executor or ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="dataset-enrichment",
        )
        self._owns_executor = executor is None

    def enqueue(self, dataset_id: str, version: str, work: EnrichmentWork) -> dict[str, Any]:
        """Schedule one idempotent job for this dataset version without waiting."""
        with self._lock:
            current = self._jobs.get(dataset_id)
            if current is not None and current.version == version:
                self._jobs.move_to_end(dataset_id)
                return current.snapshot()

            if current is not None and current.future is not None:
                # A queued old version can be removed from the executor. A
                # running callback cannot be stopped, but its result is ignored.
                current.future.cancel()

            if not self._slots.acquire(blocking=False):
                job = _Job(
                    dataset_id=dataset_id,
                    version=version,
                    status="disabled",
                    reason="queue_full",
                )
                self._store_locked(job)
                return job.snapshot()

            job = _Job(dataset_id=dataset_id, version=version, status="pending")
            self._store_locked(job)
            try:
                future = self._executor.submit(self._run, dataset_id, version, work)
            except Exception:
                self._slots.release()
                job.status = "error"
                job.error = "Enrichment could not be started."
                return job.snapshot()

            job.future = future
            future.add_done_callback(lambda _future: self._slots.release())
            return job.snapshot()

    def disable(
        self,
        dataset_id: str,
        version: str,
        reason: str = "unavailable",
    ) -> dict[str, Any]:
        """Record that optional enrichment was intentionally not scheduled."""
        with self._lock:
            current = self._jobs.get(dataset_id)
            if current is not None and current.version == version:
                return current.snapshot()
            if current is not None and current.future is not None:
                current.future.cancel()
            job = _Job(
                dataset_id=dataset_id,
                version=version,
                status="disabled",
                reason=reason,
            )
            self._store_locked(job)
            return job.snapshot()

    def get_status(self, dataset_id: str, version: str | None = None) -> dict[str, Any]:
        """Return a safe snapshot, defaulting to disabled when no job exists."""
        with self._lock:
            job = self._jobs.get(dataset_id)
            if job is None or (version is not None and job.version != version):
                return _disabled_snapshot(dataset_id, version, "not_requested")
            self._jobs.move_to_end(dataset_id)
            return job.snapshot()

    def shutdown(self, wait: bool = True) -> None:
        """Shut down an executor created by this manager (mainly useful in tests)."""
        if self._owns_executor:
            self._executor.shutdown(wait=wait, cancel_futures=True)

    def update(self, dataset_id: str, version: str, result: Mapping[str, Any]) -> None:
        """Publish partial results only to the still-running matching version."""
        with self._lock:
            job = self._matching_job_locked(dataset_id, version)
            if job is not None and job.status == "running":
                self._apply_result(job, result)
                job.progress = max(job.progress, min(99, int(result.get("progress", 10))))

    @staticmethod
    def _apply_result(job: _Job, result: Mapping[str, Any]) -> None:
        summary = result.get("summary")
        job.summary = summary[:4000] if isinstance(summary, str) and summary else None
        proposals = result.get("interpretation_proposals", {})
        if isinstance(proposals, Mapping):
            job.interpretation_proposals = {
                str(name)[:256]: deepcopy(value)
                for name, value in islice(proposals.items(), 256)
                if isinstance(name, str)
            }
        job.column_roles = dict(result.get("column_roles", {}))
        job.column_labels = dict(result.get("column_labels", {}))
        job.semantic_revision = int(result.get("semantic_revision", 0))
        coverage = result.get("coverage")
        job.coverage = deepcopy(coverage) if isinstance(coverage, dict) else None

    def _run(self, dataset_id: str, version: str, work: EnrichmentWork) -> None:
        with self._lock:
            job = self._matching_job_locked(dataset_id, version)
            if job is None:
                return
            job.status = "running"
            job.progress = 10

        try:
            result = work()
            if not isinstance(result, Mapping):
                raise TypeError("Enrichment callback must return a mapping")
            prepared = _Job(dataset_id=dataset_id, version=version, status="running")
            self._apply_result(prepared, result)
        except Exception:
            # Provider exceptions may contain prompts, sampled data, or secrets.
            # Keep the public record generic and do not log exception contents.
            with self._lock:
                job = self._matching_job_locked(dataset_id, version)
                if job is not None:
                    job.status = "error"
                    job.progress = 100
                    job.error = "Enrichment could not be completed."
            return

        with self._lock:
            job = self._matching_job_locked(dataset_id, version)
            if job is not None:
                job.summary = prepared.summary
                job.interpretation_proposals = prepared.interpretation_proposals
                job.coverage = prepared.coverage
                job.column_roles = prepared.column_roles
                job.column_labels = prepared.column_labels
                job.semantic_revision = prepared.semantic_revision
                job.status = "done"
                job.progress = 100
                job.error = None

    def _matching_job_locked(self, dataset_id: str, version: str) -> _Job | None:
        job = self._jobs.get(dataset_id)
        return job if job is not None and job.version == version else None

    def _store_locked(self, job: _Job) -> None:
        self._jobs[job.dataset_id] = job
        self._jobs.move_to_end(job.dataset_id)

        while len(self._jobs) > self._max_records:
            removable = next(
                (
                    key
                    for key, existing in self._jobs.items()
                    if existing.status in {"done", "error", "disabled"}
                    and (existing.future is None or existing.future.done())
                ),
                None,
            )
            if removable is None:
                # Active work is bounded separately by the executor slots.
                break
            del self._jobs[removable]


def _disabled_snapshot(
    dataset_id: str,
    version: str | None,
    reason: str,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "version": version,
        "status": "disabled",
        "progress": 0,
        "summary": None,
        "interpretation_proposals": {},
        "error": None,
        "reason": reason,
        "coverage": None,
    }


# Worker threads start lazily, only after an eligible caller schedules work.
manager = EnrichmentManager()


def enqueue_enrichment(
    dataset_id: str,
    version: str,
    work: EnrichmentWork,
) -> dict[str, Any]:
    return manager.enqueue(dataset_id, version, work)


def get_enrichment_status(
    dataset_id: str,
    version: str | None = None,
) -> dict[str, Any]:
    return manager.get_status(dataset_id, version)


def disable_enrichment(
    dataset_id: str,
    version: str,
    reason: str = "unavailable",
) -> dict[str, Any]:
    return manager.disable(dataset_id, version, reason)
