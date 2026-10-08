"""Bounded AI requests that overlap disk CSV preparation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from typing import Any, Callable

import pandas as pd


EARLY_SAMPLE_ROWS = 12
EARLY_SUMMARY_ROWS = 3
EARLY_SUMMARY_COLUMNS = 20


def early_ai_enabled() -> bool:
    """Allow bounded summary and label requests during disk preparation."""
    return os.getenv("ANALYTICO_EARLY_AI", "1").strip() == "1"


def _sample_identity(snapshot) -> str:
    encoded = json.dumps(
        snapshot, ensure_ascii=False, separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sample_value(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)[:160]


def _provisional_label_snapshot(disk, headers, rows, row_count, filename):
    """Build label context from raw rows without claiming full-column facts."""
    columns = []
    normalized_names = list(disk.columns)
    for index, name in enumerate(normalized_names):
        columns.append({
            "column": name,
            "original_name": str(headers[index])[:256],
            "physical_type": "unknown",
            "local_usage": {"role": "unknown", "evidence": "unknown"},
            "samples": [_sample_value(row[index]) for row in rows],
            "statistics": {},
        })
    return {
        "rows": int(row_count),
        "filename": str(filename or "").replace("\\", "/").rsplit("/", 1)[-1][:128],
        "columns": columns,
    }


@dataclass
class EarlyEnrichment:
    """Prestarted requests that can be adopted only for the matching source."""

    disk: Any
    sample_identity: str
    futures: dict[str, Any]
    dataset_id: str | None = None
    cache_version: str | None = None

    def _matches_source(self, dataset) -> bool:
        disk = getattr(dataset, "disk", None)
        if disk is not self.disk:
            return False
        try:
            samples = [
                disk.raw_source_snapshot(limit=EARLY_SAMPLE_ROWS),
                disk.raw_source_snapshot(limit=EARLY_SUMMARY_ROWS, evenly_spaced=False),
            ]
            return _sample_identity(samples) == self.sample_identity
        except Exception:
            return False

    def bind_after_store(self, dataset) -> bool:
        """Bind the request to the registered version after storage succeeds."""
        if not self._matches_source(dataset):
            return False
        self.dataset_id = dataset.id
        self.cache_version = dataset.cache_version
        return True

    def matches(self, dataset) -> bool:
        if (self.dataset_id is not None and dataset.id != self.dataset_id
                or self.cache_version is not None and dataset.cache_version != self.cache_version):
            return False
        return self._matches_source(dataset)

    def cancel(self) -> None:
        for future in self.futures.values():
            future.cancel()


def prepare_early_enrichment(
    disk,
    filename: str,
    *,
    summary_generator: Callable[[str, pd.DataFrame], str | None] | None,
    label_generator: Callable[[dict[str, Any]], dict[str, Any]] | None,
) -> EarlyEnrichment | None:
    """Submit at most one summary and one labels call from bounded raw context."""
    if not early_ai_enabled() or (summary_generator is None and label_generator is None):
        return None
    try:
        snapshot = disk.raw_source_snapshot(limit=EARLY_SAMPLE_ROWS)
        summary_snapshot = disk.raw_source_snapshot(limit=EARLY_SUMMARY_ROWS, evenly_spaced=False)
        headers, rows, row_count = snapshot
        if not headers:
            return None
        requests = {}
        if summary_generator is not None:
            summary_headers, summary_rows, _ = summary_snapshot
            summary_frame = pd.DataFrame(summary_rows, columns=summary_headers)
            summary_frame = summary_frame.iloc[:, :EARLY_SUMMARY_COLUMNS].copy(deep=True)
            requests["summary"] = lambda: summary_generator(filename, summary_frame)
        if label_generator is not None:
            label_snapshot = _provisional_label_snapshot(disk, headers, rows, row_count, filename)
            requests["labels"] = lambda: label_generator(label_snapshot)
        from services.parallel_enrichment import submit_early_requests

        futures = submit_early_requests(requests)
        if not futures:
            return None
        return EarlyEnrichment(disk, _sample_identity([snapshot, summary_snapshot]), futures)
    except Exception:
        # Early enrichment is optional and must not change CSV acceptance.
        return None
