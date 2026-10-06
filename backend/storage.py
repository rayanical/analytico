"""
Analytico Backend - Dataset Storage
In-memory dataset management with TTL expiration
"""

import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from threading import RLock
from typing import Optional

import pandas as pd
from fastapi import HTTPException


class DatasetInfo:
    """Container for uploaded dataset with metadata"""
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        filename: str, 
        cleaning_actions: list[str],
        missing_counts: dict[str, int], 
        column_types: dict[str, str],
        column_formats: dict[str, str], 
        profile: dict, 
        default_chart: Optional[dict],
        suggestions: list[str], 
        summary: Optional[str] = None,
        raw_df: Optional[pd.DataFrame] = None,
    ):
        self.id = str(uuid.uuid4())
        # A newly ingested/replaced dataset always receives a new cache version,
        # even if a caller later reuses its public ID.
        self.cache_version = str(uuid.uuid4())
        self.df = df
        # Keep the source separately from the parsed analytics view. Ingestion
        # supplies an already detached frame; callers without one get a copy.
        self.raw_df = raw_df if raw_df is not None else df.copy(deep=True)
        self.raw_missing_counts = {
            col: int(count)
            for col, count in self.raw_df.isna().sum().items()
            if count > 0
        }
        self.filename = filename
        self.cleaning_actions = cleaning_actions
        self.missing_counts = missing_counts
        self.column_types = column_types
        self.column_formats = column_formats
        self.profile = profile
        self.default_chart = default_chart
        self.suggestions = suggestions
        self.summary = summary
        self.created_at = datetime.now()
        self.touch()

    def invalidate_cached_results(self) -> None:
        """Advance the data version after any in-place data or schema edit.

        Internal callers that mutate ``df``, column metadata, or other inputs
        used to build analytics must call this method before serving requests.
        """
        self.cache_version = str(uuid.uuid4())

    @property
    def row_count(self) -> int:
        return len(self.df)

    @property
    def column_names(self) -> list[str]:
        return self.df.columns.tolist()

    def sample_frame(self, limit: int = 5) -> pd.DataFrame:
        return self.df.head(limit)

    def close(self) -> None:
        """Release a retained original CSV owned by this dataset, if present."""
        owner = getattr(self, "_source_owner", None)
        if owner is not None:
            owner.cleanup()
            self._source_owner = None

    def touch(self):
        self.last_accessed = datetime.now()

    def is_expired(self, ttl_hours: int = 1) -> bool:
        return datetime.now() - self.last_accessed > timedelta(hours=ttl_hours)


class DiskDatasetInfo:
    """A dataset handle whose source and parsed view remain on disk.

    Deliberately has no full-frame ``df`` property: callers must use bounded
    samples or the disk adapter's closed aggregation/drilldown interface.
    """

    def __init__(self, disk, filename: str):
        self.id = str(uuid.uuid4())
        self.cache_version = str(uuid.uuid4())
        self.disk = disk
        self.disk.cache_version = self.cache_version
        self.filename = filename
        self.column_types = disk.column_types
        self.column_formats = disk.column_formats
        self.missing_counts = disk.missing_counts
        self.raw_missing_counts = disk.raw_missing_counts
        self.created_at = datetime.now()
        self.touch()

    @property
    def row_count(self) -> int:
        return self.disk.row_count

    @property
    def column_names(self) -> list[str]:
        return list(self.disk.columns)

    def sample_frame(self, limit: int = 5) -> pd.DataFrame:
        return self.disk.sample_frame(limit)

    def close(self) -> None:
        self.disk.close()

    def touch(self):
        self.last_accessed = datetime.now()

    def is_expired(self, ttl_hours: int = 1) -> bool:
        return datetime.now() - self.last_accessed > timedelta(hours=ttl_hours)

    def invalidate_cached_results(self) -> None:
        self.cache_version = str(uuid.uuid4())


# Global dataset storage
DATASETS: dict[str, DatasetInfo | DiskDatasetInfo] = {}
MAX_DATASETS = 10
_DATASETS_LOCK = RLock()


def _retire_dataset(dataset):
    """Remove ownership now; close resources after the last active reader."""
    dataset._pending_close = True
    if not getattr(dataset, "_active_readers", 0):
        dataset._pending_close = False
        dataset.close()


@contextmanager
def lease_dataset(dataset_id: str, *, loader=None):
    """Keep a dataset usable during a read even if its registry entry is evicted."""
    with _DATASETS_LOCK:
        dataset = (loader or get_dataset)(dataset_id)
        dataset._active_readers = getattr(dataset, "_active_readers", 0) + 1
    try:
        yield dataset
    finally:
        with _DATASETS_LOCK:
            dataset._active_readers -= 1
            if not dataset._active_readers and getattr(dataset, "_pending_close", False):
                dataset._pending_close = False
                dataset.close()


def cleanup_expired():
    """Remove expired datasets from memory"""
    with _DATASETS_LOCK:
        expired = [k for k, v in DATASETS.items() if v.is_expired()]
        for k in expired:
            _retire_dataset(DATASETS[k])
            del DATASETS[k]


def get_dataset(dataset_id: str) -> DatasetInfo | DiskDatasetInfo:
    """Retrieve dataset by ID, with expiration check"""
    with _DATASETS_LOCK:
        cleanup_expired()
        if dataset_id not in DATASETS:
            raise HTTPException(status_code=404, detail="Dataset not found or expired. Please re-upload.")
        ds = DATASETS[dataset_id]
        ds.touch()
        return ds


def store_dataset(ds_info: DatasetInfo | DiskDatasetInfo) -> str:
    """Store dataset and return its ID"""
    with _DATASETS_LOCK:
        cleanup_expired()

        previous = DATASETS.get(ds_info.id)
        if previous is not None and previous is not ds_info:
            _retire_dataset(previous)
        # Replacing an existing handle must not evict an unrelated dataset.
        if previous is None and len(DATASETS) >= MAX_DATASETS:
            oldest_id = min(DATASETS.keys(), key=lambda k: DATASETS[k].last_accessed)
            _retire_dataset(DATASETS[oldest_id])
            del DATASETS[oldest_id]

        DATASETS[ds_info.id] = ds_info
        return ds_info.id


def replace_dataset(dataset_id: str, staged_dataset, expected_version: str):
    """Atomically publish a validated replacement, preserving the public ID."""
    with _DATASETS_LOCK:
        current = get_dataset(dataset_id)
        if current.cache_version != expected_version:
            raise HTTPException(409, "The dataset changed. Refresh the column review before applying edits.")
        staged_dataset.id = dataset_id
        DATASETS[dataset_id] = staged_dataset
        _retire_dataset(current)
        return staged_dataset
