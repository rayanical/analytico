"""
Analytico Backend - Dataset Storage
In-memory dataset management with TTL expiration
"""

import uuid
from datetime import datetime, timedelta
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
        summary: Optional[str] = None
    ):
        self.id = str(uuid.uuid4())
        self.df = df
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
    
    def touch(self):
        """Update last accessed time"""
        self.last_accessed = datetime.now()
    
    def is_expired(self, ttl_hours: int = 1) -> bool:
        """Check if dataset has expired"""
        return datetime.now() - self.last_accessed > timedelta(hours=ttl_hours)


# Global dataset storage
DATASETS: dict[str, DatasetInfo] = {}
MAX_DATASETS = 10


def cleanup_expired():
    """Remove expired datasets from memory"""
    expired = [k for k, v in DATASETS.items() if v.is_expired()]
    for k in expired:
        del DATASETS[k]


def get_dataset(dataset_id: str) -> DatasetInfo:
    """Retrieve dataset by ID, with expiration check"""
    cleanup_expired()
    if dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found or expired. Please re-upload.")
    ds = DATASETS[dataset_id]
    ds.touch()
    return ds


def store_dataset(ds_info: DatasetInfo) -> str:
    """Store dataset and return its ID"""
    cleanup_expired()
    
    # Evict oldest if at capacity
    if len(DATASETS) >= MAX_DATASETS:
        oldest_id = min(DATASETS.keys(), key=lambda k: DATASETS[k].last_accessed)
        del DATASETS[oldest_id]
    
    DATASETS[ds_info.id] = ds_info
    return ds_info.id
