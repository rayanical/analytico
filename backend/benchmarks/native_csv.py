"""Comparison hooks for the production native and legacy pandas loaders."""
from modules.disk_dataset import DiskDataset

ORIGINAL_LOADER = DiskDataset._ingest_csv_chunks
native_loader = DiskDataset._ingest_csv
