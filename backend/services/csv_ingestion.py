"""Choose bounded disk ingestion for large CSVs, retaining the small-file path."""

from pathlib import Path
import os
from threading import BoundedSemaphore

from fastapi import HTTPException

from services.ingestion_service import ingest_dataframe, queue_dataset_enrichment, _generate_business_summary
from storage import DiskDatasetInfo, store_dataset
from utils.dataframe_utils import read_csv_fast
from utils.pipeline_logging import IngestionMeasurement

_INGESTION_SLOTS = BoundedSemaphore(2)


def _config_integer(name, default, minimum=0):
    try:
        value = int(os.getenv(name, str(default)))
        if value < minimum:
            raise ValueError()
        return value
    except ValueError as error:
        raise HTTPException(500, "Invalid local ingestion configuration.") from error


def _source_size(source):
    if isinstance(source, (str, Path)):
        return Path(source).stat().st_size
    try:
        position = source.tell()
        source.seek(0, 2)
        size = source.tell()
        source.seek(position)
        return size
    except (AttributeError, OSError):
        raise HTTPException(400, "CSV input must support bounded file access.")


def ingest_csv(source, filename: str, endpoint_name: str, *, engine=None, defer_enrichment=True):
    """Read and store a CSV without unbounded simultaneous ingestion work."""
    size = _source_size(source)
    max_bytes = _config_integer("ANALYTICO_MAX_UPLOAD_BYTES", 256 * 1024 * 1024, 1)
    if size > max_bytes:
        raise HTTPException(413, "CSV exceeds the configured upload size limit.")
    selected = engine or os.getenv("ANALYTICO_INGESTION_ENGINE", "auto")
    if selected not in {"auto", "pandas", "disk"}:
        raise HTTPException(500, "Invalid local ingestion engine configuration.")
    if selected == "auto":
        selected = "disk" if size >= _config_integer("ANALYTICO_DISK_THRESHOLD_BYTES", 8 * 1024 * 1024) else "pandas"
    if not _INGESTION_SLOTS.acquire(blocking=False):
        raise HTTPException(429, "Two datasets are already being prepared. Please retry shortly.")
    try:
        with IngestionMeasurement(endpoint_name, includes_csv=True) as measurement:
            if selected == "pandas":
                with measurement.phase("csv_parse"):
                    frame = read_csv_fast(source)
                if frame.empty:
                    raise HTTPException(400, "CSV is empty.")
                if frame.shape[1] > 256 or len(frame) > 5_000_000:
                    raise HTTPException(413, "CSV exceeds the supported row or column limit.")
                return ingest_dataframe(frame, filename, endpoint_name, measurement=measurement,
                                        defer_enrichment=defer_enrichment)

            from modules.disk_dataset import DiskDataset
            with measurement.phase("disk_ingestion"):
                try:
                    disk = DiskDataset.from_csv(source, max_file_bytes=max_bytes)
                except ValueError as error:
                    limit_error = any(word in str(error) for word in ("limit", "at most", "exceeds"))
                    raise HTTPException(413 if limit_error else 400,
                                        "CSV exceeds the supported ingestion limits." if limit_error
                                        else "CSV is empty or cannot be safely read.") from error
            try:
                measurement.rows, measurement.columns = disk.row_count, len(disk.columns)
                dataset = DiskDatasetInfo(disk, filename)
                with measurement.phase("response_and_storage"):
                    response = disk.to_upload_response(dataset.id, filename)
                    if not defer_enrichment:
                        response.summary = _generate_business_summary(filename, dataset.sample_frame(3))
                    store_dataset(dataset)
                if defer_enrichment:
                    response.enrichment_status = queue_dataset_enrichment(dataset)
                return response
            except Exception:
                disk.close()
                raise
    finally:
        _INGESTION_SLOTS.release()
