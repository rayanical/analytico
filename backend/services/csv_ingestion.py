"""Bounded DuckDB ingestion for CSVs, with an explicit pandas compatibility mode."""

from pathlib import Path
import os
from threading import BoundedSemaphore

from fastapi import HTTPException

from services import ingestion_service
from services.ingestion_service import ingest_dataframe, queue_dataset_enrichment
from services.early_enrichment import early_ai_enabled, prepare_early_enrichment
from storage import DiskDatasetInfo, store_dataset, replace_dataset
from utils.dataframe_utils import read_csv_fast
from utils.pipeline_logging import IngestionMeasurement
from utils.source_files import retain_source, retain_staged_source, validate_staged_source

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


def _select_engine(engine=None):
    selected = engine or os.getenv("ANALYTICO_INGESTION_ENGINE", "auto")
    if selected not in {"auto", "pandas", "disk"}:
        raise HTTPException(500, "Invalid local ingestion engine configuration.")
    return "disk" if selected == "auto" else selected


def preload_ingestion_engine():
    """Pay native-library import cost at startup, before the first upload."""
    if _select_engine() == "disk":
        import duckdb  # noqa: F401
        from modules import disk_dataset  # noqa: F401


def ingest_csv(source, filename: str, endpoint_name: str, *, engine=None, defer_enrichment=True,
               import_settings=None, column_overrides=None, replacement=None,
               enqueue_enrichment=True, ai_column_analysis=None, staged_source=None):
    """Read and store a CSV without unbounded simultaneous ingestion work."""
    max_bytes = _config_integer("ANALYTICO_MAX_UPLOAD_BYTES", 256 * 1024 * 1024, 1)
    if staged_source is not None:
        if not isinstance(source, (str, Path)) or Path(source).absolute() != Path(staged_source.path).absolute():
            raise HTTPException(400, "The staged CSV source does not match the confirmed import.")
        if staged_source.size > max_bytes:
            raise HTTPException(413, "CSV exceeds the configured upload size limit.")
        try:
            size = validate_staged_source(staged_source, max_bytes)
        except (TypeError, ValueError) as error:
            raise HTTPException(400, "The staged CSV source is no longer valid. Please preview the file again.") from error
    else:
        size = _source_size(source)
    if size > max_bytes:
        raise HTTPException(413, "CSV exceeds the configured upload size limit.")
    selected = _select_engine(engine)
    if not _INGESTION_SLOTS.acquire(blocking=False):
        raise HTTPException(429, "Two datasets are already being prepared. Please retry shortly.")
    try:
        with IngestionMeasurement(endpoint_name, includes_csv=True) as measurement:
            if selected == "pandas":
                with measurement.phase("source_copy"):
                    if staged_source is None:
                        owner, retained_path, _ = retain_source(source, max_bytes)
                    else:
                        owner, retained_path, _ = retain_staged_source(staged_source, max_bytes)
                try:
                    with measurement.phase("csv_parse"):
                        frame = (read_csv_fast(retained_path, settings=import_settings)
                                 if import_settings is not None else read_csv_fast(retained_path))
                    if frame.empty:
                        raise HTTPException(400, "CSV is empty.")
                    if frame.shape[1] > 256 or len(frame) > 5_000_000:
                        raise HTTPException(413, "CSV exceeds the supported row or column limit.")
                    return ingest_dataframe(frame, filename, endpoint_name, measurement=measurement,
                                            defer_enrichment=defer_enrichment,
                                            import_settings=import_settings, column_overrides=column_overrides,
                                            source_owner=owner, source_path=retained_path,
                                            replacement=replacement, enqueue_enrichment=enqueue_enrichment,
                                            ai_column_analysis=ai_column_analysis)
                except Exception:
                    owner.cleanup()
                    raise

            from modules.disk_dataset import DiskDataset
            early_enrichment = None

            def on_source_loaded(raw_disk):
                nonlocal early_enrichment
                if not (defer_enrichment and enqueue_enrichment and early_ai_enabled()):
                    return
                summary_generator = (ingestion_service._generate_business_summary
                                     if os.getenv("OPENAI_API_KEY", "").strip() else None)
                provider = ("luna" if ai_column_analysis is True
                            else os.getenv("COLUMN_INTERPRETER", "off").strip().lower())
                label_generator = None
                if (ai_column_analysis is not False and provider == "luna"
                        and os.getenv("OPENAI_API_KEY", "").strip()
                        and os.getenv("COLUMN_ROLE_BACKEND", "decisions").strip().lower() == "decisions"):
                    from modules.schema_interpretation import analyze_schema_labels
                    label_generator = analyze_schema_labels
                early_enrichment = prepare_early_enrichment(
                    raw_disk, filename, summary_generator=summary_generator,
                    label_generator=label_generator,
                )

            with measurement.phase("disk_ingestion"):
                try:
                    disk = DiskDataset.from_csv(source, max_file_bytes=max_bytes,
                                                import_settings=import_settings, column_overrides=column_overrides,
                                                staged_source=staged_source,
                                                on_source_loaded=on_source_loaded)
                except ValueError as error:
                    if early_enrichment is not None:
                        early_enrichment.cancel()
                    from modules.import_policy import CSVStructureError
                    if isinstance(error, CSVStructureError):
                        raise HTTPException(413 if error.limit_exceeded else 400, str(error)) from error
                    limit_error = any(word in str(error) for word in ("limit", "at most", "exceeds"))
                    raise HTTPException(413 if limit_error else 400,
                                        "CSV exceeds the supported ingestion limits." if limit_error
                                        else "CSV is empty or cannot be safely read.") from error
                except Exception:
                    if early_enrichment is not None:
                        early_enrichment.cancel()
                    raise
            measurement.phases.update({f"disk/{name}": seconds for name, seconds in disk.ingestion_timings.items()})
            try:
                measurement.rows, measurement.columns = disk.row_count, len(disk.columns)
                dataset = DiskDatasetInfo(disk, filename)
                dataset.import_settings = disk.import_settings
                dataset.column_overrides = disk.column_overrides
                dataset.column_schema = disk.column_schema
                if replacement:
                    dataset.id = replacement[0]
                with measurement.phase("response_and_storage"):
                    response = disk.to_upload_response(dataset.id, filename)
                    if not defer_enrichment:
                        response.summary = ingestion_service._generate_business_summary(filename, dataset.sample_frame(3))
                    if replacement:
                        replace_dataset(replacement[0], dataset, replacement[1])
                    else:
                        store_dataset(dataset)
                    response.version = dataset.cache_version
                    response.column_schema = dataset.column_schema
                if defer_enrichment and enqueue_enrichment:
                    if early_enrichment is None:
                        response.enrichment_status = queue_dataset_enrichment(
                            dataset, ai_column_analysis=ai_column_analysis,
                        )
                    else:
                        response.enrichment_status = queue_dataset_enrichment(
                            dataset, ai_column_analysis=ai_column_analysis,
                            early_enrichment=early_enrichment,
                        )
                    early_enrichment = None
                return response
            except Exception:
                if early_enrichment is not None:
                    early_enrichment.cancel()
                disk.close()
                raise
    finally:
        _INGESTION_SLOTS.release()
