"""Read and atomically apply deterministic dataset schema decisions."""

from __future__ import annotations

import math
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from fastapi import HTTPException
from pydantic import ValidationError

from modules.import_policy import ColumnOverride, ImportSettings
from services.csv_ingestion import ingest_csv
from services.enrichment_service import get_enrichment_status
from storage import lease_dataset


def _plain_value(value: Any) -> Any:
    """Convert bounded preview values to JSON-safe primitives."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        stamp = pd.Timestamp(value)
        return None if pd.isna(stamp) else stamp.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if type(value) is int and abs(value) > 2**53 - 1:
        return str(value)
    if isinstance(value, str) and len(value) > 160:
        return value[:159] + "…"
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _model_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _columns_for(dataset: Any) -> list[dict[str, Any]]:
    names = list(dataset.column_names)
    saved = {
        str(item.get("column")): dict(item)
        for item in (getattr(dataset, "column_schema", None) or [])
        if isinstance(item, Mapping) and item.get("column") is not None
    }
    disk = getattr(dataset, "disk", None)
    headers = list(getattr(disk, "original_headers", []) or getattr(dataset, "original_headers", []))
    summaries = {
        summary.name: summary
        for summary in getattr(disk, "column_summaries", [])
    }
    semantic_types = getattr(dataset, "column_types", {}) or {}
    formats = getattr(dataset, "column_formats", {}) or {}
    result = []
    for index, name in enumerate(names):
        item = saved.get(name, {})
        summary = summaries.get(name)
        inferred_parse = "date" if summary and summary.is_datetime else (
            "number" if summary and summary.is_numeric else "text"
        )
        provenance = item.get("provenance", "inference")
        record = {
            "column": name,
            "original_name": item.get("original_name") or (headers[index] if index < len(headers) else name),
            "parse_as": item.get("parse_as", inferred_parse),
            "role": item.get("role") or semantic_types.get(name) or getattr(summary, "semantic_type", None),
            "format": item.get("format") or formats.get(name) or getattr(summary, "format", None),
            "display_name": item.get("display_name"),
            "unit": item.get("unit"),
            "aggregation": item.get("aggregation"),
            "provenance": provenance,
            "status": item.get("status", "confirmed" if provenance != "inference" else "suggested"),
        }
        if not item:
            record["provenance"] = "inference"
            record["status"] = "suggested"
        result.append(record)
    return result


def get_schema(dataset_id: str) -> dict[str, Any]:
    """Return current schema decisions, bounded preview, and review proposals."""
    with lease_dataset(dataset_id) as dataset:
        frame = dataset.sample_frame(10)
        columns = _columns_for(dataset)
        status = get_enrichment_status(dataset.id, dataset.cache_version)
        return {
            "dataset_id": dataset.id,
            "version": dataset.cache_version,
            "settings": ImportSettings.model_validate(
                _model_dict(getattr(dataset, "import_settings", None))
            ).model_dump(mode="json"),
            "columns": columns,
            "preview": {
                "columns": list(frame.columns),
                "rows": [
                    {str(column): _plain_value(value) for column, value in row.items()}
                    for row in frame.to_dict(orient="records")
                ],
            },
            "proposals": status.get("interpretation_proposals", {}),
        }


def _override_payload(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="python", exclude_unset=True)
        except TypeError:
            return value.model_dump(exclude_unset=True)
    if isinstance(value, Mapping):
        return dict(value)
    raise HTTPException(status_code=422, detail="Each column override must be an object.")


def _merge_overrides(dataset: Any, changes: list[Any]) -> list[ColumnOverride]:
    names = set(dataset.column_names)
    merged: dict[str, dict[str, Any]] = {}
    for existing in getattr(dataset, "column_overrides", []) or []:
        payload = _override_payload(existing)
        try:
            validated = ColumnOverride.model_validate(payload)
        except ValidationError as error:
            raise HTTPException(status_code=422, detail="Stored column policy is invalid.") from error
        merged[validated.column] = validated.model_dump(mode="python")

    for item in getattr(dataset, "column_schema", []) or []:
        if item.get("display_name"):
            merged.setdefault(item["column"], {"column": item["column"]})["display_name"] = item["display_name"]

    seen: set[str] = set()
    for change in changes:
        payload = _override_payload(change)
        try:
            validated = ColumnOverride.model_validate(payload)
        except ValidationError as error:
            raise HTTPException(status_code=422, detail="Column override is invalid.") from error
        if validated.column not in names:
            raise HTTPException(
                status_code=422,
                detail=f"Column override refers to an unknown column: {validated.column}.",
            )
        if validated.column in seen:
            raise HTTPException(
                status_code=422,
                detail=f"Column override is duplicated: {validated.column}.",
            )
        seen.add(validated.column)
        base = dict(merged.get(validated.column, {"column": validated.column}))
        base.update(payload)
        # Revalidate the complete merged row before any ingestion or mutation.
        try:
            final = ColumnOverride.model_validate(base)
        except ValidationError as error:
            raise HTTPException(status_code=422, detail="Column override is invalid.") from error
        if final.parse_as == "auto" and all(getattr(final, field) is None for field in ("role", "format", "unit", "aggregation", "display_name")):
            merged.pop(final.column, None)
        else:
            merged[final.column] = final.model_dump(mode="python")
    try:
        return [ColumnOverride.model_validate(row) for row in merged.values()]
    except ValidationError as error:
        raise HTTPException(status_code=422, detail="Column override is invalid.") from error



def _apply_label_edits(dataset: Any, expected_version: str, changes: list[Any], overrides: list[ColumnOverride]):
    """Update cosmetic metadata under the same optimistic edit/version contract."""
    from copy import deepcopy
    from storage import DATASETS, _DATASETS_LOCK
    from services.response_builders import build_upload_response

    with _DATASETS_LOCK:
        if DATASETS.get(dataset.id) is not dataset or dataset.cache_version != expected_version:
            raise HTTPException(status_code=409, detail="Dataset schema has changed. Refresh before applying edits.")
        previous_status = get_enrichment_status(dataset.id, expected_version)
        schema = deepcopy(dataset.column_schema)
        by_name = {item['column']: item for item in schema}
        for change in changes:
            payload = _override_payload(change)
            validated = ColumnOverride.model_validate(payload)
            by_name[validated.column]['display_name'] = validated.display_name
            by_name[validated.column]['display_name_provenance'] = 'user' if validated.display_name else None
        dataset.column_schema = schema
        dataset.column_overrides = overrides
        dataset.invalidate_cached_results()
        disk = getattr(dataset, 'disk', None)
        if disk:
            with disk._lock:
                disk.column_schema = schema
                disk.column_overrides = overrides
                disk.cache_version = dataset.cache_version
                for name, item in by_name.items():
                    disk._column_policies[name]['display_name'] = item.get('display_name')
                disk.column_summaries = [summary.model_copy(update={
                    'display_name': by_name[summary.name].get('display_name')}) for summary in disk.column_summaries]
                response = disk.to_upload_response(dataset.id, dataset.filename)
        else:
            dataset.df.attrs['column_schema'] = schema
            quality = max(0, 100 - sum(dataset.missing_counts.values()) / max(dataset.row_count * len(dataset.column_names), 1) * 100)
            response = build_upload_response(
                ds_info=dataset, df=dataset.df, col_types=dataset.column_types, col_formats=dataset.column_formats,
                missing_counts=dataset.missing_counts, cleaning_actions=dataset.cleaning_actions,
                quality=quality, profile=dataset.profile, default_chart=dataset.default_chart,
                suggestions=dataset.suggestions, summary=dataset.summary, column_stats=dataset.column_stats)
        response.summary = previous_status.get('summary') or response.summary
        response.version = dataset.cache_version
        response.column_schema = schema
        return response


def _source_path(dataset: Any) -> Path:
    disk = getattr(dataset, "disk", None)
    path = getattr(disk, "source_path", None) if disk is not None else getattr(dataset, "source_path", None)
    if path is None:
        raise HTTPException(status_code=409, detail="The original dataset source is no longer available.")
    source = Path(path)
    if not source.is_file():
        raise HTTPException(status_code=409, detail="The original dataset source is no longer available.")
    return source


def apply_schema(
    dataset_id: str,
    expected_version: str,
    column_overrides: list[Any],
    import_settings=None,
):
    """Rebuild and atomically replace a dataset after a complete schema edit.

    The old handle is leased while its source is copied. Ingestion receives a
    detached copy and performs the version comparison in the storage registry
    at the replacement point, so concurrent edits cannot overwrite one another.
    """
    if not isinstance(expected_version, str) or not expected_version:
        raise HTTPException(status_code=422, detail="An expected dataset version is required.")

    with lease_dataset(dataset_id) as dataset:
        if dataset.cache_version != expected_version:
            raise HTTPException(status_code=409, detail="Dataset schema has changed. Refresh before applying edits.")

        overrides = _merge_overrides(dataset, column_overrides)
        previous_settings = ImportSettings.model_validate(_model_dict(getattr(dataset, "import_settings", None)))
        settings = ImportSettings.model_validate(import_settings) if import_settings is not None else previous_settings
        if (settings.delimiter, settings.encoding) != (previous_settings.delimiter, previous_settings.encoding):
            if column_overrides:
                raise HTTPException(status_code=422, detail="Apply delimiter or encoding changes separately from column edits; then review the new columns.")
            # Column identities may change. Never transfer old overrides to a
            # different set of source fields merely because names coincide.
            overrides = []
        label_only = column_overrides and all(
            set(_override_payload(change)) <= {'column', 'display_name'}
            for change in column_overrides)
        if label_only and settings == previous_settings:
            return _apply_label_edits(dataset, expected_version, column_overrides, overrides)
        source = _source_path(dataset)

        # The lease keeps the old source alive through the ingestion copy and
        # atomic registry swap. Ingestion owns the detached staged source before
        # it publishes the replacement.
        try:
            return ingest_csv(
                source,
                dataset.filename,
                "/schema",
                import_settings=settings,
                column_overrides=overrides,
                replacement=(dataset_id, expected_version),
                defer_enrichment=True,
                enqueue_enrichment=False,
            )
        except ValidationError as error:
            raise HTTPException(status_code=422, detail="Schema settings are invalid.") from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
