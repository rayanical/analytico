"""Dataset ingestion orchestration service."""

import os
from contextlib import nullcontext
from typing import Optional

import pandas as pd

from modules import (
    auto_profile,
    clean_dataframe,
    detect_semantic_type,
    generate_default_chart,
    generate_dynamic_suggestions,
)
from models import UploadResponse
from modules.column_statistics import compute_column_statistics
from services.response_builders import build_upload_response
from storage import DatasetInfo, store_dataset, replace_dataset
from utils.pipeline_logging import IngestionMeasurement


def _generate_business_summary(filename: str, df: pd.DataFrame) -> Optional[str]:
    if not os.getenv("OPENAI_API_KEY", "").strip():
        return None

    summary = None
    try:
        from core.config import OPENAI_MODEL, chat_completion_options, get_openai_client

        client = get_openai_client()
        sample = df.head(3).iloc[:, :20].to_string(max_colwidth=50)
        columns = ", ".join(str(column)[:80] for column in df.columns[:20])
        summary_prompt = f"""Summarize this dataset in 1-2 business sentences.
            Filename: {filename[:120]}
            Columns: {columns}
            Sample Data: {sample}
            """

        summary_resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a data analyst. Be concise. No preamble."},
                {"role": "user", "content": summary_prompt},
            ],
            **chat_completion_options(100, 0.3),
            timeout=8.0,
        )
        content = summary_resp.choices[0].message.content
        summary = content.strip() if content else None
    except Exception:
        print("Summary generation unavailable")
    return summary


def ingest_dataframe(
    df: pd.DataFrame, filename: str, endpoint_name: str,
    measurement: Optional[IngestionMeasurement] = None,
    defer_enrichment: bool = False,
    import_settings=None, column_overrides=None, source_owner=None,
    source_path=None, replacement=None, enqueue_enrichment: bool = True,
) -> UploadResponse:
    """Run ingestion; callers parsing CSV supply their already-started measurement."""
    scope = nullcontext(measurement) if measurement is not None else IngestionMeasurement(endpoint_name)
    with scope as metrics:
        metrics.rows, metrics.columns = df.shape
        with metrics.phase("raw_copy"):
            raw_df = df.copy(deep=True)
        with metrics.phase("data_cleaning"):
            policy_options = {}
            if import_settings is not None or column_overrides:
                policy_options = dict(import_settings=import_settings, column_overrides=column_overrides)
            if defer_enrichment:
                df, cleaning_actions, missing_counts, col_formats, llm_col_types = clean_dataframe(df, interpret_columns=False, **policy_options)
            else:
                df, cleaning_actions, missing_counts, col_formats, llm_col_types = clean_dataframe(df, **policy_options)
        with metrics.phase("column_statistics"):
            column_stats = compute_column_statistics(df)
        with metrics.phase("semantic_detection"):
            col_types = {
                col: llm_col_types.get(col) or detect_semantic_type(df, col, column_stats)
                for col in df.columns
            }
            for item in df.attrs.get("column_schema", []):
                item["role"] = col_types[item["column"]]
                item["format"] = col_formats.get(item["column"], "general")
        with metrics.phase("data_profiling"):
            profile = auto_profile(df, col_types, col_formats, column_stats=column_stats)

        total_cells = len(df) * len(df.columns)
        missing_total = sum(missing_counts.values())
        quality = max(0, 100 - (missing_total / max(total_cells, 1) * 100))

        with metrics.phase("llm_summary"):
            summary = None if defer_enrichment else _generate_business_summary(filename, df)
        with metrics.phase("chart_metadata"):
            default_chart = generate_default_chart(df, col_types, column_stats)
            suggestions = generate_dynamic_suggestions(df, col_types, col_formats)

        with metrics.phase("response_and_storage"):
            ds_info = DatasetInfo(
                df=df, raw_df=raw_df, filename=filename,
                cleaning_actions=cleaning_actions, missing_counts=missing_counts,
                column_types=col_types, column_formats=col_formats, profile=profile,
                default_chart=default_chart, suggestions=suggestions, summary=summary,
            )
            ds_info.import_settings = import_settings
            ds_info.column_overrides = column_overrides or []
            ds_info.column_schema = df.attrs.get("column_schema", [])
            ds_info._source_owner = source_owner
            ds_info.source_path = source_path
            if replacement:
                ds_info.id = replacement[0]
            response = build_upload_response(
                ds_info=ds_info, df=df, col_types=col_types, col_formats=col_formats,
                missing_counts=missing_counts, cleaning_actions=cleaning_actions,
                quality=quality, profile=profile, default_chart=default_chart,
                suggestions=suggestions, summary=summary, column_stats=column_stats,
            )
            if replacement:
                try:
                    replace_dataset(replacement[0], ds_info, replacement[1])
                except Exception:
                    ds_info.close()
                    raise
            else:
                store_dataset(ds_info)
            response.version = ds_info.cache_version
            response.column_schema = ds_info.column_schema
        if defer_enrichment and enqueue_enrichment:
            response.enrichment_status = queue_dataset_enrichment(ds_info, raw_df)
        return response


def queue_dataset_enrichment(dataset, source_frame: Optional[pd.DataFrame] = None) -> str:
    """Optional setup failure cannot invalidate a successfully ingested dataset."""
    try:
        return _queue_dataset_enrichment(dataset, source_frame)
    except Exception:
        from services.enrichment_service import disable_enrichment
        return disable_enrichment(dataset.id, dataset.cache_version, "unavailable")["status"]


def _queue_dataset_enrichment(dataset, source_frame: Optional[pd.DataFrame] = None) -> str:
    """Queue bounded source context; jobs never change the dataset's parsed view."""
    from services.enrichment_service import enqueue_enrichment, manager
    from modules.data_janitor import _interpretation_input
    from modules.column_interpretation import interpret_column
    from storage import DATASETS
    from services.parallel_enrichment import run_parallel_enrichment

    provider = os.getenv("COLUMN_INTERPRETER", "off").strip().lower()
    has_summary_key = bool(os.getenv("OPENAI_API_KEY", "").strip())
    key_name = "AI_GATEWAY_API_KEY" if provider == "jev" else "OPENAI_API_KEY"
    has_interpreter_key = provider in {"luna", "jev"} and bool(os.getenv(key_name, "").strip())
    if not has_summary_key and not has_interpreter_key:
        return manager.disable(dataset.id, dataset.cache_version, "not_configured")["status"]

    summary_frame = dataset.sample_frame(3).iloc[:, :20].copy(deep=True)
    payloads = []
    if has_interpreter_key:
        if source_frame is not None:
            headers = [str(column) for column in source_frame.columns]
            payloads = [(str(dataset.column_names[index]), _interpretation_input(header, source_frame.iloc[:, index], headers))
                        for index, header in enumerate(headers[:256])]
        else:
            payloads = dataset.disk.interpretation_inputs(limit=256)
    dataset_id, version, filename = dataset.id, dataset.cache_version, dataset.filename

    def is_current():
        current = DATASETS.get(dataset_id)
        return current is not None and current.cache_version == version

    def proposal(payload):
        result = interpret_column(payload, provider=provider, use_cache=True)
        return {
            "status": result.status, "runtime_status": "clarification",
            "provider": result.provider or provider, "model": result.model,
            "prompt_version": result.prompt_version, "latency_ms": result.latency_ms,
            "usage": result.usage, "error_code": result.error_code,
            "cache_hit": result.cache_hit,
            "decision": result.decision.model_dump() if result.decision else None,
        }

    def work():
        return run_parallel_enrichment(
            (lambda: _generate_business_summary(filename, summary_frame)) if has_summary_key else None,
            [(column, lambda payload=payload: proposal(payload)) for column, payload in payloads],
            total_columns=len(dataset.column_names), is_current=is_current,
            publish=lambda result: manager.update(dataset_id, version, result),
        )

    return enqueue_enrichment(dataset_id, version, work)["status"]
