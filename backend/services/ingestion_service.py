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
from storage import DatasetInfo, store_dataset
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
    except Exception as e:
        print(f"Summary generation failed: {e}")
    return summary


def ingest_dataframe(
    df: pd.DataFrame, filename: str, endpoint_name: str,
    measurement: Optional[IngestionMeasurement] = None,
) -> UploadResponse:
    """Run ingestion; callers parsing CSV supply their already-started measurement."""
    scope = nullcontext(measurement) if measurement is not None else IngestionMeasurement(endpoint_name)
    with scope as metrics:
        metrics.rows, metrics.columns = df.shape
        with metrics.phase("raw_copy"):
            raw_df = df.copy(deep=True)
        with metrics.phase("data_cleaning"):
            df, cleaning_actions, missing_counts, col_formats, llm_col_types = clean_dataframe(df)
        with metrics.phase("column_statistics"):
            column_stats = compute_column_statistics(df)
        with metrics.phase("semantic_detection"):
            col_types = {
                col: llm_col_types.get(col) or detect_semantic_type(df, col, column_stats)
                for col in df.columns
            }
        with metrics.phase("data_profiling"):
            profile = auto_profile(df, col_types, col_formats, column_stats=column_stats)

        total_cells = len(df) * len(df.columns)
        missing_total = sum(missing_counts.values())
        quality = max(0, 100 - (missing_total / max(total_cells, 1) * 100))

        with metrics.phase("llm_summary"):
            summary = _generate_business_summary(filename, df)
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
            response = build_upload_response(
                ds_info=ds_info, df=df, col_types=col_types, col_formats=col_formats,
                missing_counts=missing_counts, cleaning_actions=cleaning_actions,
                quality=quality, profile=profile, default_chart=default_chart,
                suggestions=suggestions, summary=summary, column_stats=column_stats,
            )
            store_dataset(ds_info)
        return response
