"""Dataset ingestion orchestration service."""

import os
from time import perf_counter
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
from services.response_builders import build_upload_response
from storage import DatasetInfo, store_dataset
from utils.pipeline_logging import print_pipeline_timing


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


def ingest_dataframe(df: pd.DataFrame, filename: str, endpoint_name: str) -> UploadResponse:
    """Run the ingestion pipeline and return an UploadResponse."""
    endpoint_start = perf_counter()
    raw_df = df.copy(deep=True)

    t2 = perf_counter()
    df, cleaning_actions, missing_counts, col_formats, llm_col_types = clean_dataframe(df)
    t3 = perf_counter()

    col_types = {
        col: llm_col_types.get(col) or detect_semantic_type(df, col)
        for col in df.columns
    }

    t4 = perf_counter()
    profile = auto_profile(df, col_types, col_formats)
    t5 = perf_counter()

    total_cells = len(df) * len(df.columns)
    missing_total = sum(missing_counts.values())
    quality = max(0, 100 - (missing_total / max(total_cells, 1) * 100))

    t6 = perf_counter()
    summary = _generate_business_summary(filename, df)
    t7 = perf_counter()

    default_chart = generate_default_chart(df, col_types)
    suggestions = generate_dynamic_suggestions(df, col_types, col_formats)

    ds_info = DatasetInfo(
        df=df,
        raw_df=raw_df,
        filename=filename,
        cleaning_actions=cleaning_actions,
        missing_counts=missing_counts,
        column_types=col_types,
        column_formats=col_formats,
        profile=profile,
        default_chart=default_chart,
        suggestions=suggestions,
        summary=summary,
    )
    response = build_upload_response(
        ds_info=ds_info,
        df=df,
        col_types=col_types,
        col_formats=col_formats,
        missing_counts=missing_counts,
        cleaning_actions=cleaning_actions,
        quality=quality,
        profile=profile,
        default_chart=default_chart,
        suggestions=suggestions,
        summary=summary,
    )
    store_dataset(ds_info)

    t8 = perf_counter()
    durations = {
        "csv_ingestion": t2 - endpoint_start,
        "data_cleaning": t3 - t2,
        "data_profiling": t5 - t4,
        "llm_summary": t7 - t6,
        "total": t8 - endpoint_start,
    }
    print_pipeline_timing(endpoint_name, durations)

    return response
