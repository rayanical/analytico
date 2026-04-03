"""Dataset ingestion orchestration service."""

from time import perf_counter
from typing import Optional

import pandas as pd

from core.config import OPENAI_CLIENT
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
    summary = None
    try:
        summary_prompt = f"""Summarize this dataset in 1-2 business sentences.
            Filename: {filename}
            Columns: {', '.join(df.columns[:20])}
            Sample Data: {df.head(3).to_string()}
            """

        summary_resp = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data analyst. Be concise. No preamble."},
                {"role": "user", "content": summary_prompt},
            ],
            max_tokens=100,
            temperature=0.3,
        )
        summary = summary_resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Summary generation failed: {e}")
    return summary


def ingest_dataframe(df: pd.DataFrame, filename: str, endpoint_name: str) -> UploadResponse:
    """Run the ingestion pipeline and return an UploadResponse."""
    endpoint_start = perf_counter()

    t2 = perf_counter()
    df, cleaning_actions, missing_counts, col_formats, llm_col_types = clean_dataframe(df)
    t3 = perf_counter()

    col_types = {
        col: llm_col_types.get(col) or detect_semantic_type(df, col)
        for col in df.columns
    }

    t4 = perf_counter()
    profile = auto_profile(df, col_types)
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

    return build_upload_response(
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
