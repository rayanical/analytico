"""Centralized response shaping helpers."""

from typing import Optional

import pandas as pd

from models import (
    ChartResponse,
    ColumnSummary,
    DataHealth,
    DataProfile,
    DefaultChart,
    MetricSummary,
    TimeRange,
    UploadResponse,
)
from storage import DatasetInfo


def get_column_summary(df: pd.DataFrame, col: str, sem_type: str, fmt: str) -> ColumnSummary:
    """Generate column summary for API response."""
    series = df[col]
    unique_vals = series.dropna().unique()
    sample_vals = sorted([str(v) for v in unique_vals[:20]], key=str.lower)
    return ColumnSummary(
        name=col,
        dtype=str(series.dtype),
        is_numeric=pd.api.types.is_numeric_dtype(series),
        is_datetime=pd.api.types.is_datetime64_any_dtype(series),
        semantic_type=sem_type,
        format=fmt,
        unique_count=int(series.nunique()),
        sample_values=sample_vals,
    )


def build_upload_response(
    ds_info: DatasetInfo,
    df: pd.DataFrame,
    col_types: dict[str, str],
    col_formats: dict[str, str],
    missing_counts: dict[str, int],
    cleaning_actions: list[str],
    quality: float,
    profile: dict,
    default_chart: Optional[dict],
    suggestions: list[str],
    summary: Optional[str],
) -> UploadResponse:
    """Build UploadResponse payload with consistent shape."""
    return UploadResponse(
        dataset_id=ds_info.id,
        filename=ds_info.filename,
        row_count=len(df),
        columns=[
            get_column_summary(df, c, col_types.get(c, "unknown"), col_formats.get(c, "general"))
            for c in df.columns
        ],
        column_formats=col_formats,
        data_health=DataHealth(
            missing_values=missing_counts,
            cleaning_actions=cleaning_actions,
            quality_score=round(quality, 1),
        ),
        profile=DataProfile(
            top_metrics=[MetricSummary(**m) for m in profile["top_metrics"]],
            time_range=TimeRange(**profile["time_range"]) if profile["time_range"] else None,
            row_count=profile["row_count"],
            column_count=profile["column_count"],
        ),
        default_chart=DefaultChart(**default_chart) if default_chart else None,
        suggestions=suggestions,
        summary=summary,
    )


def safe_empty_chart_response(message: str, title: str = "Clarification Needed") -> ChartResponse:
    """Return a zero-data chart response for graceful clarification UX."""
    return ChartResponse(
        data=[],
        x_axis_key="",
        y_axis_keys=[],
        chart_type="empty",
        title=title,
        aggregation=None,
        row_count=0,
        y_axis_label="",
        analysis=message,
        answer=None,
    )
