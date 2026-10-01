"""Date bucketing, top-category selection, and grouped aggregation."""

from typing import Optional

import pandas as pd

from .intelligence import SemanticType

_AGGREGATIONS = {"sum", "mean", "median", "count", "min", "max"}
_BUCKET_FREQS = {"year": "Y", "month": "M", "week": "W-SUN"}


def smart_group_top_n(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    agg: str = "sum",
    n: int = 19,
    group_others: bool = True,
) -> pd.DataFrame:
    """Keep raw rows for the top groups, optionally mapping the tail to Others.

    Ranking uses the requested aggregation on the first measure. Grouping and
    measure aggregation happen later in ``aggregate_data`` exactly once.
    """
    if df.empty or x_col not in df.columns or not y_cols:
        return df
    if agg not in _AGGREGATIONS:
        raise ValueError(f"Unsupported aggregation '{agg}'.")
    if not group_others or n <= 0 or df[x_col].nunique(dropna=False) <= n:
        return df

    values = df.groupby(x_col, dropna=False, sort=False)[y_cols[0]]
    ranking = values.sum(min_count=1) if agg == "sum" else values.agg(agg)
    ranking_frame = ranking.rename("_score").to_frame()
    ranking_frame["_label"] = ["" if pd.isna(value) else str(value) for value in ranking_frame.index]
    ranking_frame = ranking_frame.sort_values(
        ["_score", "_label"], ascending=[False, True], na_position="last", kind="mergesort"
    )

    top_count = max(n - 1, 0)
    top_values = ranking_frame.index[:top_count].tolist()
    keep_mask = df[x_col].isin(top_values)
    if any(pd.isna(value) for value in top_values):
        keep_mask = keep_mask | df[x_col].isna()

    existing_labels = {str(value) for value in df[x_col].dropna().unique()}
    others_label = "Others"
    suffix = 2
    while others_label in existing_labels:
        others_label = f"Others ({suffix})"
        suffix += 1
    result = df.copy()
    result[x_col] = result[x_col].astype(object).where(keep_mask, others_label)
    result.attrs["others_label"] = others_label
    return result


def smart_resample_dates(
    df: pd.DataFrame,
    date_col: str,
    y_cols: Optional[list[str]] = None,
    agg: str = "sum",
    bucket: Optional[str] = None,
) -> tuple[pd.DataFrame, Optional[str]]:
    """Replace each datetime with its period start, preserving every source row."""
    if date_col not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        if bucket is not None:
            raise ValueError(f"Time bucketing requires a datetime column ('{date_col}').")
        return df, None

    selected_bucket = bucket
    if selected_bucket is None:
        dates = df[date_col].dropna()
        if dates.nunique() <= 100 or dates.empty:
            return df, None
        range_days = (dates.max() - dates.min()).days
        selected_bucket = "year" if range_days > 730 else "month" if range_days > 180 else "week"

    if selected_bucket not in _BUCKET_FREQS:
        raise ValueError(f"Unsupported time bucket '{selected_bucket}'.")

    result = df.copy()
    source_tz = result[date_col].dt.tz
    dates = result[date_col]
    if source_tz is not None:
        dates = dates.dt.tz_localize(None)
    bucket_starts = dates.dt.to_period(_BUCKET_FREQS[selected_bucket]).dt.start_time
    if source_tz is not None:
        bucket_starts = bucket_starts.dt.tz_localize(source_tz, ambiguous="NaT", nonexistent="shift_forward")
    result[date_col] = bucket_starts
    return result, selected_bucket


def enforce_semantic_rules(
    aggregation: str,
    y_axis_keys: list[str],
    column_types: dict[str, str],
) -> tuple[str, list[str], Optional[str]]:
    """Keep the selected aggregation while describing per-column compatibility."""
    if aggregation not in _AGGREGATIONS:
        raise ValueError(f"Unsupported aggregation '{aggregation}'.")

    identifiers = [column for column in y_axis_keys if column_types.get(column) == SemanticType.IDENTIFIER]
    warnings: list[str] = []
    if identifiers and aggregation != "count":
        raise ValueError(f"Identifier measures require count aggregation: {identifiers}")
    label = "Count of Records" if aggregation == "count" else None
    return aggregation, warnings, label


def aggregate_data(
    df: pd.DataFrame,
    x_key: str,
    y_keys: list[str],
    agg: str,
    limit: int = 0,
) -> tuple[pd.DataFrame, bool]:
    """Group once by X, preserving null groups and null sums."""
    if agg not in _AGGREGATIONS:
        raise ValueError(f"Unsupported aggregation '{agg}'.")
    if x_key not in df.columns or not y_keys or any(key not in df.columns for key in y_keys):
        raise ValueError("Aggregation columns are missing.")

    for key in y_keys:
        if not pd.api.types.is_numeric_dtype(df[key]) and agg != "count":
            raise ValueError(f"Measure '{key}' is not numeric; choose count or a numeric measure.")
    grouped = df.groupby(x_key, as_index=False, dropna=False, sort=False)[y_keys]
    if agg == "sum":
        grouped = grouped.sum(min_count=1)
    else:
        grouped = grouped.agg(agg)
    if limit > 0:
        was_capped = len(grouped) > limit
        return grouped.head(limit), was_capped
    return grouped, False
