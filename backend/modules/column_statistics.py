"""Shared, compact per-column statistics for ingestion decisions and summaries."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional

import pandas as pd
import numpy as np


@dataclass(frozen=True)
class ColumnStatistics:
    """Statistics retained for one column after a single distinct-value scan.

    ``sample_values`` preserves the first 20 observed non-null distinct values.
    It deliberately excludes the full distinct-value array used during
    computation so the cache stays bounded for high-cardinality columns.
    ``numeric_year`` is populated only for numeric columns whose name contains
    a ``year`` token.
    """

    unique_count: int
    sample_values: tuple[object, ...]
    numeric_year: Optional[bool] = None
    row_position_sequence: bool = False
    numeric_min: float | int | None = None
    numeric_max: float | int | None = None
    has_fractional_values: bool = False


ColumnStatisticsMap = Mapping[str, ColumnStatistics]

# Four-digit calendar years, not a guessed future-year cutoff.
MIN_CALENDAR_YEAR = 1000
MAX_CALENDAR_YEAR = 9999


def _is_numeric_year(series: pd.Series, column: str) -> bool:
    if "year" not in set(str(column).lower().split("_")) or not pd.api.types.is_numeric_dtype(series):
        return False
    values = series.dropna()
    return bool(
        len(values)
        and values.map(
            lambda value: float(value).is_integer() and MIN_CALENDAR_YEAR <= value <= MAX_CALENDAR_YEAR
        ).all()
    )


def is_row_position_sequence(series: pd.Series, unique_count: int | None = None) -> bool:
    """Full-column evidence of a dense 0/1-based row sequence, not a semantic label."""
    count = len(series)
    if count < 3 or not pd.api.types.is_integer_dtype(series.dtype) or series.isna().any():
        return False
    if unique_count is not None and unique_count != count:
        return False
    first = int(series.iloc[0])
    return (first in {0, 1} and int(series.iloc[-1]) == first + count - 1
            and series.is_monotonic_increasing
            and (unique_count if unique_count is not None else series.nunique()) == count)


def compute_column_statistics(df: pd.DataFrame) -> ColumnStatisticsMap:
    """Compute exact distinct counts and bounded samples for each DataFrame column.

    The distinct values are materialized once per column to derive both the
    exact count and first-observed sample. The returned mapping and records are
    immutable, and only up to 20 values per column are retained.
    """
    statistics: dict[str, ColumnStatistics] = {}
    for column in df.columns:
        series = df[column]
        # Drop nulls from the distinct array rather than copying every observed
        # row first. This preserves first-observed order and exact null semantics.
        distinct_values = series.unique()
        distinct_values = distinct_values[pd.notna(distinct_values)]
        numeric_year = None
        if "year" in set(str(column).lower().split("_")) and pd.api.types.is_numeric_dtype(series):
            numeric_year = _is_numeric_year(series, column)
        numeric_min = numeric_max = None
        if len(distinct_values) and pd.api.types.is_numeric_dtype(series):
            numeric_min, numeric_max = distinct_values.min(), distinct_values.max()
            if isinstance(numeric_min, np.generic):
                numeric_min = numeric_min.item()
            if isinstance(numeric_max, np.generic):
                numeric_max = numeric_max.item()
        statistics[column] = ColumnStatistics(
            unique_count=len(distinct_values),
            sample_values=tuple(distinct_values[:20]),
            numeric_year=numeric_year,
            row_position_sequence=is_row_position_sequence(series, len(distinct_values)),
            numeric_min=numeric_min,
            numeric_max=numeric_max,
            has_fractional_values=bool(len(distinct_values) and pd.api.types.is_float_dtype(series)
                                      and np.any(distinct_values != np.floor(distinct_values))),
        )
    return MappingProxyType(statistics)
