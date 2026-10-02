"""Shared, compact per-column statistics for ingestion decisions and summaries."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional

import pandas as pd


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


ColumnStatisticsMap = Mapping[str, ColumnStatistics]


def _is_numeric_year(series: pd.Series, column: str) -> bool:
    if "year" not in set(str(column).lower().split("_")) or not pd.api.types.is_numeric_dtype(series):
        return False
    values = series.dropna()
    return bool(
        len(values)
        and values.map(
            lambda value: float(value).is_integer() and 1000 <= value <= 2200
        ).all()
    )


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
        statistics[column] = ColumnStatistics(
            unique_count=len(distinct_values),
            sample_values=tuple(distinct_values[:20]),
            numeric_year=numeric_year,
        )
    return MappingProxyType(statistics)
