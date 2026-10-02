"""Filtering and column validation utilities."""

import difflib
from decimal import Decimal, InvalidOperation
from typing import Optional

import pandas as pd

from core.config import MAX_CHART_POINTS
from models import FilterConfig


class FilterValidationError(ValueError):
    """A filter could not be applied without changing its meaning."""


def validate_columns(df: pd.DataFrame, cols: list[str]) -> tuple[bool, list[str], dict[str, list[str]]]:
    """Validate that columns exist, suggesting alternatives for chart axes."""
    df_cols = df.columns.tolist()
    missing = [c for c in cols if c not in df_cols]
    if not missing:
        return True, [], {}
    suggestions = {c: difflib.get_close_matches(c, df_cols, n=3, cutoff=0.4) for c in missing}
    return False, missing, suggestions


def _as_numeric(value, column: str) -> float:
    if isinstance(value, bool):
        raise FilterValidationError(f"Filter for '{column}' requires a numeric value.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FilterValidationError(f"Filter for '{column}' requires a numeric value.") from exc
    if pd.isna(number) or number in (float("inf"), float("-inf")):
        raise FilterValidationError(f"Filter for '{column}' requires a finite numeric value.")
    return number


def _as_integer_operand(value, column: str):
    """Keep integer keys and fractional boundaries exact instead of coercing to float."""
    if isinstance(value, bool):
        raise FilterValidationError(f"Filter for '{column}' requires a numeric value.")
    try:
        token = str(value)
        if len(token) > 512:
            raise ValueError("Numeric operand too long")
        number = Decimal(token)
        if not number.is_finite() or abs(number.adjusted()) > 512:
            raise ValueError("Numeric operand out of range")
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise FilterValidationError(f"Filter for '{column}' requires a finite numeric value.") from exc
    return int(number) if number == number.to_integral_value() else number


def _as_datetime(value, series: pd.Series, column: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FilterValidationError(f"Filter for '{column}' requires a valid date or timestamp.") from exc
    if pd.isna(stamp):
        raise FilterValidationError(f"Filter for '{column}' requires a valid date or timestamp.")

    source_tz = series.dt.tz
    if source_tz is None and stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    elif source_tz is not None and stamp.tzinfo is None:
        stamp = stamp.tz_localize(source_tz)
    elif source_tz is not None:
        stamp = stamp.tz_convert(source_tz)
    return stamp


def _coerce_value(series: pd.Series, value, column: str):
    if pd.api.types.is_datetime64_any_dtype(series):
        return _as_datetime(value, series, column)
    if pd.api.types.is_bool_dtype(series):
        if type(value) is not bool:
            raise FilterValidationError(f"Filter for '{column}' requires a boolean value.")
        return value
    if pd.api.types.is_integer_dtype(series):
        return _as_integer_operand(value, column)
    if pd.api.types.is_numeric_dtype(series):
        return _as_numeric(value, column)
    if isinstance(value, (dict, list, tuple, set)) or value is None:
        raise FilterValidationError(f"Filter for '{column}' requires a scalar value.")
    return str(value)


def _comparison_mask(series: pd.Series, operator: str, value, column: str) -> pd.Series:
    is_datetime = pd.api.types.is_datetime64_any_dtype(series)
    is_numeric = pd.api.types.is_numeric_dtype(series)
    is_boolean = pd.api.types.is_bool_dtype(series)
    if operator == "contains":
        if is_numeric or is_datetime or is_boolean:
            raise FilterValidationError(f"The contains operator is not supported for '{column}'.")
        return series.astype("string").str.contains(value, case=False, na=False, regex=False)

    if operator != "eq" and not (is_numeric or is_datetime):
        raise FilterValidationError(f"Operator '{operator}' requires a numeric or date column ('{column}').")

    compare_value = _coerce_value(series, value, column)
    if pd.api.types.is_integer_dtype(series):
        left = series.astype(object)
    elif is_numeric:
        left = pd.to_numeric(series, errors="coerce")
    elif is_datetime:
        left = series
    elif is_boolean:
        left = series
    else:
        left = series.astype("string")

    if operator == "eq":
        mask = left == compare_value
    elif operator == "gt":
        mask = left > compare_value
    elif operator == "lt":
        mask = left < compare_value
    elif operator == "gte":
        mask = left >= compare_value
    elif operator == "lte":
        mask = left <= compare_value
    else:
        raise FilterValidationError(f"Unsupported filter operator '{operator}'.")
    return mask.fillna(False)


def _apply_one(filtered: pd.DataFrame, f: FilterConfig, applied: list[str]) -> pd.DataFrame:
    if f.column not in filtered.columns:
        raise FilterValidationError(f"Unknown filter column '{f.column}'.")
    series = filtered[f.column]

    if f.operator is not None:
        filtered = filtered[_comparison_mask(series, f.operator, f.value, f.column)]
        applied.append(f"{f.column} {f.operator} {f.value}")
        series = filtered[f.column]

    if f.values is not None:
        includes_null = any(value is None for value in f.values)
        normalized = [_coerce_value(series, value, f.column) for value in f.values if value is not None]
        if pd.api.types.is_integer_dtype(series):
            left = series.astype(object)
        elif pd.api.types.is_numeric_dtype(series):
            left = pd.to_numeric(series, errors="coerce")
        elif pd.api.types.is_datetime64_any_dtype(series):
            left = series
        elif pd.api.types.is_bool_dtype(series):
            left = series
        else:
            left = series.astype("string")
        mask = left.isin(normalized)
        if includes_null:
            mask = mask | series.isna()
        filtered = filtered[mask]
        preview = ", ".join(str(value) for value in f.values[:3])
        applied.append(f"{f.column} in ({preview})")
        series = filtered[f.column]

    if f.min_val is not None and f.max_val is not None:
        lower = _coerce_value(series, f.min_val, f.column)
        upper = _coerce_value(series, f.max_val, f.column)
        if lower > upper:
            raise FilterValidationError(f"Minimum filter value exceeds maximum for '{f.column}'.")

    if f.min_val is not None:
        filtered = filtered[_comparison_mask(series, "gte", f.min_val, f.column)]
        applied.append(f"{f.column} >= {f.min_val}")
        series = filtered[f.column]
    if f.max_val is not None:
        filtered = filtered[_comparison_mask(series, "lte", f.max_val, f.column)]
        applied.append(f"{f.column} <= {f.max_val}")

    return filtered


def resolve_effective_limit(requested_limit: Optional[int], default_limit: int) -> tuple[int, Optional[str]]:
    """Resolve a request limit while keeping chart responses within the hard cap."""
    raw_limit = requested_limit if requested_limit is not None else default_limit
    if raw_limit < 0:
        raise ValueError("Limit must be zero or a positive integer.")
    if raw_limit == 0:
        return MAX_CHART_POINTS, (
            f"Result capped at {MAX_CHART_POINTS} points for performance. "
            "Add filters to narrow the data."
        )
    if raw_limit > MAX_CHART_POINTS:
        return MAX_CHART_POINTS, (
            f"Requested {raw_limit} points, capped at {MAX_CHART_POINTS} for performance. "
            "Add filters to narrow the data."
        )
    return raw_limit, None


def apply_filters(df: pd.DataFrame, filters: Optional[list[FilterConfig]]) -> tuple[pd.DataFrame, list[str]]:
    """Apply every filter or fail clearly; malformed filters are never skipped."""
    if not filters:
        return df, []

    applied: list[str] = []
    filtered = df.copy()
    for filter_config in filters:
        filtered = _apply_one(filtered, filter_config, applied)
    return filtered, applied
