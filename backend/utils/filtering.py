"""Filtering and column validation utilities."""

import difflib
from typing import Optional

import pandas as pd

from core.config import ALLOWED_FILTER_OPERATORS, MAX_CHART_POINTS
from models import FilterConfig


def validate_columns(df: pd.DataFrame, cols: list[str]) -> tuple[bool, list[str], dict[str, list[str]]]:
    """Validate that columns exist, suggest alternatives if not."""
    df_cols = df.columns.tolist()
    missing = [c for c in cols if c not in df_cols]
    if not missing:
        return True, [], {}
    suggestions = {c: difflib.get_close_matches(c, df_cols, n=3, cutoff=0.4) for c in missing}
    return False, missing, suggestions


def _apply_operator_filter(filtered: pd.DataFrame, f: FilterConfig, applied: list[str]) -> pd.DataFrame:
    """Apply a single operator-style filter safely without mutating column dtypes."""
    if not f.operator or f.value is None:
        return filtered

    op = f.operator.lower()
    if op not in ALLOWED_FILTER_OPERATORS:
        applied.append(f"{f.column}: skipped invalid operator '{f.operator}'")
        return filtered

    col = filtered[f.column]
    mask = None

    if op == "eq":
        mask = col.astype(str) == str(f.value)
        applied.append(f"{f.column} == {f.value}")
    elif op == "contains":
        mask = col.astype(str).str.contains(str(f.value), case=False, na=False, regex=False)
        applied.append(f"{f.column} contains '{f.value}'")
    elif op in {"gt", "lt", "gte", "lte"}:
        if pd.api.types.is_datetime64_any_dtype(col):
            cmp_value = pd.to_datetime(f.value, errors="coerce")
            if pd.isna(cmp_value):
                applied.append(f"{f.column}: skipped invalid datetime value '{f.value}'")
                return filtered
            if op == "gt":
                mask = col > cmp_value
                applied.append(f"{f.column} > {f.value}")
            elif op == "lt":
                mask = col < cmp_value
                applied.append(f"{f.column} < {f.value}")
            elif op == "gte":
                mask = col >= cmp_value
                applied.append(f"{f.column} >= {f.value}")
            elif op == "lte":
                mask = col <= cmp_value
                applied.append(f"{f.column} <= {f.value}")
        else:
            try:
                num_value = float(f.value)
            except (TypeError, ValueError):
                applied.append(f"{f.column}: skipped invalid numeric value '{f.value}'")
                return filtered
            num_col = pd.to_numeric(col, errors="coerce")
            if op == "gt":
                mask = num_col > num_value
                applied.append(f"{f.column} > {f.value}")
            elif op == "lt":
                mask = num_col < num_value
                applied.append(f"{f.column} < {f.value}")
            elif op == "gte":
                mask = num_col >= num_value
                applied.append(f"{f.column} >= {f.value}")
            elif op == "lte":
                mask = num_col <= num_value
                applied.append(f"{f.column} <= {f.value}")

    if mask is None:
        return filtered
    return filtered[mask.fillna(False)]


def resolve_effective_limit(requested_limit: Optional[int], default_limit: int) -> tuple[int, Optional[str]]:
    """Resolve limit with hard cap to avoid frontend rendering performance issues."""
    raw_limit = requested_limit if requested_limit is not None else default_limit
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
    """Apply filter configurations to dataframe."""
    if not filters:
        return df, []

    applied = []
    filtered = df.copy()

    for f in filters:
        if f.column not in filtered.columns:
            continue

        filtered = _apply_operator_filter(filtered, f, applied)

        if f.values:
            str_values = [str(v) for v in f.values]
            filtered = filtered[filtered[f.column].astype(str).isin(str_values)]
            applied.append(f"{f.column}: {', '.join(str(v) for v in f.values[:3])}")

        if f.min_val is not None:
            filtered = _apply_operator_filter(
                filtered,
                FilterConfig(column=f.column, operator="gte", value=f.min_val),
                applied,
            )
        if f.max_val is not None:
            filtered = _apply_operator_filter(
                filtered,
                FilterConfig(column=f.column, operator="lte", value=f.max_val),
                applied,
            )

    return filtered, applied
