"""
Analytico Backend - Intelligence Module
Semantic detection, auto-analysis, chart generation, and profiling
"""

from typing import Optional
import pandas as pd

from modules.column_statistics import ColumnStatisticsMap, compute_column_statistics, _is_numeric_year as validated_numeric_year
from modules.column_usage import infer_column_usage, default_chart_plan


class SemanticType:
    METRIC = "metric"
    IDENTIFIER = "identifier"
    TEMPORAL = "temporal"
    CATEGORICAL = "categorical"


def _physical_type(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    return "number" if pd.api.types.is_numeric_dtype(series) else "text"


def dataframe_column_usages(df, column_stats=None, column_types=None):
    statistics = column_stats if column_stats is not None else compute_column_statistics(df)
    schema = {item["column"]: item for item in df.attrs.get("column_schema", [])}
    usages = {}
    for column in df.columns:
        policy = schema.get(column, {})
        role = policy.get("role") if policy.get("provenance") == "override" else None
        if column_types is not None and role is None:
            # Preserve existing reviewed/applied decisions supplied by ingestion.
            role = column_types[column] if column_types[column] != infer_column_usage(column, _physical_type(df[column]), statistics[column]).role else None
        usages[column] = infer_column_usage(column, _physical_type(df[column]), statistics[column],
            role_override=role, aggregation_override=policy.get("aggregation"))
    return usages


def detect_semantic_type(df, col, column_stats=None) -> str:
    statistics = column_stats if column_stats is not None else compute_column_statistics(df[[col]])
    return infer_column_usage(col, _physical_type(df[col]), statistics[col]).role


def _accepted_metric_decision(df: pd.DataFrame, column: str) -> Optional[dict]:
    for item in df.attrs.get("column_schema", []):
        if item.get("column") == column and item.get("aggregation") is not None:
            return {"recommended_aggregation": item["aggregation"]}
    metadata = df.attrs.get("column_interpretations", {}).get(column, {})
    decision = metadata.get("decision")
    if (metadata.get("runtime_status") == "applied" and decision
            and decision.get("recommended_aggregation") in {"sum", "mean"}):
        return decision
    return None


def generate_default_chart(
    df: pd.DataFrame,
    column_types: dict[str, str],
    column_stats: Optional[ColumnStatisticsMap] = None,
) -> Optional[dict]:
    """Generate the best default chart configuration"""
    statistics = column_stats if column_stats is not None else compute_column_statistics(df)
    return default_chart_plan(list(df.columns), dataframe_column_usages(df, statistics, column_types), statistics)


def _is_numeric_year(
    series: pd.Series,
    col: str,
    column_stats: Optional[ColumnStatisticsMap] = None,
) -> bool:
    if "year" not in set(str(col).lower().split("_")) or not pd.api.types.is_numeric_dtype(series):
        return False
    statistics = column_stats.get(col) if column_stats is not None else None
    if statistics is not None and statistics.numeric_year is not None:
        return statistics.numeric_year
    return validated_numeric_year(series, col)



def auto_profile(
    df: pd.DataFrame,
    column_types: dict[str, str],
    column_formats: Optional[dict[str, str]] = None,
    column_stats: Optional[ColumnStatisticsMap] = None,
) -> dict:
    """Generate executive summary / auto-profile"""
    profile = {
        "top_metrics": [],
        "time_range": None,
        "row_count": len(df),
        "column_count": len(df.columns)
    }
    
    # Find metric columns for summary
    usages = dataframe_column_usages(df, column_stats, column_types)
    metric_cols = [c for c in df.columns if usages[c].automatic_measure]
    column_formats = column_formats or {}
    
    for col in metric_cols[:3]:  # Top 3 metrics
        decision = _accepted_metric_decision(df, col)
        if decision and decision["recommended_aggregation"] in {"none", "count"}:
            continue
        series = df[col].dropna()
        if len(series) == 0:
            continue
        decision = _accepted_metric_decision(df, col)
        recommended_aggregation = decision["recommended_aggregation"] if decision else "mean"
        profile["top_metrics"].append({
            "name": col,
            "total": float(series.sum()),
            "average": float(series.mean()),
            "min": float(series.min()),
            "max": float(series.max()),
            "aggregation": recommended_aggregation,
        })
    
    # Find temporal columns for range
    temporal_cols = [
        c for c, t in column_types.items()
        if t == SemanticType.TEMPORAL
        and (
            pd.api.types.is_datetime64_any_dtype(df[c])
            or _is_numeric_year(df[c], c, column_stats)
        )
    ]
    if temporal_cols:
        date_col = temporal_cols[0]
        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
            valid_dates = df[date_col].dropna()
            if len(valid_dates) > 0:
                profile["time_range"] = {
                    "column": date_col,
                    "start": str(valid_dates.min()),
                    "end": str(valid_dates.max())
                }

    return profile


def generate_dynamic_suggestions(df: pd.DataFrame, column_types: dict[str, str], column_formats: dict[str, str], column_stats=None) -> list[str]:
    """Generate concise, intent-diverse, dataset-aware example prompts."""
    suggestions: list[str] = []
    
    usages = dataframe_column_usages(df, column_stats, column_types)
    metric_cols = [c for c in df.columns if usages[c].automatic_measure]
    categorical_cols = [c for c, t in column_types.items() if t == SemanticType.CATEGORICAL]
    temporal_cols = [c for c, t in column_types.items() if t == SemanticType.TEMPORAL]
    
    def fmt(col: str) -> str:
        return col.replace('_', ' ')

    def normalize_prompt(text: str, max_len: int = 72) -> str:
        compact = " ".join(text.split())
        if len(compact) <= max_len:
            return compact
        return compact[: max_len - 1].rstrip() + "…"

    # 1) Breakdown intent
    if categorical_cols and metric_cols:
        suggestions.append(
            f"Show {fmt(metric_cols[0])} by categories in {fmt(categorical_cols[0])}"
        )

    # 2) Trend intent
    if temporal_cols and metric_cols:
        suggestions.append(
            f"How does {fmt(metric_cols[0])} trend over {fmt(temporal_cols[0])}?"
        )

    # 3) Distribution/extreme intent
    if categorical_cols and metric_cols:
        suggestions.append(
            f"Which {fmt(categorical_cols[0])} categories have the highest {fmt(metric_cols[0])}?"
        )
    elif metric_cols:
        suggestions.append(f"What is the distribution of {fmt(metric_cols[0])}?")

    # Fallbacks to ensure useful and diverse prompts
    if len(metric_cols) >= 2:
        suggestions.append(f"Compare {fmt(metric_cols[0])} vs {fmt(metric_cols[1])}")
    if metric_cols:
        suggestions.append(f"What is the average {fmt(metric_cols[0])}?")
    suggestions.append("Give me a quick summary of this dataset")

    # Deduplicate while preserving order, then length-guard for UI.
    deduped: list[str] = []
    seen = set()
    for prompt in suggestions:
        normalized = normalize_prompt(prompt)
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
        if len(deduped) == 3:
            break

    while len(deduped) < 3:
        deduped.append("Show top insights from this data")

    return deduped[:3]
