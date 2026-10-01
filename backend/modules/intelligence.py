"""
Analytico Backend - Intelligence Module
Semantic detection, auto-analysis, chart generation, and profiling
"""

from typing import Optional
import pandas as pd


class SemanticType:
    METRIC = "metric"
    IDENTIFIER = "identifier"
    TEMPORAL = "temporal"
    CATEGORICAL = "categorical"


def detect_semantic_type(df: pd.DataFrame, col: str) -> str:
    """Detect semantic type of a column"""
    series = df[col]
    tokens = set(col.lower().split("_"))
    
    # Check for datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return SemanticType.TEMPORAL
    
    # Check for identifier patterns
    if tokens & {"id", "code", "key", "name", "email", "phone", "address", "zip", "postal", "ssn", "account", "serial", "ref"}:
        return SemanticType.IDENTIFIER
    
    # Numeric columns
    if pd.api.types.is_numeric_dtype(series):
        if "year" in tokens:
            years = series.dropna()
            if len(years) and years.map(lambda value: float(value).is_integer() and 1000 <= value <= 2200).all():
                return SemanticType.TEMPORAL
        unique_ratio = series.nunique() / max(len(series), 1)
        # High cardinality numeric = likely metric
        if unique_ratio > 0.5:
            return SemanticType.METRIC
        # Low cardinality numeric = could be categorical
        if series.nunique() < 20:
            return SemanticType.CATEGORICAL
        return SemanticType.METRIC
    
    # Non-numeric with low cardinality = categorical
    if series.nunique() < 50:
        return SemanticType.CATEGORICAL
    
    return SemanticType.IDENTIFIER


def generate_default_chart(df: pd.DataFrame, column_types: dict[str, str]) -> Optional[dict]:
    """Generate the best default chart configuration"""
    # Find temporal, categorical, and metric columns
    temporal_cols = [
        c for c, t in column_types.items()
        if t == SemanticType.TEMPORAL
        and (pd.api.types.is_datetime64_any_dtype(df[c]) or _is_numeric_year(df[c], c))
    ]
    categorical_cols = [c for c, t in column_types.items() if t == SemanticType.CATEGORICAL]
    metric_cols = [
        c for c, t in column_types.items()
        if t == SemanticType.METRIC and pd.api.types.is_numeric_dtype(df[c])
    ]
    
    if not metric_cols:
        return None

    metric = metric_cols[0]
    aggregation = "mean"
    
    # Best case: temporal x-axis with metric y-axis
    if temporal_cols:
        temporal = temporal_cols[0]
        return {
            "x_axis_key": temporal,
            "y_axis_keys": [metric],
            "chart_type": "line",
            "aggregation": aggregation,
            "title": f"Mean {metric} by {temporal}".replace('_', ' ').title(),
            "analysis": f"Shows the mean of {metric} for each observed {temporal} value.",
        }
    
    # Second best: categorical x-axis with metric y-axis
    if categorical_cols:
        # Pick categorical with reasonable cardinality
        best_cat = min(categorical_cols, key=lambda c: abs(df[c].nunique() - 10))
        return {
            "x_axis_key": best_cat,
            "y_axis_keys": [metric],
            "chart_type": "bar",
            "aggregation": aggregation,
            "title": f"Mean {metric} by {best_cat}".replace('_', ' ').title(),
            "analysis": f"Shows the mean of {metric} for each observed {best_cat} value.",
        }

    return None


def _is_numeric_year(series: pd.Series, col: str) -> bool:
    if "year" not in set(str(col).lower().split("_")) or not pd.api.types.is_numeric_dtype(series):
        return False
    values = series.dropna()
    return bool(len(values) and values.map(lambda value: float(value).is_integer() and 1000 <= value <= 2200).all())


def _non_additive_metric(col: str, column_format: Optional[str]) -> bool:
    tokens = set(str(col).lower().split("_"))
    joined_name = str(col).lower()
    return column_format == "percentage" or "life_exp" in joined_name or bool(
        tokens & {"rate", "ratio", "pct", "percent", "percentage", "lifeexp", "expectancy", "score", "index", "average", "mean", "age", "temperature"}
    )


def auto_profile(
    df: pd.DataFrame,
    column_types: dict[str, str],
    column_formats: Optional[dict[str, str]] = None,
) -> dict:
    """Generate executive summary / auto-profile"""
    profile = {
        "top_metrics": [],
        "time_range": None,
        "row_count": len(df),
        "column_count": len(df.columns)
    }
    
    # Find metric columns for summary
    metric_cols = [
        c for c, t in column_types.items()
        if t == SemanticType.METRIC and pd.api.types.is_numeric_dtype(df[c])
    ]
    column_formats = column_formats or {}
    
    for col in metric_cols[:3]:  # Top 3 metrics
        series = df[col].dropna()
        if len(series) == 0:
            continue
        recommended_aggregation = "mean" if _non_additive_metric(col, column_formats.get(col)) else "sum"
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
        and (pd.api.types.is_datetime64_any_dtype(df[c]) or _is_numeric_year(df[c], c))
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


def generate_dynamic_suggestions(df: pd.DataFrame, column_types: dict[str, str], column_formats: dict[str, str]) -> list[str]:
    """Generate concise, intent-diverse, dataset-aware example prompts."""
    suggestions: list[str] = []
    
    metric_cols = [
        c for c, t in column_types.items()
        if t == SemanticType.METRIC and pd.api.types.is_numeric_dtype(df[c])
    ]
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
