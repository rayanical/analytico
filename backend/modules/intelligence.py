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
    col_lower = col.lower()
    
    # Check for datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return SemanticType.TEMPORAL
    
    # Check for date-like column names
    if any(kw in col_lower for kw in ['date', 'time', 'year', 'month', 'day', 'timestamp']):
        return SemanticType.TEMPORAL
    
    # Check for identifier patterns
    if any(kw in col_lower for kw in ['id', 'code', 'key', 'name', 'email', 'phone', 'address']):
        return SemanticType.IDENTIFIER
    
    # Numeric columns
    if pd.api.types.is_numeric_dtype(series):
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
    temporal_cols = [c for c, t in column_types.items() if t == SemanticType.TEMPORAL]
    categorical_cols = [c for c, t in column_types.items() if t == SemanticType.CATEGORICAL]
    metric_cols = [c for c, t in column_types.items() if t == SemanticType.METRIC]
    
    if not metric_cols:
        return None
    
    # Best case: temporal x-axis with metric y-axis
    if temporal_cols:
        return {
            "x_axis_key": temporal_cols[0],
            "y_axis_keys": metric_cols[:2],
            "chart_type": "line",
            "aggregation": "sum",
            "title": f"{', '.join(metric_cols[:2])} Over Time".replace('_', ' ').title(),
            "analysis": f"Tracking {metric_cols[0]} over time reveals historical trends and seasonality. This data helps identify growth patterns and potential cyclical behavior impacting {temporal_cols[0]}."
        }
    
    # Second best: categorical x-axis with metric y-axis
    if categorical_cols:
        # Pick categorical with reasonable cardinality
        best_cat = min(categorical_cols, key=lambda c: abs(df[c].nunique() - 10))
        return {
            "x_axis_key": best_cat,
            "y_axis_keys": metric_cols[:2],
            "chart_type": "bar",
            "aggregation": "sum",
            "title": f"{', '.join(metric_cols[:2])} by {best_cat}".replace('_', ' ').title(),
            "analysis": f"Comparing {metric_cols[0]} across {best_cat} segments highlights performance variances. This breakdown identifies which {best_cat} categories are driving the most value."
        }
    
    # Fallback: first two metrics as composed chart
    if len(metric_cols) >= 2:
        return {
            "x_axis_key": metric_cols[0],
            "y_axis_keys": metric_cols[1:3],
            "chart_type": "composed",
            "aggregation": "sum",
            "title": f"Correlation: {metric_cols[0]} vs {metric_cols[1]}".replace('_', ' ').title(),
            "analysis": f"Analyzing the relationship between {metric_cols[0]} and {metric_cols[1]}. This correlation view helps determine if an increase in one metric drives changes in the other."
        }
    
    return None


def auto_profile(df: pd.DataFrame, column_types: dict[str, str]) -> dict:
    """Generate executive summary / auto-profile"""
    # Sample frame reserved for heavy categorical profiling operations.
    # Keep deterministic for stable outputs across runs.
    sample_df = df.sample(n=min(100000, len(df)), random_state=42) if len(df) > 0 else df

    profile = {
        "top_metrics": [],
        "time_range": None,
        "row_count": len(df),
        "column_count": len(df.columns)
    }
    
    # Find metric columns for summary
    metric_cols = [c for c, t in column_types.items() if t == SemanticType.METRIC]
    
    for col in metric_cols[:3]:  # Top 3 metrics
        series = df[col].dropna()
        if len(series) == 0:
            continue
        profile["top_metrics"].append({
            "name": col,
            "total": float(series.sum()),
            "average": float(series.mean()),
            "min": float(series.min()),
            "max": float(series.max())
        })
    
    # Find temporal columns for range
    temporal_cols = [c for c, t in column_types.items() if t == SemanticType.TEMPORAL]
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

    # Hook for future heavy categorical profile metrics: use sample_df for
    # value_counts/nunique-style operations, while preserving current output shape.
    _ = sample_df
    
    return profile


def generate_dynamic_suggestions(df: pd.DataFrame, column_types: dict[str, str], column_formats: dict[str, str]) -> list[str]:
    """Generate concise, intent-diverse, dataset-aware example prompts."""
    suggestions: list[str] = []
    
    metric_cols = [c for c, t in column_types.items() if t == SemanticType.METRIC]
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
