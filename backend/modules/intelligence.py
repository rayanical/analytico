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
    
    return profile


def generate_dynamic_suggestions(df: pd.DataFrame, column_types: dict[str, str], column_formats: dict[str, str]) -> list[str]:
    """Generate 3 contextual suggestions based on actual column names"""
    suggestions = []
    
    metric_cols = [c for c, t in column_types.items() if t == SemanticType.METRIC]
    categorical_cols = [c for c, t in column_types.items() if t == SemanticType.CATEGORICAL]
    temporal_cols = [c for c, t in column_types.items() if t == SemanticType.TEMPORAL]
    
    def fmt(col: str) -> str:
        return col.replace('_', ' ')
    
    # Suggestion 1: Breakdown by category
    if categorical_cols and metric_cols:
        suggestions.append(f"Show {fmt(metric_cols[0])} by {fmt(categorical_cols[0])}")
    
    # Suggestion 2: Time trend
    if temporal_cols and metric_cols:
        suggestions.append(f"How has {fmt(metric_cols[0])} changed over time?")
    
    # Suggestion 3: Top N analysis
    if categorical_cols and metric_cols:
        suggestions.append(f"What are the top 10 {fmt(categorical_cols[0])}s by {fmt(metric_cols[0])}?")
    
    # Fallback suggestions
    if len(metric_cols) >= 2:
        suggestions.append(f"Compare {fmt(metric_cols[0])} and {fmt(metric_cols[1])}")
    
    if len(suggestions) < 3 and metric_cols:
        suggestions.append(f"What's the average {fmt(metric_cols[0])}?")
    
    # Ensure we have 3
    while len(suggestions) < 3:
        suggestions.append("Show me a summary of the data")
    
    return suggestions[:3]
