"""
Analytico Backend - Aggregation Module
Smart grouping, date resampling, and semantic rule enforcement
"""

from typing import Optional
import pandas as pd

from .intelligence import SemanticType


def smart_group_top_n(
    df: pd.DataFrame, 
    x_col: str, 
    y_cols: list[str], 
    agg: str = "sum", 
    n: int = 19, 
    group_others: bool = True
) -> pd.DataFrame:
    """Group by x_col, keep top N by sum of first y_col, combine rest as 'Others'"""
    if df.empty:
        return df
        
    # If grouping is disabled, just return top N
    if not group_others:
        first_y = y_cols[0] if y_cols else None
        if first_y and first_y in df.columns:
            # Aggregate first to avoid duplicates
            grouped = df.groupby(x_col, as_index=False)[y_cols].agg(agg)
            return grouped.sort_values(first_y, ascending=False).head(n)
        return df.head(n)

    first_y = y_cols[0] if y_cols else None
    
    # If using "Others", we want n-1 actual items + 1 "Others"
    target_items = n - 1 if n > 1 else 1

    if df[x_col].nunique() <= n:
        return df
    
    # Calculate totals for ranking (always use sum for ranking)
    primary_y = y_cols[0]
    totals = df.groupby(x_col)[primary_y].sum().sort_values(ascending=False)
    
    # Keep top (n-1) to leave room for "Others"
    top_categories = totals.head(target_items).index.tolist()
    
    # Split into top and others
    df_copy = df.copy()
    df_copy['_x_grouped'] = df_copy[x_col].apply(lambda x: x if x in top_categories else 'Others')
    
    # Aggregate using the requested aggregation type
    agg_map = {"sum": "sum", "mean": "mean", "count": "count", "min": "min", "max": "max"}
    grouped = df_copy.groupby('_x_grouped', as_index=False)[y_cols].agg(agg_map.get(agg, "sum"))
    grouped = grouped.rename(columns={'_x_grouped': x_col})
    
    # Sort: top categories first, Others last
    def sort_key(val):
        if val == 'Others':
            return (1, '')
        return (0, str(val))
    
    grouped['_sort'] = grouped[x_col].apply(sort_key)
    grouped = grouped.sort_values('_sort').drop('_sort', axis=1)
    
    return grouped


def smart_resample_dates(
    df: pd.DataFrame, 
    date_col: str, 
    y_cols: list[str], 
    agg: str = "sum"
) -> tuple[pd.DataFrame, str]:
    """Resample datetime data into appropriate periods (year/month/week)"""
    if date_col not in df.columns:
        return df, date_col
    
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        return df, date_col
    
    unique_dates = df[date_col].nunique()
    if unique_dates <= 100:
        return df, date_col
    
    # Calculate date range
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    range_days = (max_date - min_date).days
    
    df_copy = df.copy()
    
    # Resample based on range
    if range_days > 730:  # > 2 years
        df_copy['_period'] = df_copy[date_col].dt.to_period('Y').astype(str)
        new_col = f"{date_col}_year"
    elif range_days > 180:  # > 6 months
        df_copy['_period'] = df_copy[date_col].dt.to_period('M').astype(str)
        new_col = f"{date_col}_month"
    else:
        df_copy['_period'] = df_copy[date_col].dt.to_period('W').astype(str)
        new_col = f"{date_col}_week"
    
    # Aggregate using the requested aggregation type
    agg_map = {"sum": "sum", "mean": "mean", "count": "count", "min": "min", "max": "max"}
    grouped = df_copy.groupby('_period', as_index=False)[y_cols].agg(agg_map.get(agg, "sum"))
    grouped = grouped.rename(columns={'_period': new_col})
    
    return grouped, new_col


def enforce_semantic_rules(
    aggregation: str, 
    y_axis_keys: list[str], 
    column_types: dict[str, str]
) -> tuple[str, list[str], Optional[str]]:
    """Override aggregation for identifier or categorical columns. Returns (agg, warnings, y_axis_label)."""
    warnings = []
    y_axis_label = None
    
    identifier_cols = [col for col in y_axis_keys if column_types.get(col) == SemanticType.IDENTIFIER]
    categorical_cols = [col for col in y_axis_keys if column_types.get(col) == SemanticType.CATEGORICAL]
    
    # Identifier columns should use COUNT
    if identifier_cols and aggregation in ['sum', 'mean']:
        warnings.append(f"Changed to COUNT for identifier columns: {identifier_cols}")
        return 'count', warnings, "Count of Records"
    
    # Categorical columns should use COUNT (non-numeric fix)
    if categorical_cols and aggregation in ['sum', 'mean']:
        warnings.append(f"Switched to COUNT for categorical column: {categorical_cols[0]} (non-numeric data)")
        return 'count', warnings, "Count of Records"
    
    # If aggregation is already COUNT, set appropriate label
    if aggregation == 'count':
        y_axis_label = "Count of Records"
    
    return aggregation, warnings, y_axis_label


def aggregate_data(df: pd.DataFrame, x_key: str, y_keys: list[str], agg: str) -> pd.DataFrame:
    """Aggregate data by x_key, applying aggregation to y_keys"""
    agg_map = {"sum": "sum", "mean": "mean", "count": "count", "min": "min", "max": "max"}
    grouped = df.groupby(x_key, as_index=False)[y_keys].agg(agg_map.get(agg, "sum"))
    # Convert x_key to string to avoid mixed type sorting issues
    grouped[x_key] = grouped[x_key].astype(str)
    return grouped.sort_values(x_key).head(100)
