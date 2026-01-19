"""
Analytico Backend V4 - Zero Friction Enterprise Analytics
Smart ingestion, auto-analysis, robust aggregation, explainability
"""

import difflib
import json
import os
import re
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

app = FastAPI(
    title="Analytico API V4",
    description="Zero Friction Enterprise Analytics",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================================
# Module 1: Data Janitor (Smart Ingestion)
# ============================================================================

def normalize_header(header: str) -> str:
    """Convert header to snake_case"""
    clean = re.sub(r'[^\w\s]', '', header.strip())
    clean = re.sub(r'\s+', '_', clean)
    return clean.lower()


def detect_column_format(series: pd.Series, col_name: str) -> str:
    """Detect the display format for a column"""
    col_lower = col_name.lower()
    
    # Currency keywords
    if any(kw in col_lower for kw in ['revenue', 'sales', 'cost', 'price', 'amount', 'profit', 'income', 'salary', 'budget', 'spend']):
        return 'currency'
    
    # Percentage keywords
    if any(kw in col_lower for kw in ['rate', 'percent', 'pct', 'ratio', 'growth', 'margin', 'yield']):
        return 'percentage'
    
    # Check actual values for currency pattern
    if series.dtype == 'object':
        sample = series.dropna().head(50).astype(str)
        if len(sample) > 0:
            currency_count = sample.str.contains(r'^\$', regex=True).sum()
            if currency_count > len(sample) * 0.5:
                return 'currency'
            pct_count = sample.str.contains(r'%$', regex=True).sum()
            if pct_count > len(sample) * 0.5:
                return 'percentage'
    
    return 'number'


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, int], dict[str, str]]:
    """
    Clean DataFrame and extract metadata.
    Returns: (cleaned_df, cleaning_actions, missing_counts, column_formats)
    """
    cleaning_actions = []
    missing_counts = {}
    column_formats = {}
    
    # 1. Header Normalization
    original_cols = df.columns.tolist()
    new_cols = [normalize_header(col) for col in original_cols]
    renamed = [(o, n) for o, n in zip(original_cols, new_cols) if o != n]
    if renamed:
        cleaning_actions.append(f"Normalized {len(renamed)} column headers")
    df.columns = new_cols
    
    # 2. Type Repair and Format Detection
    for col in df.columns:
        if df[col].dtype == 'object':
            sample = df[col].dropna().head(100).astype(str)
            if len(sample) == 0:
                continue
            
            # Currency pattern
            currency_matches = sample.str.replace(',', '').str.match(r'^\$?[\d,]+\.?\d*$').sum()
            if currency_matches > len(sample) * 0.5:
                try:
                    df[col] = df[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    column_formats[col] = 'currency'
                    cleaning_actions.append(f"Converted '{col}' from currency text to numeric")
                except Exception:
                    pass
                continue
            
            # Percentage pattern
            pct_matches = sample.str.match(r'^\d+\.?\d*%$').sum()
            if pct_matches > len(sample) * 0.5:
                try:
                    df[col] = df[col].astype(str).str.replace('%', '', regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce') / 100
                    column_formats[col] = 'percentage'
                    cleaning_actions.append(f"Converted '{col}' from percentage text to decimal")
                except Exception:
                    pass
                continue
            
            # Try date parsing
            try:
                parsed = pd.to_datetime(df[col], errors='coerce')
                if parsed.notna().sum() > len(df) * 0.7:
                    df[col] = parsed
                    column_formats[col] = 'date'
                    cleaning_actions.append(f"Parsed '{col}' as date")
            except Exception:
                pass
    
    # Detect formats for remaining numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if col not in column_formats:
            column_formats[col] = detect_column_format(df[col], col)
    
    # 3. Auto-Imputation
    for col in numeric_cols:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            missing_counts[col] = int(missing_count)
            df[col] = df[col].fillna(0)
    
    if missing_counts:
        total = sum(missing_counts.values())
        cleaning_actions.append(f"Filled {total} missing numeric values")
    
    return df, cleaning_actions, missing_counts, column_formats


def auto_profile(df: pd.DataFrame, column_types: dict[str, str]) -> dict[str, Any]:
    """Generate executive summary / auto-profile"""
    profile = {
        'top_metrics': [],
        'time_range': None,
        'row_count': len(df),
        'column_count': len(df.columns)
    }
    
    # Only consider actual METRIC columns, not identifiers/temporal
    metric_cols = [col for col, stype in column_types.items() if stype == SemanticType.METRIC]
    if metric_cols:
        variances = {col: df[col].var() for col in metric_cols if col in df.columns and df[col].var() > 0}
        sorted_cols = sorted(variances.keys(), key=lambda x: variances[x], reverse=True)[:3]
        
        for col in sorted_cols:
            profile['top_metrics'].append({
                'name': col,
                'total': float(df[col].sum()),
                'average': float(df[col].mean()),
                'min': float(df[col].min()),
                'max': float(df[col].max())
            })
    
    # Find date range
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    if date_cols:
        date_col = date_cols[0]
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        if pd.notna(min_date) and pd.notna(max_date):
            profile['time_range'] = {
                'column': date_col,
                'start': min_date.isoformat(),
                'end': max_date.isoformat()
            }
    
    return profile


# ============================================================================
# Module 2: Zero Friction Intelligence (Auto-Analysis)
# ============================================================================

class SemanticType:
    METRIC = "metric"
    IDENTIFIER = "identifier"
    TEMPORAL = "temporal"
    CATEGORICAL = "categorical"


def detect_semantic_type(df: pd.DataFrame, col: str) -> str:
    """Detect semantic type of a column"""
    series = df[col]
    col_lower = col.lower()
    
    # Actual datetime columns
    if pd.api.types.is_datetime64_any_dtype(series):
        return SemanticType.TEMPORAL
    
    # For numeric columns
    if pd.api.types.is_numeric_dtype(series):
        unique_ratio = series.nunique() / max(len(series), 1)
        
        # Time-period keywords - treat as TEMPORAL even if numeric
        # (e.g., Period=2006.01, Year=2020)
        temporal_keywords = ['period', 'year', 'fy', 'fiscal', 'month', 'quarter']
        if any(kw in col_lower for kw in temporal_keywords):
            return SemanticType.TEMPORAL
        
        # Identifier detection
        id_keywords = ['id', 'key', 'code', 'num', 'number', 'zip', 'postal', 'phone', 'ssn', 'sku', 'reference']
        if unique_ratio > 0.9 and any(kw in col_lower for kw in id_keywords):
            return SemanticType.IDENTIFIER
        
        return SemanticType.METRIC
    
    # String columns - check for temporal keywords
    temporal_keywords = ['date', 'month', 'quarter', 'week', 'day']
    if any(kw in col_lower for kw in temporal_keywords):
        return SemanticType.TEMPORAL
    
    return SemanticType.CATEGORICAL


def generate_default_chart(df: pd.DataFrame, column_types: dict[str, str]) -> Optional[dict]:
    """Generate the best default chart configuration"""
    numeric_cols = [c for c, t in column_types.items() if t == SemanticType.METRIC]
    temporal_cols = [c for c, t in column_types.items() if t == SemanticType.TEMPORAL]
    categorical_cols = [c for c, t in column_types.items() if t == SemanticType.CATEGORICAL]
    
    if not numeric_cols:
        return None
    
    # Find highest variance numeric column
    variances = {col: df[col].var() for col in numeric_cols if col in df.columns and df[col].var() > 0}
    if not variances:
        return None
    
    best_metric = max(variances, key=variances.get)
    
    # Option 1: Date exists - Line chart over time
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    if date_cols:
        return {
            'x_axis_key': date_cols[0],
            'y_axis_keys': [best_metric],
            'chart_type': 'line',
            'aggregation': 'sum',
            'title': f'{best_metric.replace("_", " ").title()} Over Time',
            'reasoning': f"I created a line chart showing {best_metric} over time because date data is available."
        }
    
    # Option 2: Temporal column (year, month) - Line chart
    if temporal_cols:
        return {
            'x_axis_key': temporal_cols[0],
            'y_axis_keys': [best_metric],
            'chart_type': 'line',
            'aggregation': 'sum',
            'title': f'{best_metric.replace("_", " ").title()} by {temporal_cols[0].replace("_", " ").title()}',
            'reasoning': f"I created a line chart using {temporal_cols[0]} as the time axis."
        }
    
    # Option 3: Categorical with 2-25 unique values - Bar chart
    suitable_cats = []
    for col in categorical_cols:
        if col in df.columns:
            unique = df[col].nunique()
            if 2 <= unique <= 25:
                suitable_cats.append((col, unique))
    
    if suitable_cats:
        # Prefer columns with moderate cardinality (around 8-12)
        best_cat = sorted(suitable_cats, key=lambda x: abs(x[1] - 10))[0][0]
        return {
            'x_axis_key': best_cat,
            'y_axis_keys': [best_metric],
            'chart_type': 'bar',
            'aggregation': 'mean',
            'title': f'Average {best_metric.replace("_", " ").title()} by {best_cat.replace("_", " ").title()}',
            'reasoning': f"I created a bar chart comparing average {best_metric} across {best_cat} categories."
        }
    
    return None


def generate_dynamic_suggestions(df: pd.DataFrame, column_types: dict[str, str], column_formats: dict[str, str]) -> list[str]:
    """Generate 3 contextual suggestions based on actual column names"""
    suggestions = []
    
    numeric_cols = [c for c, t in column_types.items() if t == SemanticType.METRIC]
    temporal_cols = [c for c, t in column_types.items() if t == SemanticType.TEMPORAL]
    categorical_cols = [c for c, t in column_types.items() if t == SemanticType.CATEGORICAL]
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Format column names for display
    def fmt(col: str) -> str:
        return col.replace('_', ' ').title()
    
    # Suggestion 1: Time-based if available
    if date_cols and numeric_cols:
        suggestions.append(f"Show me {fmt(numeric_cols[0])} over time")
    elif temporal_cols and numeric_cols:
        suggestions.append(f"Show me {fmt(numeric_cols[0])} by {fmt(temporal_cols[0])}")
    
    # Suggestion 2: Comparison by category
    if categorical_cols and numeric_cols:
        cat = categorical_cols[0]
        metric = numeric_cols[0]
        suggestions.append(f"Compare {fmt(metric)} by {fmt(cat)}")
    
    # Suggestion 3: Multi-metric comparison
    if len(numeric_cols) >= 2:
        suggestions.append(f"Compare {fmt(numeric_cols[0])} vs {fmt(numeric_cols[1])}")
    elif numeric_cols:
        suggestions.append(f"What's the average {fmt(numeric_cols[0])}?")
    
    # Ensure we have 3
    while len(suggestions) < 3:
        suggestions.append("Show me a summary of the data")
    
    return suggestions[:3]


# ============================================================================
# Module 3: Smart Aggregation
# ============================================================================

def smart_group_top_n(df: pd.DataFrame, x_col: str, y_cols: list[str], n: int = 19) -> pd.DataFrame:
    """Group by x_col, keep top N by sum of first y_col, combine rest as 'Others'"""
    if df[x_col].nunique() <= n + 1:
        return df
    
    # Calculate totals for ranking
    primary_y = y_cols[0]
    totals = df.groupby(x_col)[primary_y].sum().sort_values(ascending=False)
    
    top_categories = totals.head(n).index.tolist()
    
    # Split into top and others
    df_copy = df.copy()
    df_copy['_x_grouped'] = df_copy[x_col].apply(lambda x: x if x in top_categories else 'Others')
    
    # Aggregate
    grouped = df_copy.groupby('_x_grouped', as_index=False)[y_cols].sum()
    grouped = grouped.rename(columns={'_x_grouped': x_col})
    
    # Sort: top categories first, Others last
    def sort_key(val):
        if val == 'Others':
            return (1, '')
        return (0, str(val))
    
    grouped['_sort'] = grouped[x_col].apply(sort_key)
    grouped = grouped.sort_values('_sort').drop('_sort', axis=1)
    
    return grouped


def smart_resample_dates(df: pd.DataFrame, date_col: str, y_cols: list[str]) -> tuple[pd.DataFrame, str]:
    """Auto-resample date data if too many points"""
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
    
    grouped = df_copy.groupby('_period', as_index=False)[y_cols].sum()
    grouped = grouped.rename(columns={'_period': new_col})
    
    return grouped, new_col


def enforce_semantic_rules(aggregation: str, y_axis_keys: list[str], column_types: dict[str, str]) -> tuple[str, list[str]]:
    """Override aggregation for identifier columns"""
    warnings = []
    identifier_cols = [col for col in y_axis_keys if column_types.get(col) == SemanticType.IDENTIFIER]
    
    if identifier_cols and aggregation in ['sum', 'mean']:
        warnings.append(f"Changed to COUNT for identifier columns: {identifier_cols}")
        return 'count', warnings
    
    return aggregation, warnings


# ============================================================================
# Dataset Storage
# ============================================================================

class DatasetInfo:
    def __init__(self, df: pd.DataFrame, filename: str, cleaning_actions: list[str],
                 missing_counts: dict[str, int], column_types: dict[str, str],
                 column_formats: dict[str, str], profile: dict, default_chart: Optional[dict],
                 suggestions: list[str]):
        self.df = df
        self.filename = filename
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.cleaning_actions = cleaning_actions
        self.missing_counts = missing_counts
        self.column_types = column_types
        self.column_formats = column_formats
        self.profile = profile
        self.default_chart = default_chart
        self.suggestions = suggestions
    
    def touch(self):
        self.last_accessed = datetime.now()
    
    def is_expired(self, ttl_hours: int = 1) -> bool:
        return datetime.now() - self.last_accessed > timedelta(hours=ttl_hours)


DATASETS: dict[str, DatasetInfo] = {}
MAX_DATASETS = 10


def cleanup_expired():
    expired = [k for k, v in DATASETS.items() if v.is_expired()]
    for k in expired:
        del DATASETS[k]


def get_dataset(dataset_id: str) -> DatasetInfo:
    cleanup_expired()
    if dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found or expired. Please re-upload.")
    ds = DATASETS[dataset_id]
    ds.touch()
    return ds


# ============================================================================
# Pydantic Models
# ============================================================================

class MetricSummary(BaseModel):
    name: str
    total: float
    average: float
    min: float
    max: float


class TimeRange(BaseModel):
    column: str
    start: str
    end: str


class DataProfile(BaseModel):
    top_metrics: list[MetricSummary]
    time_range: Optional[TimeRange]
    row_count: int
    column_count: int


class DataHealth(BaseModel):
    missing_values: dict[str, int]
    cleaning_actions: list[str]
    quality_score: float


class DefaultChart(BaseModel):
    x_axis_key: str
    y_axis_keys: list[str]
    chart_type: str
    aggregation: str
    title: str
    reasoning: str


class ColumnSummary(BaseModel):
    name: str
    dtype: str
    is_numeric: bool
    is_datetime: bool
    semantic_type: str
    format: str
    unique_count: int
    sample_values: list[Any]


class UploadResponse(BaseModel):
    dataset_id: str
    filename: str
    row_count: int
    columns: list[ColumnSummary]
    column_formats: dict[str, str]
    data_health: DataHealth
    profile: DataProfile
    default_chart: Optional[DefaultChart]
    suggestions: list[str]


class FilterConfig(BaseModel):
    column: str
    values: Optional[list[Any]] = None
    min_val: Optional[Any] = None
    max_val: Optional[Any] = None


class AggregateRequest(BaseModel):
    dataset_id: str
    x_axis_key: str
    y_axis_keys: list[str]
    aggregation: str = "sum"
    chart_type: str = "bar"
    filters: Optional[list[FilterConfig]] = None


class ChartResponse(BaseModel):
    data: list[dict[str, Any]]
    x_axis_key: str
    y_axis_keys: list[str]
    chart_type: str
    title: str
    x_axis_label: Optional[str] = None
    y_axis_label: Optional[str] = None
    row_count: int
    reasoning: Optional[str] = None
    warnings: Optional[list[str]] = None
    applied_filters: Optional[list[str]] = None


class QueryRequest(BaseModel):
    dataset_id: str
    user_prompt: str
    filters: Optional[list[FilterConfig]] = None


# ============================================================================
# Helpers
# ============================================================================

def get_column_summary(df: pd.DataFrame, col: str, sem_type: str, fmt: str) -> ColumnSummary:
    series = df[col]
    return ColumnSummary(
        name=col,
        dtype=str(series.dtype),
        is_numeric=pd.api.types.is_numeric_dtype(series),
        is_datetime=pd.api.types.is_datetime64_any_dtype(series),
        semantic_type=sem_type,
        format=fmt,
        unique_count=int(series.nunique()),
        sample_values=series.dropna().head(5).tolist()
    )


def validate_columns(df: pd.DataFrame, cols: list[str]) -> tuple[bool, list[str], dict[str, list[str]]]:
    df_cols = df.columns.tolist()
    missing = [c for c in cols if c not in df_cols]
    if not missing:
        return True, [], {}
    suggestions = {c: difflib.get_close_matches(c, df_cols, n=3, cutoff=0.4) for c in missing}
    return False, missing, suggestions


def apply_filters(df: pd.DataFrame, filters: Optional[list[FilterConfig]]) -> tuple[pd.DataFrame, list[str]]:
    if not filters:
        return df, []
    
    applied = []
    filtered = df.copy()
    
    for f in filters:
        if f.column not in filtered.columns:
            continue
        if f.values:
            filtered = filtered[filtered[f.column].isin(f.values)]
            applied.append(f"{f.column}: {', '.join(str(v) for v in f.values[:3])}")
        if f.min_val is not None:
            filtered = filtered[filtered[f.column] >= f.min_val]
            applied.append(f"{f.column} >= {f.min_val}")
        if f.max_val is not None:
            filtered = filtered[filtered[f.column] <= f.max_val]
            applied.append(f"{f.column} <= {f.max_val}")
    
    return filtered, applied


def aggregate_data(df: pd.DataFrame, x_key: str, y_keys: list[str], agg: str) -> pd.DataFrame:
    agg_map = {"sum": "sum", "mean": "mean", "count": "count", "min": "min", "max": "max"}
    grouped = df.groupby(x_key, as_index=False)[y_keys].agg(agg_map.get(agg, "sum"))
    # Convert x_key to string to avoid mixed type sorting issues
    grouped[x_key] = grouped[x_key].astype(str)
    return grouped.sort_values(x_key).head(100)


def df_to_markdown(df: pd.DataFrame, n: int = 5) -> str:
    sample = df.head(n)
    header = "| " + " | ".join(str(c) for c in sample.columns) + " |"
    sep = "| " + " | ".join("---" for _ in sample.columns) + " |"
    rows = ["| " + " | ".join(str(v)[:25] for v in row) + " |" for _, row in sample.iterrows()]
    return "\n".join([header, sep] + rows)


# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT = """You are a data visualization assistant. Given a user question and dataset metadata, return JSON with:

1. xAxisKey: Column for X-axis
2. yAxisKeys: Numeric columns for Y-axis
3. chartType: "bar", "line", "area", "pie", or "composed"
4. aggregation: "sum", "mean", "count", "min", "max"
5. title: Chart title
6. xAxisLabel: Human-readable X-axis label (e.g., "Department")
7. yAxisLabel: Human-readable Y-axis label (e.g., "Total Revenue ($)")
8. reasoning: 1-2 sentences explaining your choice
9. calculated_field: Optional {name, expression} for derived metrics

Rules:
- IDENTIFIER columns: use COUNT only
- TEMPORAL columns: prefer as X-axis
- Format labels based on column format (currency → include $)

Return ONLY raw JSON, no markdown."""


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/")
async def root():
    return {"status": "ok", "version": "4.0.0", "datasets": len(DATASETS)}


@app.post("/upload", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")
    
    try:
        df = pd.read_csv(file.file)
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV is empty.")
        
        # Module 1: Clean data
        df, cleaning_actions, missing_counts, column_formats = clean_dataframe(df)
        
        # Detect semantic types FIRST (needed for profile)
        column_types = {col: detect_semantic_type(df, col) for col in df.columns}
        
        # Now profile with semantic types
        profile = auto_profile(df, column_types)
        
        # Module 2: Generate default chart and suggestions
        default_chart = generate_default_chart(df, column_types)
        suggestions = generate_dynamic_suggestions(df, column_types, column_formats)
        
        # Quality score
        total_cells = len(df) * len(df.columns)
        missing_total = sum(missing_counts.values())
        quality = max(0, 100 - (missing_total / max(total_cells, 1) * 100))
        
        # Store
        cleanup_expired()
        if len(DATASETS) >= MAX_DATASETS:
            oldest = min(DATASETS, key=lambda k: DATASETS[k].last_accessed)
            del DATASETS[oldest]
        
        dataset_id = str(uuid.uuid4())
        DATASETS[dataset_id] = DatasetInfo(
            df, file.filename, cleaning_actions, missing_counts,
            column_types, column_formats, profile, default_chart, suggestions
        )
        
        columns = [
            get_column_summary(df, col, column_types[col], column_formats.get(col, 'number'))
            for col in df.columns
        ]
        
        return UploadResponse(
            dataset_id=dataset_id,
            filename=file.filename,
            row_count=len(df),
            columns=columns,
            column_formats=column_formats,
            data_health=DataHealth(
                missing_values=missing_counts,
                cleaning_actions=cleaning_actions,
                quality_score=round(quality, 1)
            ),
            profile=DataProfile(
                top_metrics=[MetricSummary(**m) for m in profile['top_metrics']],
                time_range=TimeRange(**profile['time_range']) if profile['time_range'] else None,
                row_count=profile['row_count'],
                column_count=profile['column_count']
            ),
            default_chart=DefaultChart(**default_chart) if default_chart else None,
            suggestions=suggestions
        )
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Empty CSV.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/aggregate", response_model=ChartResponse)
async def aggregate_endpoint(request: AggregateRequest):
    ds = get_dataset(request.dataset_id)
    df = ds.df
    
    # Validate
    all_cols = [request.x_axis_key] + request.y_axis_keys
    valid, missing, suggestions = validate_columns(df, all_cols)
    if not valid:
        msg = "; ".join(f"'{c}' not found, try: {suggestions.get(c, [])}" for c in missing)
        raise HTTPException(status_code=400, detail=msg)
    
    # Enforce semantic rules
    agg, warnings = enforce_semantic_rules(request.aggregation, request.y_axis_keys, ds.column_types)
    
    # Apply filters
    filtered, applied_filters = apply_filters(df, request.filters)
    if filtered.empty:
        raise HTTPException(status_code=400, detail="No data matches filters.")
    
    # Module 3: Smart aggregation
    # Check if date column needs resampling
    if pd.api.types.is_datetime64_any_dtype(df.get(request.x_axis_key)):
        filtered, new_x = smart_resample_dates(filtered, request.x_axis_key, request.y_axis_keys)
        x_key = new_x
    else:
        x_key = request.x_axis_key
        # Top N + Others for high cardinality
        if x_key in filtered.columns and filtered[x_key].nunique() > 20:
            filtered = smart_group_top_n(filtered, x_key, request.y_axis_keys)
    
    # Aggregate
    result = aggregate_data(filtered, x_key, request.y_axis_keys, agg)
    
    data = result.to_dict(orient='records')
    for row in data:
        for k in row:
            if pd.isna(row[k]):
                row[k] = 0
    
    return ChartResponse(
        data=data,
        x_axis_key=x_key,
        y_axis_keys=request.y_axis_keys,
        chart_type=request.chart_type,
        title=f"{', '.join(request.y_axis_keys)} by {x_key}".replace('_', ' ').title(),
        row_count=len(data),
        warnings=warnings or None,
        applied_filters=applied_filters or None
    )


@app.post("/query", response_model=ChartResponse)
async def query_endpoint(request: QueryRequest):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OpenAI API key not configured.")
    
    ds = get_dataset(request.dataset_id)
    df = ds.df
    
    # Build context
    cols_info = [f"- {c} ({ds.column_types.get(c, '?').upper()}, format: {ds.column_formats.get(c, 'number')})" for c in df.columns]
    sample = df_to_markdown(df, 5)
    
    try:
        user_msg = f"""Question: {request.user_prompt}

Columns:
{chr(10).join(cols_info)}

Sample:
{sample}

Rows: {len(df)}"""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.3,
            max_tokens=800
        )
        
        content = resp.choices[0].message.content or ""
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:-1])
        
        config = json.loads(cleaned)
        
        # Validate
        all_cols = [config["xAxisKey"]] + config["yAxisKeys"]
        valid, missing, sugg = validate_columns(df, all_cols)
        if not valid:
            msg = "; ".join(f"'{c}' not found, try: {sugg.get(c, [])}" for c in missing)
            raise HTTPException(status_code=400, detail=msg)
        
        # Apply filters
        filtered, applied = apply_filters(df, request.filters)
        if filtered.empty:
            raise HTTPException(status_code=400, detail="No data after filters.")
        
        # Enforce semantic rules
        agg = config.get("aggregation", "sum")
        agg, warnings = enforce_semantic_rules(agg, config["yAxisKeys"], ds.column_types)
        
        # Smart aggregation
        x_key = config["xAxisKey"]
        y_keys = config["yAxisKeys"]
        
        if pd.api.types.is_datetime64_any_dtype(filtered.get(x_key)):
            filtered, x_key = smart_resample_dates(filtered, x_key, y_keys)
        elif x_key in filtered.columns and filtered[x_key].nunique() > 20:
            filtered = smart_group_top_n(filtered, x_key, y_keys)
        
        result = aggregate_data(filtered, x_key, y_keys, agg)
        
        data = result.to_dict(orient='records')
        for row in data:
            for k in row:
                if pd.isna(row[k]):
                    row[k] = 0
        
        return ChartResponse(
            data=data,
            x_axis_key=x_key,
            y_axis_keys=y_keys,
            chart_type=config.get("chartType", "bar"),
            title=config.get("title", "Chart"),
            x_axis_label=config.get("xAxisLabel"),
            y_axis_label=config.get("yAxisLabel"),
            row_count=len(data),
            reasoning=config.get("reasoning", "AI-generated configuration."),
            warnings=warnings or None,
            applied_filters=applied or None
        )
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Parse error: {e}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
