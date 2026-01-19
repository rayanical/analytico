"""
Analytico Backend V3 - FastAPI Server
Self-service data visualization with AI-powered chart generation
Features: Smart ingestion, semantic guardrails, health scorecard, explainability
"""

import difflib
import json
import os
import re
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Analytico API V3",
    description="AI-powered data visualization with robustness pillars",
    version="3.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================================
# Pillar 1: Data Janitor (Smart Ingestion)
# ============================================================================

def normalize_header(header: str) -> str:
    """Convert header to snake_case"""
    # Strip whitespace and special chars
    clean = re.sub(r'[^\w\s]', '', header.strip())
    # Replace spaces with underscores
    clean = re.sub(r'\s+', '_', clean)
    # Convert to lowercase
    return clean.lower()


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """
    Clean and repair a DataFrame.
    Returns: (cleaned_df, cleaning_actions, missing_counts)
    """
    cleaning_actions = []
    missing_counts = {}
    
    # 1. Header Normalization
    original_cols = df.columns.tolist()
    new_cols = [normalize_header(col) for col in original_cols]
    
    # Track renamed columns
    renamed = [(orig, new) for orig, new in zip(original_cols, new_cols) if orig != new]
    if renamed:
        cleaning_actions.append(f"Normalized {len(renamed)} column headers to snake_case")
    
    df.columns = new_cols
    
    # 2. Type Repair for string columns
    for col in df.columns:
        if df[col].dtype == 'object':
            # Sample non-null values
            sample = df[col].dropna().head(100).astype(str)
            if len(sample) == 0:
                continue
            
            # Check for currency pattern ($1,234.56)
            currency_pattern = r'^\$?[\d,]+\.?\d*$'
            currency_matches = sample.str.replace(',', '').str.match(currency_pattern).sum()
            
            if currency_matches > len(sample) * 0.5:
                try:
                    # Remove $ and , then convert
                    df[col] = df[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    cleaning_actions.append(f"Converted '{col}' from currency text to numeric")
                except Exception:
                    pass
                continue
            
            # Check for percentage pattern (15%, 0.15)
            pct_pattern = r'^\d+\.?\d*%$'
            pct_matches = sample.str.match(pct_pattern).sum()
            
            if pct_matches > len(sample) * 0.5:
                try:
                    df[col] = df[col].astype(str).str.replace('%', '', regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce') / 100
                    cleaning_actions.append(f"Converted '{col}' from percentage text to decimal")
                except Exception:
                    pass
    
    # 3. Auto-Imputation for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            missing_counts[col] = int(missing_count)
            df[col] = df[col].fillna(0)
    
    if missing_counts:
        total = sum(missing_counts.values())
        cleaning_actions.append(f"Filled {total} missing numeric values with 0")
    
    return df, cleaning_actions, missing_counts


# ============================================================================
# Pillar 2: Semantic Guardrails
# ============================================================================

class SemanticType:
    METRIC = "metric"         # Normal numeric for aggregation
    IDENTIFIER = "identifier" # IDs, keys, codes - only COUNT
    TEMPORAL = "temporal"     # Year, month - prefer as X-axis
    CATEGORICAL = "categorical"


def detect_semantic_type(df: pd.DataFrame, col: str) -> str:
    """Detect the semantic type of a column"""
    series = df[col]
    col_lower = col.lower()
    
    # Check for temporal patterns
    temporal_keywords = ['year', 'month', 'quarter', 'fy', 'fiscal', 'period', 'week', 'day']
    if any(kw in col_lower for kw in temporal_keywords):
        return SemanticType.TEMPORAL
    
    # Check for identifiers (numeric with high uniqueness)
    if pd.api.types.is_numeric_dtype(series):
        unique_ratio = series.nunique() / max(len(series), 1)
        id_keywords = ['id', 'key', 'code', 'num', 'number', 'zip', 'postal', 'phone', 'ssn']
        
        if unique_ratio > 0.9 and any(kw in col_lower for kw in id_keywords):
            return SemanticType.IDENTIFIER
        
        return SemanticType.METRIC
    
    return SemanticType.CATEGORICAL


def enforce_semantic_rules(
    aggregation: str,
    y_axis_keys: list[str],
    column_types: dict[str, str]
) -> tuple[str, list[str]]:
    """
    Enforce semantic rules on aggregation.
    Returns: (adjusted_aggregation, warnings)
    """
    warnings = []
    
    # Check if any y-axis column is an identifier
    identifier_cols = [col for col in y_axis_keys if column_types.get(col) == SemanticType.IDENTIFIER]
    
    if identifier_cols and aggregation in ['sum', 'mean']:
        warnings.append(f"Changed aggregation from '{aggregation}' to 'count' for identifier columns: {identifier_cols}")
        return 'count', warnings
    
    return aggregation, warnings


# ============================================================================
# Pillar 4: Explainability Helpers
# ============================================================================

def df_to_markdown_sample(df: pd.DataFrame, n: int = 5) -> str:
    """Convert first N rows of DataFrame to Markdown table"""
    sample = df.head(n)
    
    # Build header
    header = "| " + " | ".join(str(col) for col in sample.columns) + " |"
    separator = "| " + " | ".join("---" for _ in sample.columns) + " |"
    
    # Build rows
    rows = []
    for _, row in sample.iterrows():
        row_str = "| " + " | ".join(str(v)[:30] for v in row.values) + " |"
        rows.append(row_str)
    
    return "\n".join([header, separator] + rows)


def compute_calculated_field(df: pd.DataFrame, expression: str, field_name: str) -> pd.DataFrame:
    """
    Compute a calculated field from an expression.
    Example: expression = "revenue - cost", field_name = "profit"
    """
    try:
        # Parse simple arithmetic expressions
        # Supported: addition, subtraction, multiplication, division
        df = df.copy()
        
        # Replace column names with df['col'] syntax
        expr = expression
        for col in df.columns:
            # Match whole word only
            expr = re.sub(rf'\b{re.escape(col)}\b', f"df['{col}']", expr)
        
        # Evaluate safely
        df[field_name] = eval(expr)
        return df
    except Exception as e:
        # If calculation fails, return original df
        return df


# ============================================================================
# Dataset Storage
# ============================================================================

class DatasetInfo:
    def __init__(self, df: pd.DataFrame, filename: str, 
                 cleaning_actions: list[str], missing_counts: dict[str, int],
                 column_types: dict[str, str]):
        self.df = df
        self.filename = filename
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.cleaning_actions = cleaning_actions
        self.missing_counts = missing_counts
        self.column_types = column_types
    
    def touch(self):
        self.last_accessed = datetime.now()
    
    def is_expired(self, ttl_hours: int = 1) -> bool:
        return datetime.now() - self.last_accessed > timedelta(hours=ttl_hours)


DATASETS: dict[str, DatasetInfo] = {}
MAX_DATASETS = 10
DATASET_TTL_HOURS = 1


def cleanup_expired_datasets():
    expired = [k for k, v in DATASETS.items() if v.is_expired(DATASET_TTL_HOURS)]
    for key in expired:
        del DATASETS[key]


def get_dataset(dataset_id: str) -> DatasetInfo:
    cleanup_expired_datasets()
    
    if dataset_id not in DATASETS:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_id}' not found or expired. Please re-upload."
        )
    
    dataset = DATASETS[dataset_id]
    dataset.touch()
    return dataset


# ============================================================================
# Pydantic Models
# ============================================================================

class DataHealth(BaseModel):
    missing_values: dict[str, int]
    cleaning_actions: list[str]
    quality_score: float  # 0-100


class ColumnSummary(BaseModel):
    name: str
    dtype: str
    is_numeric: bool
    is_datetime: bool
    semantic_type: str
    unique_count: int
    sample_values: list[Any]
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean_val: Optional[float] = None


class UploadResponse(BaseModel):
    dataset_id: str
    filename: str
    row_count: int
    columns: list[ColumnSummary]
    data_health: DataHealth


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


class AggregateResponse(BaseModel):
    data: list[dict[str, Any]]
    x_axis_key: str
    y_axis_keys: list[str]
    chart_type: str
    title: str
    row_count: int
    reasoning: Optional[str] = None
    warnings: Optional[list[str]] = None


class QueryRequest(BaseModel):
    dataset_id: str
    user_prompt: str
    filters: Optional[list[FilterConfig]] = None


class QueryResponse(BaseModel):
    data: list[dict[str, Any]]
    x_axis_key: str
    y_axis_keys: list[str]
    chart_type: str
    title: str
    row_count: int
    reasoning: str
    warnings: Optional[list[str]] = None


# ============================================================================
# Helper Functions
# ============================================================================

def get_column_summary(df: pd.DataFrame, col: str, semantic_type: str) -> ColumnSummary:
    series = df[col]
    is_numeric = pd.api.types.is_numeric_dtype(series)
    is_datetime = pd.api.types.is_datetime64_any_dtype(series)
    
    summary = ColumnSummary(
        name=col,
        dtype=str(series.dtype),
        is_numeric=is_numeric,
        is_datetime=is_datetime,
        semantic_type=semantic_type,
        unique_count=int(series.nunique()),
        sample_values=series.dropna().head(5).tolist()
    )
    
    if is_numeric:
        summary.min_val = float(series.min()) if pd.notnull(series.min()) else None
        summary.max_val = float(series.max()) if pd.notnull(series.max()) else None
        summary.mean_val = float(series.mean()) if pd.notnull(series.mean()) else None
    
    return summary


def validate_columns(df: pd.DataFrame, columns: list[str]) -> tuple[bool, list[str], dict[str, list[str]]]:
    df_columns = df.columns.tolist()
    missing = [col for col in columns if col not in df_columns]
    
    if not missing:
        return True, [], {}
    
    suggestions = {}
    for col in missing:
        matches = difflib.get_close_matches(col, df_columns, n=3, cutoff=0.4)
        suggestions[col] = matches
    
    return False, missing, suggestions


def apply_filters(df: pd.DataFrame, filters: Optional[list[FilterConfig]]) -> pd.DataFrame:
    if not filters:
        return df
    
    filtered_df = df.copy()
    
    for f in filters:
        if f.column not in filtered_df.columns:
            continue
        
        if f.values is not None:
            filtered_df = filtered_df[filtered_df[f.column].isin(f.values)]
        
        if f.min_val is not None:
            filtered_df = filtered_df[filtered_df[f.column] >= f.min_val]
        if f.max_val is not None:
            filtered_df = filtered_df[filtered_df[f.column] <= f.max_val]
    
    return filtered_df


def aggregate_data(
    df: pd.DataFrame,
    x_axis_key: str,
    y_axis_keys: list[str],
    aggregation: str = "sum"
) -> pd.DataFrame:
    agg_funcs = {"sum": "sum", "mean": "mean", "count": "count", "min": "min", "max": "max"}
    agg_func = agg_funcs.get(aggregation, "sum")
    
    grouped = df.groupby(x_axis_key, as_index=False)[y_axis_keys].agg(agg_func)
    grouped = grouped.sort_values(x_axis_key)
    
    if len(grouped) > 100:
        grouped = grouped.head(100)
    
    return grouped


def generate_chart_title(x_axis: str, y_axes: list[str], chart_type: str) -> str:
    y_str = " vs ".join(y_axes[:3])
    if len(y_axes) > 3:
        y_str += f" (+{len(y_axes) - 3} more)"
    return f"{y_str} by {x_axis}"


# ============================================================================
# System Prompt for OpenAI (Updated for Explainability)
# ============================================================================

SYSTEM_PROMPT = """You are a data visualization assistant. You receive a user question, dataset columns with semantic types, and a sample of the data.

Return a raw JSON object (no markdown) with:
1. xAxisKey: Column for X-axis (prefer TEMPORAL or CATEGORICAL columns)
2. yAxisKeys: List of METRIC columns for Y-axis
3. chartType: "bar", "line", "area", "pie", or "composed"
4. aggregation: "sum", "mean", "count", "min", or "max"
5. title: Descriptive chart title
6. reasoning: 1-2 sentences explaining WHY you chose this configuration
7. calculated_field (optional): Object with {name, expression} if user wants derived metrics (e.g., profit = revenue - cost)

Semantic Type Rules:
- IDENTIFIER columns (IDs, codes): Use COUNT, never SUM/AVG
- TEMPORAL columns (year, month): Prefer as X-axis
- METRIC columns: Use for Y-axis with appropriate aggregation

Example Output:
{
  "xAxisKey": "department",
  "yAxisKeys": ["salary", "bonus"],
  "chartType": "bar",
  "aggregation": "sum",
  "title": "Total Compensation by Department",
  "reasoning": "I selected a Bar Chart because 'department' is categorical. I used 'sum' for 'salary' and 'bonus' to show total spend per department.",
  "calculated_field": null
}

Return ONLY the JSON object."""


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Analytico API V3",
        "active_datasets": len(DATASETS)
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)):
    """Upload and clean a CSV file"""
    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")
    
    try:
        df = pd.read_csv(file.file)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="The CSV file is empty.")
        
        # Pillar 1: Clean the DataFrame
        df, cleaning_actions, missing_counts = clean_dataframe(df)
        
        # Pillar 2: Detect semantic types
        column_types = {col: detect_semantic_type(df, col) for col in df.columns}
        
        # Calculate quality score
        total_cells = len(df) * len(df.columns)
        missing_total = sum(missing_counts.values())
        quality_score = max(0, 100 - (missing_total / max(total_cells, 1) * 100))
        
        # Cleanup and store
        cleanup_expired_datasets()
        if len(DATASETS) >= MAX_DATASETS:
            oldest_key = min(DATASETS, key=lambda k: DATASETS[k].last_accessed)
            del DATASETS[oldest_key]
        
        dataset_id = str(uuid.uuid4())
        DATASETS[dataset_id] = DatasetInfo(
            df, file.filename, cleaning_actions, missing_counts, column_types
        )
        
        # Build response
        columns = [
            get_column_summary(df, col, column_types[col]) 
            for col in df.columns
        ]
        
        return UploadResponse(
            dataset_id=dataset_id,
            filename=file.filename,
            row_count=len(df),
            columns=columns,
            data_health=DataHealth(
                missing_values=missing_counts,
                cleaning_actions=cleaning_actions,
                quality_score=round(quality_score, 1)
            )
        )
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Empty or malformed CSV.")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Parse error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/aggregate", response_model=AggregateResponse)
async def aggregate_data_endpoint(request: AggregateRequest):
    """Aggregate data with semantic guardrails"""
    dataset = get_dataset(request.dataset_id)
    df = dataset.df
    
    # Validate columns
    all_columns = [request.x_axis_key] + request.y_axis_keys
    is_valid, missing, suggestions = validate_columns(df, all_columns)
    
    if not is_valid:
        error_parts = []
        for col in missing:
            if suggestions.get(col):
                error_parts.append(f"'{col}' not found. Did you mean: {', '.join(suggestions[col])}?")
            else:
                error_parts.append(f"'{col}' not found.")
        raise HTTPException(status_code=400, detail=" ".join(error_parts))
    
    # Validate y-axis columns are numeric
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    non_numeric = [col for col in request.y_axis_keys if col not in numeric_cols]
    if non_numeric:
        raise HTTPException(status_code=400, detail=f"Y-axis columns must be numeric: {non_numeric}")
    
    # Pillar 2: Enforce semantic rules
    aggregation, warnings = enforce_semantic_rules(
        request.aggregation,
        request.y_axis_keys,
        dataset.column_types
    )
    
    # Apply filters
    filtered_df = apply_filters(df, request.filters)
    
    if filtered_df.empty:
        raise HTTPException(status_code=400, detail="No data matches filters.")
    
    # Aggregate
    aggregated = aggregate_data(filtered_df, request.x_axis_key, request.y_axis_keys, aggregation)
    
    # Convert to records
    data = aggregated.to_dict(orient='records')
    for row in data:
        for key in row:
            if pd.isna(row[key]):
                row[key] = 0
    
    return AggregateResponse(
        data=data,
        x_axis_key=request.x_axis_key,
        y_axis_keys=request.y_axis_keys,
        chart_type=request.chart_type,
        title=generate_chart_title(request.x_axis_key, request.y_axis_keys, request.chart_type),
        row_count=len(data),
        warnings=warnings if warnings else None
    )


@app.post("/query", response_model=QueryResponse)
async def query_data(request: QueryRequest):
    """AI-powered chart generation with context and explainability"""
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OpenAI API key not configured.")
    
    dataset = get_dataset(request.dataset_id)
    df = dataset.df
    
    # Build column info with semantic types
    columns_info = []
    for col in df.columns:
        sem_type = dataset.column_types.get(col, "unknown")
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        type_label = f"{sem_type.upper()}, {'numeric' if is_numeric else 'text'}"
        columns_info.append(f"- {col} ({type_label})")
    
    # Pillar 4: Context injection with sample data
    sample_table = df_to_markdown_sample(df, 5)
    
    try:
        user_message = f"""User Question: {request.user_prompt}

Available Columns:
{chr(10).join(columns_info)}

Sample Data (first 5 rows):
{sample_table}

Dataset has {len(df)} rows."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=800
        )
        
        content = response.choices[0].message.content
        if not content:
            raise HTTPException(status_code=500, detail="OpenAI returned empty response.")
        
        # Parse JSON
        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1])
        
        chart_config = json.loads(cleaned)
        
        # Validate columns
        all_columns = [chart_config["xAxisKey"]] + chart_config["yAxisKeys"]
        is_valid, missing, suggestions = validate_columns(df, all_columns)
        
        if not is_valid:
            error_parts = []
            for col in missing:
                if suggestions.get(col):
                    error_parts.append(f"Column '{col}' not found. Did you mean: {', '.join(suggestions[col])}?")
                else:
                    error_parts.append(f"Column '{col}' not found.")
            raise HTTPException(status_code=400, detail=" ".join(error_parts))
        
        # Handle calculated fields
        working_df = df.copy()
        y_keys = chart_config["yAxisKeys"]
        
        if chart_config.get("calculated_field"):
            calc = chart_config["calculated_field"]
            if isinstance(calc, dict) and "name" in calc and "expression" in calc:
                working_df = compute_calculated_field(working_df, calc["expression"], calc["name"])
                if calc["name"] in working_df.columns:
                    y_keys = y_keys + [calc["name"]]
        
        # Apply filters
        filtered_df = apply_filters(working_df, request.filters)
        
        if filtered_df.empty:
            raise HTTPException(status_code=400, detail="No data matches filters.")
        
        # Enforce semantic rules
        aggregation = chart_config.get("aggregation", "sum")
        aggregation, warnings = enforce_semantic_rules(aggregation, y_keys, dataset.column_types)
        
        # Aggregate
        aggregated = aggregate_data(filtered_df, chart_config["xAxisKey"], y_keys, aggregation)
        
        # Convert to records
        data = aggregated.to_dict(orient='records')
        for row in data:
            for key in row:
                if pd.isna(row[key]):
                    row[key] = 0
        
        return QueryResponse(
            data=data,
            x_axis_key=chart_config["xAxisKey"],
            y_axis_keys=y_keys,
            chart_type=chart_config.get("chartType", "bar"),
            title=chart_config.get("title", "Chart"),
            row_count=len(data),
            reasoning=chart_config.get("reasoning", "AI-generated chart configuration."),
            warnings=warnings if warnings else None
        )
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/dataset/{dataset_id}/preview")
async def preview_dataset(dataset_id: str, limit: int = 100):
    dataset = get_dataset(dataset_id)
    df = dataset.df.head(limit)
    
    return {
        "data": df.to_dict(orient='records'),
        "total_rows": len(dataset.df),
        "showing": len(df),
        "data_health": {
            "cleaning_actions": dataset.cleaning_actions,
            "missing_values": dataset.missing_counts
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
