"""
Analytico Backend V2 - FastAPI Server
Self-service data visualization with AI-powered chart generation
Features: Dataset persistence, aggregation, filtering, fuzzy validation
"""

import difflib
import json
import os
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
    title="Analytico API V2",
    description="AI-powered data visualization backend with aggregation",
    version="2.0.0"
)

# Configure CORS for frontend
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
# Dataset Storage (In-Memory with Expiry)
# ============================================================================

class DatasetInfo:
    def __init__(self, df: pd.DataFrame, filename: str):
        self.df = df
        self.filename = filename
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
    
    def touch(self):
        """Update last accessed time"""
        self.last_accessed = datetime.now()
    
    def is_expired(self, ttl_hours: int = 1) -> bool:
        """Check if dataset has expired"""
        return datetime.now() - self.last_accessed > timedelta(hours=ttl_hours)


# In-memory dataset store
DATASETS: dict[str, DatasetInfo] = {}
MAX_DATASETS = 10
DATASET_TTL_HOURS = 1


def cleanup_expired_datasets():
    """Remove expired datasets"""
    expired = [k for k, v in DATASETS.items() if v.is_expired(DATASET_TTL_HOURS)]
    for key in expired:
        del DATASETS[key]


def get_dataset(dataset_id: str) -> DatasetInfo:
    """Get dataset by ID, raises 404 if not found"""
    cleanup_expired_datasets()
    
    if dataset_id not in DATASETS:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_id}' not found or expired. Please re-upload your file."
        )
    
    dataset = DATASETS[dataset_id]
    dataset.touch()
    return dataset


# ============================================================================
# Pydantic Models
# ============================================================================

class ColumnSummary(BaseModel):
    name: str
    dtype: str
    is_numeric: bool
    is_datetime: bool
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


class FilterConfig(BaseModel):
    column: str
    values: Optional[list[Any]] = None  # For categorical
    min_val: Optional[Any] = None       # For numeric/date range
    max_val: Optional[Any] = None       # For numeric/date range


class AggregateRequest(BaseModel):
    dataset_id: str
    x_axis_key: str
    y_axis_keys: list[str]
    aggregation: str = "sum"  # sum, mean, count, min, max
    chart_type: str = "bar"
    filters: Optional[list[FilterConfig]] = None


class AggregateResponse(BaseModel):
    data: list[dict[str, Any]]
    x_axis_key: str
    y_axis_keys: list[str]
    chart_type: str
    title: str
    row_count: int


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


class ErrorResponse(BaseModel):
    error: str
    suggestions: Optional[list[str]] = None


# ============================================================================
# Helper Functions
# ============================================================================

def get_column_summary(df: pd.DataFrame, col: str) -> ColumnSummary:
    """Generate summary for a single column"""
    series = df[col]
    is_numeric = pd.api.types.is_numeric_dtype(series)
    is_datetime = pd.api.types.is_datetime64_any_dtype(series)
    
    summary = ColumnSummary(
        name=col,
        dtype=str(series.dtype),
        is_numeric=is_numeric,
        is_datetime=is_datetime,
        unique_count=int(series.nunique()),
        sample_values=series.dropna().head(5).tolist()
    )
    
    if is_numeric:
        summary.min_val = float(series.min()) if pd.notnull(series.min()) else None
        summary.max_val = float(series.max()) if pd.notnull(series.max()) else None
        summary.mean_val = float(series.mean()) if pd.notnull(series.mean()) else None
    
    return summary


def validate_columns(df: pd.DataFrame, columns: list[str]) -> tuple[bool, list[str], dict[str, list[str]]]:
    """
    Validate that columns exist in DataFrame.
    Returns: (is_valid, missing_columns, suggestions_map)
    """
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
    """Apply filters to DataFrame"""
    if not filters:
        return df
    
    filtered_df = df.copy()
    
    for f in filters:
        if f.column not in filtered_df.columns:
            continue
        
        # Categorical filter
        if f.values is not None:
            filtered_df = filtered_df[filtered_df[f.column].isin(f.values)]
        
        # Range filter
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
    """Aggregate data by grouping on x_axis_key"""
    agg_funcs = {
        "sum": "sum",
        "mean": "mean",
        "count": "count",
        "min": "min",
        "max": "max"
    }
    
    agg_func = agg_funcs.get(aggregation, "sum")
    
    # Group and aggregate
    grouped = df.groupby(x_axis_key, as_index=False)[y_axis_keys].agg(agg_func)
    
    # Sort by x-axis for better visualization
    grouped = grouped.sort_values(x_axis_key)
    
    # Limit to 100 rows max for performance
    if len(grouped) > 100:
        grouped = grouped.head(100)
    
    return grouped


def generate_chart_title(x_axis: str, y_axes: list[str], chart_type: str) -> str:
    """Generate a descriptive title for the chart"""
    y_str = " vs ".join(y_axes[:3])
    if len(y_axes) > 3:
        y_str += f" (+{len(y_axes) - 3} more)"
    return f"{y_str} by {x_axis}"


# ============================================================================
# System Prompt for OpenAI
# ============================================================================

SYSTEM_PROMPT = """You are a data visualization assistant. You receive a user question and dataset metadata.
Return a raw JSON object (no markdown) that maps the user's request to columns for a Recharts graph.

Rules:
1. xAxisKey MUST be a column from the provided columns list
2. yAxisKeys MUST only contain NUMERIC columns from the provided list
3. chartType must be: "bar", "line", "area", "pie", or "composed"
4. aggregation must be: "sum", "mean", "count", "min", or "max"
5. Generate a descriptive title

Example Output:
{"xAxisKey": "month", "yAxisKeys": ["revenue", "cost"], "chartType": "bar", "aggregation": "sum", "title": "Revenue vs Cost by Month"}

Return ONLY the JSON object."""


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "Analytico API V2",
        "active_datasets": len(DATASETS)
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file. The file is stored server-side and a dataset_id is returned.
    Only metadata (columns, summary) is sent to the frontend.
    """
    # Validate file type
    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")
    
    try:
        # Read CSV
        df = pd.read_csv(file.file)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="The CSV file is empty.")
        
        # Clean up expired datasets and check limit
        cleanup_expired_datasets()
        if len(DATASETS) >= MAX_DATASETS:
            # Remove oldest dataset
            oldest_key = min(DATASETS, key=lambda k: DATASETS[k].last_accessed)
            del DATASETS[oldest_key]
        
        # Generate dataset ID and store
        dataset_id = str(uuid.uuid4())
        DATASETS[dataset_id] = DatasetInfo(df, file.filename)
        
        # Generate column summaries
        columns = [get_column_summary(df, col) for col in df.columns]
        
        return UploadResponse(
            dataset_id=dataset_id,
            filename=file.filename,
            row_count=len(df),
            columns=columns
        )
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The CSV file is empty or malformed.")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/aggregate", response_model=AggregateResponse)
async def aggregate_data_endpoint(request: AggregateRequest):
    """
    Aggregate data from a stored dataset. Used by the Manual Chart Builder.
    """
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
        raise HTTPException(
            status_code=400,
            detail=f"Y-axis columns must be numeric. Non-numeric: {non_numeric}"
        )
    
    # Apply filters
    filtered_df = apply_filters(df, request.filters)
    
    if filtered_df.empty:
        raise HTTPException(status_code=400, detail="No data matches the applied filters.")
    
    # Aggregate
    aggregated = aggregate_data(
        filtered_df,
        request.x_axis_key,
        request.y_axis_keys,
        request.aggregation
    )
    
    # Convert to records
    data = aggregated.to_dict(orient='records')
    
    # Clean NaN values
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
        row_count=len(data)
    )


@app.post("/query", response_model=QueryResponse)
async def query_data(request: QueryRequest):
    """
    Generate a chart configuration from a natural language query.
    Uses OpenAI to interpret the request, then aggregates server-side.
    """
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key not configured."
        )
    
    dataset = get_dataset(request.dataset_id)
    df = dataset.df
    
    # Build column info for AI
    columns_info = []
    for col in df.columns:
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        col_type = "numeric" if is_numeric else "categorical"
        columns_info.append(f"{col} ({col_type})")
    
    try:
        # Call OpenAI
        user_message = f"""User Question: {request.user_prompt}

Available Columns:
{chr(10).join(columns_info)}

Dataset has {len(df)} rows."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=500
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
        
        # Validate columns exist
        all_columns = [chart_config["xAxisKey"]] + chart_config["yAxisKeys"]
        is_valid, missing, suggestions = validate_columns(df, all_columns)
        
        if not is_valid:
            error_parts = []
            for col in missing:
                if suggestions.get(col):
                    error_parts.append(f"Column '{col}' not found. Did you mean: {', '.join(suggestions[col])}?")
                else:
                    error_parts.append(f"Column '{col}' not found in dataset.")
            raise HTTPException(status_code=400, detail=" ".join(error_parts))
        
        # Apply filters
        filtered_df = apply_filters(df, request.filters)
        
        if filtered_df.empty:
            raise HTTPException(status_code=400, detail="No data matches the applied filters.")
        
        # Aggregate
        aggregation = chart_config.get("aggregation", "sum")
        aggregated = aggregate_data(
            filtered_df,
            chart_config["xAxisKey"],
            chart_config["yAxisKeys"],
            aggregation
        )
        
        # Convert to records
        data = aggregated.to_dict(orient='records')
        
        # Clean NaN values
        for row in data:
            for key in row:
                if pd.isna(row[key]):
                    row[key] = 0
        
        return QueryResponse(
            data=data,
            x_axis_key=chart_config["xAxisKey"],
            y_axis_keys=chart_config["yAxisKeys"],
            chart_type=chart_config.get("chartType", "bar"),
            title=chart_config.get("title", "Chart"),
            row_count=len(data)
        )
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/dataset/{dataset_id}/preview")
async def preview_dataset(dataset_id: str, limit: int = 100):
    """Preview first N rows of a dataset"""
    dataset = get_dataset(dataset_id)
    df = dataset.df.head(limit)
    
    return {
        "data": df.to_dict(orient='records'),
        "total_rows": len(dataset.df),
        "showing": len(df)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
