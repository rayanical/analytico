"""
Analytico Backend V5 - Zero Friction Enterprise Analytics
Modular architecture with smart ingestion, auto-analysis, and agentic features
"""

import ast
import contextlib
import difflib
import io
import json
import os
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any, Optional, TextIO

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import ValidationError

# Import models
from models import (
    MetricSummary, TimeRange, DataProfile, DataHealth, DefaultChart,
    ColumnSummary, UploadResponse, FilterConfig, AggregateRequest,
    ChartResponse, QueryRequest, DrillDownRequest,
)

# Import storage
from storage import DatasetInfo, get_dataset, store_dataset, cleanup_expired

# Import modules
from modules import (
    clean_dataframe,
    SemanticType,
    detect_semantic_type,
    generate_default_chart,
    auto_profile,
    generate_dynamic_suggestions,
    smart_group_top_n,
    smart_resample_dates,
    enforce_semantic_rules,
    aggregate_data,
)

load_dotenv()

app = FastAPI(
    title="Analytico API V5",
    description="Zero Friction Enterprise Analytics - Modular Architecture",
    version="5.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DEMO_DATASET_PATH = Path(__file__).resolve().parent / "datasets" / "2021_Green_Taxi_Trip_Data_20260221.csv"
ALLOWED_FILTER_OPERATORS = {"eq", "gt", "lt", "gte", "lte", "contains"}
MAX_CHART_POINTS = 500


# ============================================================================
# Helper Functions
# ============================================================================

def get_column_summary(df: pd.DataFrame, col: str, sem_type: str, fmt: str) -> ColumnSummary:
    """Generate column summary for API response"""
    series = df[col]
    unique_vals = series.dropna().unique()
    sample_vals = sorted([str(v) for v in unique_vals[:20]], key=str.lower)
    return ColumnSummary(
        name=col,
        dtype=str(series.dtype),
        is_numeric=pd.api.types.is_numeric_dtype(series),
        is_datetime=pd.api.types.is_datetime64_any_dtype(series),
        semantic_type=sem_type,
        format=fmt,
        unique_count=int(series.nunique()),
        sample_values=sample_vals
    )


def validate_columns(df: pd.DataFrame, cols: list[str]) -> tuple[bool, list[str], dict[str, list[str]]]:
    """Validate that columns exist, suggest alternatives if not"""
    df_cols = df.columns.tolist()
    missing = [c for c in cols if c not in df_cols]
    if not missing:
        return True, [], {}
    suggestions = {c: difflib.get_close_matches(c, df_cols, n=3, cutoff=0.4) for c in missing}
    return False, missing, suggestions


def _safe_empty_chart_response(message: str, title: str = "Clarification Needed") -> ChartResponse:
    return ChartResponse(
        data=[],
        x_axis_key="",
        y_axis_keys=[],
        chart_type="empty",
        title=title,
        aggregation=None,
        row_count=0,
        y_axis_label="",
        analysis=message,
        answer=None
    )


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
        if pd.api.types.is_datetime64_any_dtype(col):
            cmp_value = pd.to_datetime(f.value, errors="coerce")
            if pd.isna(cmp_value):
                applied.append(f"{f.column}: skipped invalid datetime value '{f.value}'")
                return filtered
            mask = col == cmp_value
        elif pd.api.types.is_numeric_dtype(col):
            try:
                cmp_value = float(f.value)
            except (TypeError, ValueError):
                applied.append(f"{f.column}: skipped invalid numeric value '{f.value}'")
                return filtered
            num_col = pd.to_numeric(col, errors="coerce")
            mask = num_col == cmp_value
        else:
            mask = col.astype(str) == str(f.value)
        applied.append(f"{f.column} == {f.value}")
    elif op == "contains":
        mask = col.astype(str).str.contains(str(f.value), case=False, na=False, regex=False)
        applied.append(f"{f.column} contains '{f.value}'")
    elif op in {"gt", "lt", "gte", "lte"}:
        # Datetime-aware comparisons for NLP date filters.
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


def _resolve_effective_limit(requested_limit: Optional[int], default_limit: int) -> tuple[int, Optional[str]]:
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
    """Apply filter configurations to dataframe"""
    if not filters:
        return df, []
    
    applied = []
    filtered = df.copy()
    
    for f in filters:
        if f.column not in filtered.columns:
            continue

        filtered = _apply_operator_filter(filtered, f, applied)

        # Legacy compatibility: values behaves like IN-list equality filter.
        if f.values:
            str_values = [str(v) for v in f.values]
            filtered = filtered[filtered[f.column].astype(str).isin(str_values)]
            applied.append(f"{f.column}: {', '.join(str(v) for v in f.values[:3])}")

        # Legacy compatibility: min/max map to gte/lte semantics.
        if f.min_val is not None:
            filtered = _apply_operator_filter(
                filtered,
                FilterConfig(column=f.column, operator="gte", value=f.min_val),
                applied
            )
        if f.max_val is not None:
            filtered = _apply_operator_filter(
                filtered,
                FilterConfig(column=f.column, operator="lte", value=f.max_val),
                applied
            )
    
    return filtered, applied


def df_to_markdown(df: pd.DataFrame, n: int = 5) -> str:
    """Convert dataframe head to markdown table"""
    sample = df.head(n)
    header = "| " + " | ".join(str(c) for c in sample.columns) + " |"
    sep = "| " + " | ".join("---" for _ in sample.columns) + " |"
    rows = ["| " + " | ".join(str(v)[:25] for v in row) + " |" for _, row in sample.iterrows()]
    return "\n".join([header, sep] + rows)


def read_csv_fast(source: str | Path | TextIO) -> pd.DataFrame:
    """Read CSV using pyarrow when available, with safe fallback."""
    try:
        return pd.read_csv(source, engine="pyarrow")
    except Exception:
        if hasattr(source, "seek"):
            source.seek(0)
            return pd.read_csv(source, low_memory=False)
        return pd.read_csv(source, low_memory=False)


def secure_exec(code: str, df: pd.DataFrame) -> Any:
    """Execute generated Python code securely using AST whitelisting"""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"Syntax Error: {e}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Security: Imports are not allowed.")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in ['open', 'eval', 'exec', 'compile']:
                raise ValueError(f"Security: Function '{node.func.id}' is banned.")
            if isinstance(node.func, ast.Attribute) and node.func.attr == '__builtins__':
                raise ValueError("Security: Access to __builtins__ is banned.")

    local_scope = {'df': df.copy(), 'pd': pd, 'np': np, 'result': None}
    
    capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(capture):
            exec(code, {'__builtins__': {}}, local_scope)
    except Exception as e:
        return f"Runtime Error: {e}\n{traceback.format_exc()}"
    
    output = capture.getvalue().strip()
    if local_scope.get('result') is not None:
        return local_scope['result']
    return output or "No output."


def print_pipeline_timing(endpoint: str, durations: dict[str, float]) -> None:
    """Print formatted execution timings for ingestion pipeline phases."""
    print(f"\n=== {endpoint} Pipeline Timing ===")
    print(f"CSV Ingestion: {durations['csv_ingestion']:.2f}s")
    print(f"Data Cleaning: {durations['data_cleaning']:.2f}s")
    print(f"Data Profiling: {durations['data_profiling']:.2f}s")
    print(f"LLM Summary Generation: {durations['llm_summary']:.2f}s")
    print(f"Total Pipeline: {durations['total']:.2f}s")
    print("=" * (len(endpoint) + 20))


# ============================================================================
# System Prompt for LLM
# ============================================================================

SYSTEM_PROMPT = """You are a data visualization assistant. Given a user question and dataset metadata, return JSON with:

1. xAxisKey: Column for X-axis
2. yAxisKeys: Numeric columns for Y-axis
3. chartType: "bar", "line", "area", "pie", or "composed"
4. aggregation: "sum", "mean", "count", "min", "max"
5. title: Chart title
6. xAxisLabel: Human-readable X-axis label (e.g., "Department")
7. yAxisLabel: Human-readable Y-axis label (e.g., "Total Revenue ($)")
8. analysis: A 2-sentence business insight. First sentence summarizes what is shown, second sentence highlights the key trend or outlier.
9. calculated_field: Optional {name, expression} for derived metrics

Rules:
- IDENTIFIER columns: use COUNT only
- TEMPORAL columns: prefer as X-axis
- Format labels based on column format (currency → include $)

ROUTING DECISION:
- Primary goal: produce a JSON chart configuration that visualizes the answer.
- If the user asks for basic aggregations (sum, average/mean, min, max, count), grouped by category or time, you MUST return JSON config and MUST NOT use the Python tool.
- Infer the best chart type even if the user asks for a single number; default to a chartable aggregation JSON response whenever possible.
- Use the Python tool only as a fallback for advanced statistics or multi-step transformations that cannot be represented as a standard aggregated chart.

Return ONLY raw JSON, no markdown."""


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/")
async def root():
    return {"status": "ok", "version": "5.0.0", "architecture": "modular"}


@app.get("/validate/{dataset_id}")
async def validate_dataset(dataset_id: str):
    """Check if a dataset ID is still valid (exists in memory)"""
    cleanup_expired()
    from storage import DATASETS
    if dataset_id in DATASETS:
        ds = DATASETS[dataset_id]
        ds.touch()
        return {"valid": True, "filename": ds.filename}
    return {"valid": False}


@app.post("/upload", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")
    
    try:
        endpoint_start = perf_counter()
        t0 = perf_counter()
        df = read_csv_fast(file.file)
        t1 = perf_counter()
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV is empty.")
        
        # Clean data using data_janitor module
        t2 = perf_counter()
        df, cleaning_actions, missing_counts, col_formats, llm_col_types = clean_dataframe(df)
        t3 = perf_counter()
        
        # Use LLM semantic types when available, fallback to heuristic detection per column.
        col_types = {
            col: llm_col_types.get(col) or detect_semantic_type(df, col)
            for col in df.columns
        }
        
        # Generate profile
        t4 = perf_counter()
        profile = auto_profile(df, col_types)
        t5 = perf_counter()
        
        # Quality score
        total_cells = len(df) * len(df.columns)
        missing_total = sum(missing_counts.values())
        quality = max(0, 100 - (missing_total / max(total_cells, 1) * 100))
        
        # Generate Business Summary
        summary = None
        t6 = perf_counter()
        try:
            summary_prompt = f"""Summarize this dataset in 1-2 business sentences. 
            Filename: {file.filename}
            Columns: {', '.join(df.columns[:20])}
            Sample Data: {df.head(3).to_string()}
            """
            
            summary_resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a data analyst. Be concise. No preamble."},
                    {"role": "user", "content": summary_prompt}
                ],
                max_tokens=100,
                temperature=0.3
            )
            summary = summary_resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"Summary generation failed: {e}")
        t7 = perf_counter()

        # Generate Default Chart
        default_chart = generate_default_chart(df, col_types)
        
        # Generate Suggestions
        suggestions = generate_dynamic_suggestions(df, col_types, col_formats)
        
        # Store dataset
        ds_info = DatasetInfo(
            df=df, 
            filename=file.filename, 
            cleaning_actions=cleaning_actions, 
            missing_counts=missing_counts,
            column_types=col_types,
            column_formats=col_formats,
            profile=profile,
            default_chart=default_chart,
            suggestions=suggestions,
            summary=summary
        )
        store_dataset(ds_info)

        t8 = perf_counter()
        durations = {
            "csv_ingestion": t1 - t0,
            "data_cleaning": t3 - t2,
            "data_profiling": t5 - t4,
            "llm_summary": t7 - t6,
            "total": t8 - endpoint_start,
        }
        print_pipeline_timing("/upload", durations)

        return UploadResponse(
            dataset_id=ds_info.id,
            filename=ds_info.filename,
            row_count=len(df),
            columns=[
                get_column_summary(df, c, col_types.get(c, "unknown"), col_formats.get(c, "general"))
                for c in df.columns
            ],
            column_formats=col_formats,
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
            suggestions=suggestions,
            summary=summary
        )
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Empty CSV.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load-demo", response_model=UploadResponse)
async def load_demo_dataset():
    filename = "taxi_1m_rows.csv"
    if not DEMO_DATASET_PATH.exists():
        raise HTTPException(status_code=500, detail="Demo dataset file not found.")

    try:
        endpoint_start = perf_counter()
        t0 = perf_counter()
        df = read_csv_fast(DEMO_DATASET_PATH)
        t1 = perf_counter()
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV is empty.")
        
        # Clean data using data_janitor module
        t2 = perf_counter()
        df, cleaning_actions, missing_counts, col_formats, llm_col_types = clean_dataframe(df)
        t3 = perf_counter()
        
        # Use LLM semantic types when available, fallback to heuristic detection per column.
        col_types = {
            col: llm_col_types.get(col) or detect_semantic_type(df, col)
            for col in df.columns
        }
        
        # Generate profile
        t4 = perf_counter()
        profile = auto_profile(df, col_types)
        t5 = perf_counter()
        
        # Quality score
        total_cells = len(df) * len(df.columns)
        missing_total = sum(missing_counts.values())
        quality = max(0, 100 - (missing_total / max(total_cells, 1) * 100))
        
        # Generate Business Summary
        summary = None
        t6 = perf_counter()
        try:
            summary_prompt = f"""Summarize this dataset in 1-2 business sentences. 
            Filename: {filename}
            Columns: {', '.join(df.columns[:20])}
            Sample Data: {df.head(3).to_string()}
            """
            
            summary_resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a data analyst. Be concise. No preamble."},
                    {"role": "user", "content": summary_prompt}
                ],
                max_tokens=100,
                temperature=0.3
            )
            summary = summary_resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"Summary generation failed: {e}")
        t7 = perf_counter()

        # Generate Default Chart
        default_chart = generate_default_chart(df, col_types)
        
        # Generate Suggestions
        suggestions = generate_dynamic_suggestions(df, col_types, col_formats)
        
        # Store dataset
        ds_info = DatasetInfo(
            df=df, 
            filename=filename, 
            cleaning_actions=cleaning_actions, 
            missing_counts=missing_counts,
            column_types=col_types,
            column_formats=col_formats,
            profile=profile,
            default_chart=default_chart,
            suggestions=suggestions,
            summary=summary
        )
        store_dataset(ds_info)

        t8 = perf_counter()
        durations = {
            "csv_ingestion": t1 - t0,
            "data_cleaning": t3 - t2,
            "data_profiling": t5 - t4,
            "llm_summary": t7 - t6,
            "total": t8 - endpoint_start,
        }
        print_pipeline_timing("/load-demo", durations)

        return UploadResponse(
            dataset_id=ds_info.id,
            filename=ds_info.filename,
            row_count=len(df),
            columns=[
                get_column_summary(df, c, col_types.get(c, "unknown"), col_formats.get(c, "general"))
                for c in df.columns
            ],
            column_formats=col_formats,
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
            suggestions=suggestions,
            summary=summary
        )
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Empty CSV.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/aggregate", response_model=ChartResponse)
async def aggregate_endpoint(request: AggregateRequest):
    if request.include_analysis and not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OpenAI API key not configured.")

    ds = get_dataset(request.dataset_id)
    df = ds.df
    
    # Validate columns
    all_cols = [request.x_axis_key] + request.y_axis_keys
    valid, missing, suggestions = validate_columns(df, all_cols)
    if not valid:
        msg = "; ".join(f"'{c}' not found, try: {suggestions.get(c, [])}" for c in missing)
        raise HTTPException(status_code=400, detail=msg)
    
    # Enforce semantic rules
    agg, warnings, y_axis_label = enforce_semantic_rules(request.aggregation, request.y_axis_keys, ds.column_types)
    
    # Apply filters
    filtered, applied_filters = apply_filters(df, request.filters)
    if filtered.empty:
        raise HTTPException(status_code=400, detail="No data matches filters.")
    
    # Smart aggregation for dates
    if pd.api.types.is_datetime64_any_dtype(df.get(request.x_axis_key)):
        filtered, new_x = smart_resample_dates(filtered, request.x_axis_key, request.y_axis_keys, agg)
        x_key = new_x
    else:
        x_key = request.x_axis_key
    
    # Smart grouping (Top N + Others) with hard cap for rendering performance
    limit, cap_warning = _resolve_effective_limit(request.limit, default_limit=50)
    group_others = request.group_others if request.group_others is not None else True

    if request.x_axis_key in filtered.columns:
        unique_cnt = filtered[request.x_axis_key].nunique()
        if unique_cnt > limit:
            filtered = smart_group_top_n(
                filtered, 
                request.x_axis_key, 
                request.y_axis_keys, 
                request.aggregation, 
                n=limit, 
                group_others=group_others
            )
    
    # Aggregate
    result, was_capped = aggregate_data(filtered, x_key, request.y_axis_keys, agg, limit=limit)
    
    # Smart sorting defaults based on semantic type
    # If user specified sort_by, use it. Otherwise, apply smart default.
    sort_by = request.sort_by
    
    if sort_by is None:
        # Find the x_axis column to determine its semantic type
        x_col_meta = next((c for c in ds.columns if c['name'] == request.x_axis_key), None)
        if x_col_meta:
            semantic_type = x_col_meta.get('semantic_type', 'categorical')
            # Smart default: numeric/temporal → sort by label, categorical → sort by value
            if semantic_type in ['metric', 'temporal']:
                sort_by = 'label'
            else:
                sort_by = 'value'
        else:
            sort_by = 'value'  # Fallback default
    
    # Apply sorting
    if sort_by == "value" and request.y_axis_keys:
        result = result.sort_values(request.y_axis_keys[0], ascending=False)
    elif sort_by == "label":
        result = result.sort_values(x_key, ascending=True)
    
    analysis = None
    if request.include_analysis:
        try:
            summary_md = df_to_markdown(result.head(20), n=20)
            prompt = f"""Analyze this data summary (2 sentences). 1. Context, 2. Insight.

Data:
{summary_md}"""

            analysis_resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a data analyst. Be concise. Only two sentences."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=120,
                temperature=0.3
            )
            analysis = analysis_resp.choices[0].message.content.strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    data = result.to_dict(orient='records')
    for row in data:
        for k in row:
            if pd.isna(row[k]):
                row[k] = 0
    
    final_warnings = list(warnings or [])
    if cap_warning and (was_capped or request.limit == 0 or (request.limit is not None and request.limit > MAX_CHART_POINTS)):
        final_warnings.append(cap_warning)

    return ChartResponse(
        data=data,
        x_axis_key=x_key,
        y_axis_keys=request.y_axis_keys,
        chart_type=request.chart_type,
        title=f"{', '.join(request.y_axis_keys)} by {x_key}".replace('_', ' ').title(),
        aggregation=agg,
        y_axis_label=y_axis_label,
        row_count=len(data),
        analysis=analysis,
        warnings=final_warnings or None,
        applied_filters=applied_filters or None
    )


@app.post("/query", response_model=ChartResponse)
async def query_endpoint(request: QueryRequest):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OpenAI API key not configured.")
    
    ds = get_dataset(request.dataset_id)
    df = ds.df
    valid_columns = df.columns.tolist()
    preferred_metric_columns = [c for c, t in ds.column_types.items() if t == SemanticType.METRIC][:8]
    preferred_temporal_columns = [c for c, t in ds.column_types.items() if t == SemanticType.TEMPORAL][:5]
    
    # Build context
    cols_info = [f"- {c} ({ds.column_types.get(c, '?').upper()}, format: {ds.column_formats.get(c, 'number')})" for c in df.columns]
    sample = df_to_markdown(df, 5)
    
    try:
        user_msg = f"""Question: {request.user_prompt}

Columns:
{chr(10).join(cols_info)}

Sample:
{sample}

Rows: {len(df)}

STRICT COLUMN RULE:
You may ONLY use these exact column names for xAxisKey, yAxisKeys, and filters:
{json.dumps(valid_columns)}

FILTER SCHEMA RULE:
filters MUST be an array of objects with:
- column: one of the exact valid column names
- operator: one of eq, gt, lt, gte, lte, contains
- value: a primitive value

COLUMN PREFERENCE RULE:
Prefer these metric columns for yAxisKeys when relevant:
{json.dumps(preferred_metric_columns)}
Prefer these temporal columns for xAxisKey when trend/time intent is asked:
{json.dumps(preferred_temporal_columns)}

PYTHON TOOL RULE:
If you choose to use generate_python_analysis, you must apply all requested filtering directly in your pandas code.
Do not rely on the JSON filters array in that path."""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.3,
            max_tokens=800,
            tools=[{
                "type": "function",
                "function": {
                    "name": "generate_python_analysis",
                    "description": "Only use this fallback tool if the user's request requires advanced statistics (e.g., correlation, standard deviation, forecasting) or multi-step DataFrame transformations that cannot be represented by a standard aggregated chart. Do NOT use this tool for simple averages, sums, or counts.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The python code to execute. Assign final answer to variable 'result' or print it."
                            },
                            "explanation": {
                                "type": "string",
                                "description": "Explanation of the analysis."
                            }
                        },
                        "required": ["code", "explanation"]
                    }
                }
            }]
        )
        
        tool_calls = resp.choices[0].message.tool_calls
        
        if tool_calls:
            # Python Execution Path
            call = tool_calls[0]
            args = json.loads(call.function.arguments)
            code = args['code']
            code = code.strip()
            if code.startswith("```"):
                code_lines = code.splitlines()
                if code_lines:
                    code_lines = code_lines[1:]
                if code_lines and code_lines[-1].strip().startswith("```"):
                    code_lines = code_lines[:-1]
                code = "\n".join(code_lines).strip()
            explanation = args.get('explanation', '')
            
            try:
                filtered, _ = apply_filters(df, request.filters)
                exec_result = secure_exec(code, filtered)
                
                # Handle DataFrame results - convert to chartable format
                if isinstance(exec_result, pd.DataFrame) and len(exec_result) > 0:
                    # Extract column names dynamically
                    cols = exec_result.columns.tolist()
                    x_key = cols[0] if len(cols) > 0 else ""
                    y_keys = cols[1:] if len(cols) > 1 else []
                    
                    # Convert to records
                    data = exec_result.head(100).to_dict(orient='records')
                    for row in data:
                        for k in row:
                            if pd.isna(row[k]):
                                row[k] = 0
                    
                    return ChartResponse(
                        data=data,
                        x_axis_key=x_key,
                        y_axis_keys=y_keys if y_keys else [x_key],
                        chart_type="bar",
                        title=f"Analysis: {request.user_prompt[:50]}...",
                        aggregation=None,
                        row_count=len(data),
                        y_axis_label="",
                        analysis=explanation,
                        answer=None
                    )
                elif isinstance(exec_result, pd.Series):
                    # Convert Series to DataFrame for charting
                    result_df = exec_result.reset_index()
                    result_df.columns = ['category', 'value']
                    data = result_df.head(50).to_dict(orient='records')
                    
                    return ChartResponse(
                        data=data,
                        x_axis_key="category",
                        y_axis_keys=["value"],
                        chart_type="bar",
                        title=f"Analysis: {request.user_prompt[:50]}...",
                        aggregation=None,
                        row_count=len(data),
                        y_axis_label="",
                        analysis=explanation,
                        answer=None
                    )
                else:
                    # Scalar or text result
                    final_answer = f"{explanation}\n\nResult: {exec_result}"
                    return ChartResponse(
                        data=[], 
                        x_axis_key="", 
                        y_axis_keys=[], 
                        chart_type="empty", 
                        title="Analysis Result", 
                        aggregation=None,
                        row_count=0,
                        y_axis_label="",
                        analysis=final_answer,
                        answer=str(exec_result)
                    )

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Execution Failed: {str(e)}")

        # Legacy JSON Path - with text-only fallback
        content = resp.choices[0].message.content or ""
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:-1])
        
        # Try to parse as JSON, fallback to text response if it fails
        try:
            config = json.loads(cleaned)
        except json.JSONDecodeError:
            # AI returned plain text instead of JSON - return as text answer
            return ChartResponse(
                data=[],
                x_axis_key="",
                y_axis_keys=[],
                chart_type="empty",
                title="AI Response",
                aggregation=None,
                row_count=0,
                y_axis_label="",
                analysis=content,  # Original content as analysis
                answer=content
            )
        if not isinstance(config, dict):
            return _safe_empty_chart_response("I couldn't find that specific column in the data.")
        
        x_axis_candidate = config.get("xAxisKey")
        y_axis_candidate = config.get("yAxisKeys", [])
        metric_cols = [c for c, t in ds.column_types.items() if t == SemanticType.METRIC][:3]
        temporal_cols = [c for c, t in ds.column_types.items() if t == SemanticType.TEMPORAL][:2]
        category_cols = [c for c, t in ds.column_types.items() if t in {SemanticType.CATEGORICAL, SemanticType.IDENTIFIER}][:3]
        examples = []
        if metric_cols and category_cols:
            examples.append(f"Show average {metric_cols[0]} by {category_cols[0]}")
            examples.append(f"Top categories in {category_cols[0]} by total {metric_cols[0]}")
        if metric_cols and temporal_cols:
            examples.append(f"Trend total {metric_cols[0]} by {temporal_cols[0]}")
        while len(examples) < 3:
            examples.append("Show total by category")
        guidance = (
            "I need one grouping column and at least one metric to build this chart. "
            f"Try: '{examples[0]}', '{examples[1]}', or '{examples[2]}'."
        )
        if not isinstance(x_axis_candidate, str) or not x_axis_candidate:
            return _safe_empty_chart_response(guidance)
        if not isinstance(y_axis_candidate, list) or not all(isinstance(c, str) for c in y_axis_candidate):
            return _safe_empty_chart_response(guidance)
        if len(y_axis_candidate) == 0:
            return _safe_empty_chart_response(guidance)

        # Validate axis columns and gracefully degrade if hallucinated.
        all_cols = [x_axis_candidate] + y_axis_candidate
        valid, missing, sugg = validate_columns(df, all_cols)
        if not valid:
            suggestions = []
            for c in missing:
                for s in sugg.get(c, []):
                    if s not in suggestions:
                        suggestions.append(s)
            msg = "I couldn't find one or more requested columns in this dataset."
            if suggestions:
                msg += f" Did you mean: {', '.join(suggestions[:5])}?"
            return _safe_empty_chart_response(msg)

        # Parse and validate LLM filters, then merge with UI filters.
        llm_raw_filters = config.get("filters", [])
        llm_filters: list[FilterConfig] = []
        if isinstance(llm_raw_filters, list):
            for raw_filter in llm_raw_filters:
                if not isinstance(raw_filter, dict):
                    continue
                try:
                    parsed_filter = FilterConfig(**raw_filter)
                    if isinstance(parsed_filter.value, (dict, list, tuple, set)):
                        continue
                    llm_filters.append(parsed_filter)
                except (ValidationError, TypeError):
                    continue

        for f in llm_filters:
            if f.column not in valid_columns:
                return _safe_empty_chart_response("I couldn't find that specific column in the data.")
            if f.operator and f.operator.lower() not in ALLOWED_FILTER_OPERATORS:
                return _safe_empty_chart_response("I couldn't find that specific column in the data.")

        combined_filters = (request.filters or []) + llm_filters
        
        # Apply filters
        filtered, applied = apply_filters(df, combined_filters)
        if filtered.empty:
            return _safe_empty_chart_response(
                "No rows match the active filters. Try relaxing filters or broadening the date/category selection.",
                title="No Data After Filters"
            )
        
        # Enforce semantic rules
        agg = config.get("aggregation", "sum")
        agg, warnings, auto_y_label = enforce_semantic_rules(agg, y_axis_candidate, ds.column_types)
        
        # Smart aggregation
        x_key = x_axis_candidate
        y_keys = y_axis_candidate
        limit, cap_warning = _resolve_effective_limit(request.limit, default_limit=20)

        if pd.api.types.is_datetime64_any_dtype(filtered.get(x_key)):
            filtered, x_key = smart_resample_dates(filtered, x_key, y_keys, agg)
        elif x_key in filtered.columns:
            group_others = request.group_others if request.group_others is not None else True
            
            if filtered[x_key].nunique() > limit:
                filtered = smart_group_top_n(filtered, x_key, y_keys, agg, n=limit, group_others=group_others)
        
        result, was_capped = aggregate_data(filtered, x_key, y_keys, agg, limit=limit)
        
        data = result.to_dict(orient='records')
        for row in data:
            for k in row:
                if pd.isna(row[k]):
                    row[k] = 0
        
        final_y_label = auto_y_label or config.get("yAxisLabel")
        
        final_warnings = list(warnings or [])
        if cap_warning and (was_capped or request.limit == 0 or (request.limit is not None and request.limit > MAX_CHART_POINTS)):
            final_warnings.append(cap_warning)

        return ChartResponse(
            data=data,
            x_axis_key=x_key,
            y_axis_keys=y_keys,
            chart_type=config.get("chartType", "bar"),
            title=config.get("title", "Chart"),
            aggregation=agg,
            x_axis_label=config.get("xAxisLabel"),
            y_axis_label=final_y_label,
            row_count=len(data),
            analysis=config.get("analysis", "AI-generated configuration."),
            warnings=final_warnings or None,
            applied_filters=applied or None,
            llm_filters=combined_filters or None
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/drilldown")
async def drilldown_endpoint(request: DrillDownRequest):
    ds = get_dataset(request.dataset_id)
    df = ds.df
    
    filtered, _ = apply_filters(df, request.filters)
    
    # Get raw data
    data = filtered.head(request.limit).to_dict(orient='records')
    # Clean NaNs
    for row in data:
        for k in row:
            if pd.isna(row[k]):
                row[k] = None
                
    return {
        "data": data,
        "total_rows": len(filtered),
        "limit": request.limit
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
