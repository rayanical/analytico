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
from typing import Any, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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


def apply_filters(df: pd.DataFrame, filters: Optional[list[FilterConfig]]) -> tuple[pd.DataFrame, list[str]]:
    """Apply filter configurations to dataframe"""
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


def df_to_markdown(df: pd.DataFrame, n: int = 5) -> str:
    """Convert dataframe head to markdown table"""
    sample = df.head(n)
    header = "| " + " | ".join(str(c) for c in sample.columns) + " |"
    sep = "| " + " | ".join("---" for _ in sample.columns) + " |"
    rows = ["| " + " | ".join(str(v)[:25] for v in row) + " |" for _, row in sample.iterrows()]
    return "\n".join([header, sep] + rows)


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
        df = pd.read_csv(file.file)
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV is empty.")
        
        # Clean data using data_janitor module
        df, cleaning_actions, missing_counts, col_formats = clean_dataframe(df)
        
        # Detect semantic types using intelligence module
        col_types = {col: detect_semantic_type(df, col) for col in df.columns}
        
        # Generate profile
        profile = auto_profile(df, col_types)
        
        # Quality score
        total_cells = len(df) * len(df.columns)
        missing_total = sum(missing_counts.values())
        quality = max(0, 100 - (missing_total / max(total_cells, 1) * 100))
        
        # Generate Business Summary
        summary = None
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
    
    # Smart grouping (Top N + Others)
    limit = request.limit or 50
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
    result = aggregate_data(filtered, x_key, request.y_axis_keys, agg)
    
    # Sorting
    if request.sort_by == "value" and request.y_axis_keys:
        result = result.sort_values(request.y_axis_keys[0], ascending=False)
    elif request.sort_by == "label":
        result = result.sort_values(x_key, ascending=True)
    
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
        y_axis_label=y_axis_label,
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
            max_tokens=800,
            tools=[{
                "type": "function",
                "function": {
                    "name": "generate_python_analysis",
                    "description": "Write a Python script to answer complex questions or perform advanced calculations. df is available.",
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
                        row_count=len(data),
                        y_axis_label="",
                        reasoning=explanation,
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
                        row_count=len(data),
                        y_axis_label="",
                        reasoning=explanation,
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
                        row_count=0,
                        y_axis_label="",
                        reasoning=final_answer,
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
                row_count=0,
                y_axis_label="",
                reasoning=content,  # Original content as reasoning
                answer=content
            )
        
        # Validate
        all_cols = [config["xAxisKey"]] + config["yAxisKeys"]
        valid, missing, sugg = validate_columns(df, all_cols)
        if not valid:
            msg = "; ".join(f"'{c}' not found, try: {sugg.get(c, [])}'" for c in missing)
            raise HTTPException(status_code=400, detail=msg)
        
        # Apply filters
        filtered, applied = apply_filters(df, request.filters)
        if filtered.empty:
            raise HTTPException(status_code=400, detail="No data after filters.")
        
        # Enforce semantic rules
        agg = config.get("aggregation", "sum")
        agg, warnings, auto_y_label = enforce_semantic_rules(agg, config["yAxisKeys"], ds.column_types)
        
        # Smart aggregation
        x_key = config["xAxisKey"]
        y_keys = config["yAxisKeys"]
        
        if pd.api.types.is_datetime64_any_dtype(filtered.get(x_key)):
            filtered, x_key = smart_resample_dates(filtered, x_key, y_keys, agg)
        elif x_key in filtered.columns:
            limit = request.limit or 20
            group_others = request.group_others if request.group_others is not None else True
            
            if filtered[x_key].nunique() > limit:
                filtered = smart_group_top_n(filtered, x_key, y_keys, agg, n=limit, group_others=group_others)
        
        result = aggregate_data(filtered, x_key, y_keys, agg)
        
        data = result.to_dict(orient='records')
        for row in data:
            for k in row:
                if pd.isna(row[k]):
                    row[k] = 0
        
        final_y_label = auto_y_label or config.get("yAxisLabel")
        
        return ChartResponse(
            data=data,
            x_axis_key=x_key,
            y_axis_keys=y_keys,
            chart_type=config.get("chartType", "bar"),
            title=config.get("title", "Chart"),
            x_axis_label=config.get("xAxisLabel"),
            y_axis_label=final_y_label,
            row_count=len(data),
            reasoning=config.get("reasoning", "AI-generated configuration."),
            warnings=warnings or None,
            applied_filters=applied or None
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
