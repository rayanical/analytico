"""LLM-driven query orchestration service."""

import json
import os

import pandas as pd
from fastapi import HTTPException
from pydantic import ValidationError

from core.config import ALLOWED_FILTER_OPERATORS, MAX_CHART_POINTS, OPENAI_CLIENT, SYSTEM_PROMPT
from models import ChartResponse, FilterConfig, QueryRequest
from modules import SemanticType, aggregate_data, enforce_semantic_rules, smart_group_top_n, smart_resample_dates
from storage import get_dataset
from services.response_builders import safe_empty_chart_response
from utils.dataframe_utils import df_to_markdown
from utils.execution import secure_exec
from utils.filtering import apply_filters, resolve_effective_limit, validate_columns


def run_query(request: QueryRequest) -> ChartResponse:
    """Execute query endpoint behavior."""
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OpenAI API key not configured.")

    ds = get_dataset(request.dataset_id)
    df = ds.df
    valid_columns = df.columns.tolist()
    preferred_metric_columns = [c for c, t in ds.column_types.items() if t == SemanticType.METRIC][:8]
    preferred_temporal_columns = [c for c, t in ds.column_types.items() if t == SemanticType.TEMPORAL][:5]

    cols_info = [
        f"- {c} ({ds.column_types.get(c, '?').upper()}, format: {ds.column_formats.get(c, 'number')})"
        for c in df.columns
    ]
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

        resp = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=800,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "generate_python_analysis",
                        "description": "Only use this fallback tool if the user's request requires advanced statistics (e.g., correlation, standard deviation, forecasting) or multi-step DataFrame transformations that cannot be represented by a standard aggregated chart. Do NOT use this tool for simple averages, sums, or counts.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "code": {
                                    "type": "string",
                                    "description": "The python code to execute. Assign final answer to variable 'result' or print it.",
                                },
                                "explanation": {
                                    "type": "string",
                                    "description": "Explanation of the analysis.",
                                },
                            },
                            "required": ["code", "explanation"],
                        },
                    },
                }
            ],
        )

        tool_calls = resp.choices[0].message.tool_calls

        if tool_calls:
            call = tool_calls[0]
            args = json.loads(call.function.arguments)
            code = args["code"]
            code = code.strip()
            if code.startswith("```"):
                code_lines = code.splitlines()
                if code_lines:
                    code_lines = code_lines[1:]
                if code_lines and code_lines[-1].strip().startswith("```"):
                    code_lines = code_lines[:-1]
                code = "\n".join(code_lines).strip()
            explanation = args.get("explanation", "")

            try:
                filtered, _ = apply_filters(df, request.filters)
                exec_result = secure_exec(code, filtered)

                if isinstance(exec_result, pd.DataFrame) and len(exec_result) > 0:
                    cols = exec_result.columns.tolist()
                    x_key = cols[0] if len(cols) > 0 else ""
                    y_keys = cols[1:] if len(cols) > 1 else []

                    data = exec_result.head(100).to_dict(orient="records")
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
                        answer=None,
                    )
                if isinstance(exec_result, pd.Series):
                    result_df = exec_result.reset_index()
                    result_df.columns = ["category", "value"]
                    data = result_df.head(50).to_dict(orient="records")

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
                        answer=None,
                    )

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
                    answer=str(exec_result),
                )

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Execution Failed: {str(e)}")

        content = resp.choices[0].message.content or ""
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:-1])

        try:
            config = json.loads(cleaned)
        except json.JSONDecodeError:
            return ChartResponse(
                data=[],
                x_axis_key="",
                y_axis_keys=[],
                chart_type="empty",
                title="AI Response",
                aggregation=None,
                row_count=0,
                y_axis_label="",
                analysis=content,
                answer=content,
            )

        if not isinstance(config, dict):
            return safe_empty_chart_response("I couldn't find that specific column in the data.")

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
            return safe_empty_chart_response(guidance)
        if not isinstance(y_axis_candidate, list) or not all(isinstance(c, str) for c in y_axis_candidate):
            return safe_empty_chart_response(guidance)
        if len(y_axis_candidate) == 0:
            return safe_empty_chart_response(guidance)

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
            return safe_empty_chart_response(msg)

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
                return safe_empty_chart_response("I couldn't find that specific column in the data.")
            if f.operator and f.operator.lower() not in ALLOWED_FILTER_OPERATORS:
                return safe_empty_chart_response("I couldn't find that specific column in the data.")

        combined_filters = (request.filters or []) + llm_filters

        filtered, applied = apply_filters(df, combined_filters)
        if filtered.empty:
            return safe_empty_chart_response(
                "No rows match the active filters. Try relaxing filters or broadening the date/category selection.",
                title="No Data After Filters",
            )

        agg = config.get("aggregation", "sum")
        agg, warnings, auto_y_label = enforce_semantic_rules(agg, y_axis_candidate, ds.column_types)

        x_key = x_axis_candidate
        y_keys = y_axis_candidate
        limit, cap_warning = resolve_effective_limit(request.limit, default_limit=20)

        if pd.api.types.is_datetime64_any_dtype(filtered.get(x_key)):
            filtered, x_key = smart_resample_dates(filtered, x_key, y_keys, agg)
        elif x_key in filtered.columns:
            group_others = request.group_others if request.group_others is not None else True
            if filtered[x_key].nunique() > limit:
                filtered = smart_group_top_n(filtered, x_key, y_keys, agg, n=limit, group_others=group_others)

        result, was_capped = aggregate_data(filtered, x_key, y_keys, agg, limit=limit)

        data = result.to_dict(orient="records")
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
            llm_filters=combined_filters or None,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
