"""Aggregation orchestration service."""

import os

import pandas as pd
from fastapi import HTTPException

from core.config import MAX_CHART_POINTS, OPENAI_CLIENT
from models import AggregateRequest, ChartResponse
from modules import aggregate_data, enforce_semantic_rules, smart_group_top_n, smart_resample_dates
from storage import get_dataset
from utils.dataframe_utils import df_to_markdown
from utils.filtering import apply_filters, resolve_effective_limit, validate_columns


def run_aggregate(request: AggregateRequest) -> ChartResponse:
    """Execute aggregate endpoint behavior."""
    if request.include_analysis and not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OpenAI API key not configured.")

    ds = get_dataset(request.dataset_id)
    df = ds.df

    all_cols = [request.x_axis_key] + request.y_axis_keys
    valid, missing, suggestions = validate_columns(df, all_cols)
    if not valid:
        msg = "; ".join(f"'{c}' not found, try: {suggestions.get(c, [])}" for c in missing)
        raise HTTPException(status_code=400, detail=msg)

    agg, warnings, y_axis_label = enforce_semantic_rules(request.aggregation, request.y_axis_keys, ds.column_types)

    filtered, applied_filters = apply_filters(df, request.filters)
    if filtered.empty:
        raise HTTPException(status_code=400, detail="No data matches filters.")

    if pd.api.types.is_datetime64_any_dtype(df.get(request.x_axis_key)):
        filtered, new_x = smart_resample_dates(filtered, request.x_axis_key, request.y_axis_keys, agg)
        x_key = new_x
    else:
        x_key = request.x_axis_key

    limit, cap_warning = resolve_effective_limit(request.limit, default_limit=50)
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
                group_others=group_others,
            )

    result, was_capped = aggregate_data(filtered, x_key, request.y_axis_keys, agg, limit=limit)

    sort_by = request.sort_by
    if sort_by is None:
        x_col_meta = next((c for c in ds.columns if c["name"] == request.x_axis_key), None)
        if x_col_meta:
            semantic_type = x_col_meta.get("semantic_type", "categorical")
            if semantic_type in ["metric", "temporal"]:
                sort_by = "label"
            else:
                sort_by = "value"
        else:
            sort_by = "value"

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

            analysis_resp = OPENAI_CLIENT.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a data analyst. Be concise. Only two sentences."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=120,
                temperature=0.3,
            )
            analysis = analysis_resp.choices[0].message.content.strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    data = result.to_dict(orient="records")
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
        title=f"{', '.join(request.y_axis_keys)} by {x_key}".replace("_", " ").title(),
        aggregation=agg,
        y_axis_label=y_axis_label,
        row_count=len(data),
        analysis=analysis,
        warnings=final_warnings or None,
        applied_filters=applied_filters or None,
    )
