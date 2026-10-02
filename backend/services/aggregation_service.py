"""One shared execution path for manual charts and AI-planned charts."""

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import HTTPException

from core.config import MAX_CHART_POINTS, OPENAI_MODEL, chat_completion_options, get_openai_client
from models import AggregateRequest, ChartResponse
from modules import aggregate_data, enforce_semantic_rules, smart_group_top_n, smart_resample_dates
from storage import get_dataset, lease_dataset
from utils.dataframe_utils import df_to_markdown
from utils.bounded_cache import BoundedTTLCache
from utils.filtering import FilterValidationError, apply_filters, resolve_effective_limit, validate_columns


_CHART_CACHE = BoundedTTLCache[str, ChartResponse](
    max_entries=256,
    max_bytes=24 * 1024 * 1024,
    ttl_seconds=5 * 60,
)


def _json_value(value: Any):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        stamp = pd.Timestamp(value)
        return None if pd.isna(stamp) else stamp.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if type(value) is int and abs(value) > 2**53 - 1:
        return str(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def records_to_json(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {column: _json_value(value) for column, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _sort_result(result: pd.DataFrame, x_key: str, y_keys: list[str], sort_by: str) -> pd.DataFrame:
    if sort_by == "value" and y_keys:
        return result.sort_values(y_keys[0], ascending=False, na_position="last", kind="mergesort")
    axis = result[x_key]
    if pd.api.types.is_numeric_dtype(axis) or pd.api.types.is_datetime64_any_dtype(axis):
        return result.sort_values(x_key, ascending=True, na_position="last", kind="mergesort")
    return result.sort_values(
        x_key,
        ascending=True,
        na_position="last",
        kind="mergesort",
        key=lambda values: values.map(lambda item: "" if pd.isna(item) else str(item)),
    )


def _chart_cache_key(dataset: Any, request: AggregateRequest) -> str:
    """Hash the dataset version and every validated request field."""
    version = getattr(dataset, "cache_version", None)
    dataset_id = getattr(dataset, "id", "")
    if isinstance(version, str) and version:
        version = f"{dataset_id}:{version}"
    else:
        # Test adapters and older in-process adapters may not expose a version.
        # Object identity prevents one replacement object from inheriting a
        # cached response; mutable adapters must provide/update cache_version.
        version = f"{dataset_id}:{id(dataset)}:{id(dataset.df)}"
    serialized_request = json.dumps(
        request.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    payload = f"{version}:{serialized_request}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _with_analysis(chart: ChartResponse, request: AggregateRequest) -> ChartResponse:
    """Generate fresh analysis for a chart, including when its rows were cached."""
    summary = df_to_markdown(pd.DataFrame(chart.data).head(20), n=20)
    prompt = f"""Analyze this computed chart result in two concise sentences.
State what the aggregation shows and one pattern or exception visible in these values.
Do not claim trends beyond the rows provided.

X-axis: {request.x_axis_key}
Measures: {', '.join(request.y_axis_keys)}
Aggregation: {chart.aggregation}
Computed chart rows:
{summary}"""
    try:
        analysis_response = get_openai_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a concise data analyst. Use only the supplied computed results."},
                {"role": "user", "content": prompt},
            ],
            **chat_completion_options(120, 0.3),
        )
        analysis = (analysis_response.choices[0].message.content or "").strip() or None
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Analysis failed: {error}") from error
    return chart.model_copy(update={"analysis": analysis}, deep=True)


def run_aggregate(request: AggregateRequest) -> ChartResponse:
    """Validate filters, aggregate source rows once, then sort and limit."""
    with lease_dataset(request.dataset_id, loader=get_dataset) as ds:
        if request.x_axis_key in request.y_axis_keys:
            raise HTTPException(400, "Choose different columns for the X-axis and measures.")
        chart = _run_dataset_aggregate(ds, request.model_copy(update={"include_analysis": False}))
    return _with_analysis(chart, request) if request.include_analysis else chart


def _run_dataset_aggregate(ds, request: AggregateRequest) -> ChartResponse:
    if hasattr(ds, "disk"):
        cache_key = _chart_cache_key(ds, request)
        chart = _CHART_CACHE.get(cache_key)
        if chart is None:
            try:
                aggregation, warnings, y_axis_label = enforce_semantic_rules(
                    request.aggregation, request.y_axis_keys, ds.column_types
                )
                chart = ds.disk.aggregate(request.model_copy(update={"aggregation": aggregation}))
                chart.warnings = [*warnings, *(chart.warnings or [])] or None
                chart.y_axis_label = y_axis_label
            except (ValueError, FilterValidationError) as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
            _CHART_CACHE.set(cache_key, chart)
        return _with_analysis(chart, request) if request.include_analysis else chart
    df = ds.df

    valid, missing, suggestions = validate_columns(df, [request.x_axis_key, *request.y_axis_keys])
    if not valid:
        message = "; ".join(
            f"'{column}' not found, try: {suggestions.get(column, [])}" for column in missing
        )
        raise HTTPException(status_code=400, detail=message)

    try:
        aggregation, warnings, y_axis_label = enforce_semantic_rules(
            request.aggregation, request.y_axis_keys, ds.column_types
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    cache_key = _chart_cache_key(ds, request)
    cached = _CHART_CACHE.get(cache_key)
    if cached is not None:
        return _with_analysis(cached, request) if request.include_analysis else cached

    try:
        filtered, applied_filters = apply_filters(df, request.filters)
        if filtered.empty:
            raise HTTPException(status_code=400, detail="No data matches filters.")

        limit, cap_warning = resolve_effective_limit(request.limit, default_limit=50)
        group_others = request.group_others if request.group_others is not None else True
        is_datetime = pd.api.types.is_datetime64_any_dtype(filtered[request.x_axis_key])
        if request.time_bucket is not None and not is_datetime:
            raise HTTPException(status_code=400, detail="Time bucketing requires a datetime X-axis.")

        time_bucket = None
        others_label = None
        working = filtered
        if is_datetime:
            try:
                working, time_bucket = smart_resample_dates(
                    working,
                    request.x_axis_key,
                    request.y_axis_keys,
                    aggregation,
                    bucket=request.time_bucket,
                )
            except ValueError as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
        elif group_others and working[request.x_axis_key].nunique(dropna=False) > limit:
            working = smart_group_top_n(
                working,
                request.x_axis_key,
                request.y_axis_keys,
                aggregation,
                n=limit,
                group_others=True,
            )
            others_label = working.attrs.get("others_label")

        try:
            result, _ = aggregate_data(
                working,
                request.x_axis_key,
                request.y_axis_keys,
                aggregation,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        sort_by = request.sort_by
        if sort_by is None:
            axis_type = ds.column_types.get(request.x_axis_key)
            sort_by = "label" if axis_type in {"metric", "temporal"} else "value"
        result = _sort_result(result, request.x_axis_key, request.y_axis_keys, sort_by)
        was_capped = len(result) > limit
        if was_capped:
            result = result.head(limit)

    except FilterValidationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    final_warnings = list(warnings)
    interpretations = df.attrs.get("column_interpretations", {})
    review_columns = [col for col in [request.x_axis_key, *request.y_axis_keys]
                      if col in interpretations and interpretations[col]["runtime_status"] != "applied"]
    if review_columns:
        final_warnings.append("Interpretation needs review for: " + ", ".join(dict.fromkeys(review_columns)))
    if cap_warning and (was_capped or request.limit == 0 or (request.limit is not None and request.limit > MAX_CHART_POINTS)):
        final_warnings.append(cap_warning)

    chart = ChartResponse(
        data=records_to_json(result),
        x_axis_key=request.x_axis_key,
        y_axis_keys=request.y_axis_keys,
        chart_type=request.chart_type,
        title=f"{', '.join(request.y_axis_keys)} by {request.x_axis_key}".replace("_", " ").title(),
        aggregation=aggregation,
        y_axis_label=y_axis_label,
        row_count=len(result),
        analysis=None,
        warnings=final_warnings or None,
        applied_filters=applied_filters or None,
        filters=request.filters or None,
        source_x_axis_key=request.x_axis_key,
        time_bucket=time_bucket,
        others_label=others_label,
    )
    response = _with_analysis(chart, request) if request.include_analysis else chart
    # Cache only after the complete request succeeds. Analysis is per request and
    # is always regenerated, while the deterministic chart rows remain reusable.
    _CHART_CACHE.set(cache_key, chart)
    return response
