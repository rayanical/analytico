"""AI planning for charts; all data execution stays in run_aggregate."""

import json

from fastapi import HTTPException
from pydantic import ValidationError

from core.config import OPENAI_MODEL, chat_completion_options, SYSTEM_PROMPT, get_openai_client
from models import AggregateRequest, ChartResponse, FilterConfig, QueryPlan, QueryRequest
from services.aggregation_service import run_aggregate
from services.response_builders import safe_empty_chart_response
from storage import get_dataset
from utils.dataframe_utils import df_to_markdown
from utils.filtering import validate_columns


_QUERY_PLAN_SCHEMA = QueryPlan.model_json_schema()


def _clarification(message: str) -> ChartResponse:
    return safe_empty_chart_response(message)


def run_query(request: QueryRequest) -> ChartResponse:
    """Ask the provider for a validated plan, then run the shared chart pipeline."""
    # Resolve the dataset first so missing data has a useful 404 even when AI is off.
    dataset = get_dataset(request.dataset_id)
    df = dataset.df
    column_details = [
        f"- {column} ({dataset.column_types.get(column, 'unknown').upper()}, "
        f"format: {dataset.column_formats.get(column, 'number')})"
        for column in df.columns
    ]
    preferred_metrics = [column for column, kind in dataset.column_types.items() if kind == "metric"][:8]
    preferred_dates = [column for column, kind in dataset.column_types.items() if kind == "temporal"][:5]
    user_message = f"""Question: {request.user_prompt}

Dataset columns:
{chr(10).join(column_details)}

Sample rows:
{df_to_markdown(df, 5)}

Row count: {len(df)}
Exact column names: {json.dumps(df.columns.tolist())}
Preferred measures: {json.dumps(preferred_metrics)}
Preferred date columns: {json.dumps(preferred_dates)}

Use the supplied schema exactly. For unsupported calculations, return a clarification plan."""

    try:
        response = get_openai_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "analytics_query_plan",
                    "strict": True,
                    "schema": _QUERY_PLAN_SCHEMA,
                },
            },
            **chat_completion_options(800, 0),
        )
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"AI query planning failed: {error}") from error

    message = response.choices[0].message
    # The provider has no tools registered. Treat any unexpected tool request as
    # an invalid plan and never interpret or execute its arguments.
    if getattr(message, "tool_calls", None):
        return _clarification(
            "The query planner returned an unsupported action. Please rephrase this as a grouped chart or ask for a supported aggregation."
        )
    content = message.content or ""
    try:
        plan = QueryPlan.model_validate_json(content)
    except (ValidationError, ValueError, TypeError):
        return _clarification(
            "I couldn't build a valid chart plan from that request. Try a grouped sum, average, count, minimum, or maximum."
        )

    if plan.kind == "clarification":
        return _clarification(plan.clarification or "Please clarify the calculation you want to see.")

    assert plan.x_axis_key is not None
    all_requested_columns = [plan.x_axis_key, *plan.y_axis_keys, *(item.column for item in plan.filters)]
    valid, missing, _ = validate_columns(df, all_requested_columns)
    if not valid:
        return _clarification(
            "The chart plan referenced a column that is not in this dataset. Please try again using the listed column names."
        )

    interpretations = df.attrs.get("column_interpretations", {})
    review_columns = [column for column in all_requested_columns
                      if column in interpretations and interpretations[column]["runtime_status"] != "applied"]
    if review_columns:
        return _clarification(
            "These columns need interpretation review before an AI chart can use them: "
            + ", ".join(dict.fromkeys(review_columns))
            + ". Source values are preserved; review their units and parsing in a manual chart."
        )

    ai_filters = [
        FilterConfig(column=item.column, operator=item.operator, value=item.value)
        for item in plan.filters
    ]
    effective_filters = [*(request.filters or []), *ai_filters]
    aggregate_request = AggregateRequest(
        dataset_id=request.dataset_id,
        x_axis_key=plan.x_axis_key,
        y_axis_keys=plan.y_axis_keys,
        aggregation=plan.aggregation,
        chart_type=plan.chart_type,
        filters=effective_filters or None,
        limit=request.limit,
        sort_by=request.sort_by,
        group_others=request.group_others,
    )

    try:
        chart = run_aggregate(aggregate_request)
    except HTTPException as error:
        if error.status_code == 400:
            return _clarification(str(error.detail))
        raise

    chart.title = plan.title or chart.title
    chart.x_axis_label = plan.x_axis_label
    if plan.y_axis_label:
        chart.y_axis_label = plan.y_axis_label
    chart.filters = effective_filters or None
    chart.llm_filters = ai_filters or None
    # Planner text is not an analysis of the computed result.
    chart.analysis = None
    return chart
