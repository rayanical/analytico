"""AI planning for charts; all data execution stays in run_aggregate."""

import json
import os

from fastapi import HTTPException
from pydantic import ValidationError

from core.config import OPENAI_MODEL, chat_completion_options, SYSTEM_PROMPT, get_openai_client
from models import AggregateRequest, ChartResponse, FilterConfig, QueryPlan, QueryRequest
from services.aggregation_service import run_aggregate
from services.response_builders import safe_empty_chart_response
from storage import get_dataset, lease_dataset
from modules.query_decisions import query_context, bounded_context, plan_decisions
from utils.filtering import validate_columns


def _strict_plan_schema():
    schema = QueryPlan.model_json_schema()
    for definition in [schema, *schema.get('$defs', {}).values()]:
        if 'properties' in definition:
            definition['required'] = list(definition['properties'])
            for field in definition['properties'].values():
                field.pop('default', None)
    return schema


_QUERY_PLAN_SCHEMA = _strict_plan_schema()


def _clarification(message: str) -> ChartResponse:
    return safe_empty_chart_response(message)


def _ensure_dataset_unchanged(dataset_id: str, snapshot, snapshot_version) -> None:
    """Do not apply a plan or return an aggregate for a replaced dataset."""
    current = get_dataset(dataset_id)
    if current is not snapshot or (
        snapshot_version is not None
        and getattr(current, "cache_version", None) != snapshot_version
    ):
        raise HTTPException(
            status_code=409,
            detail="The dataset changed while the chart was being planned. Please retry the request.",
        )


def run_query(request: QueryRequest) -> ChartResponse:
    """Ask the provider for a validated plan, then run the shared chart pipeline."""
    # Resolve the dataset first so missing data has a useful 404 even when AI is off.
    with lease_dataset(request.dataset_id, loader=get_dataset) as dataset:
        dataset_version = getattr(dataset, "cache_version", None)
        df = dataset.sample_frame(5) if hasattr(dataset, "sample_frame") else dataset.df
        try:
            context = bounded_context(query_context(dataset, df, request))
        except ValueError as error:
            raise HTTPException(413, "This dataset's schema exceeds the AI planning context limit. Use a manual chart or fewer columns.") from error
    user_message = f"""Dataset context: {json.dumps(context, ensure_ascii=False, allow_nan=False)}

Use exact column keys in the plan; display names are cosmetic aliases.
Respect reviewed units and aggregation settings. AI descriptions and roles are inferences.
Only ask for clarification when the calculation is unsupported or material ambiguity remains.
Active filters already apply; return only additional requested filters.
Use the supplied output schema exactly."""

    def generative_plan():
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
                "The query planner returned an unsupported action. Please ask for a grouped chart or an overall total, average, or count."
            )
        content = message.content or ""
        try:
            plan = QueryPlan.model_validate_json(content)
        except (ValidationError, ValueError, TypeError):
            return _clarification(
                "I couldn't build a valid chart plan from that request. Try an overall or grouped sum, average, count, minimum, or maximum."
            )
        return plan

    backend = os.getenv("QUERY_PLANNER_BACKEND", "decisions").strip().lower()
    if backend not in {"decisions", "generative"}:
        raise HTTPException(500, "Invalid query planner configuration.")
    plan = plan_decisions(context) if backend == "decisions" else None
    if plan is None:
        plan = generative_plan()
    _ensure_dataset_unchanged(request.dataset_id, dataset, dataset_version)
    if isinstance(plan, ChartResponse):
        return plan

    if plan.kind == "clarification":
        return _clarification(plan.clarification or "Please clarify the calculation you want to see.")

    blocked_measures = [item["column"] for item in getattr(dataset, "column_schema", [])
                        if item.get("aggregation") == "none" and item["column"] in plan.y_axis_keys]
    if blocked_measures:
        return _clarification("The reviewed schema marks these columns as not aggregatable: "
                              + ", ".join(blocked_measures) + ". Review their column settings first.")
    all_requested_columns = [*([plan.x_axis_key] if plan.x_axis_key is not None else []), *plan.y_axis_keys, *(item.column for item in plan.filters)]
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
        FilterConfig(column=item.column, operator=item.operator, value=item.value, values=item.values)
        for item in plan.filters
    ]
    effective_filters = [*(request.filters or []), *ai_filters]
    aggregate_request = AggregateRequest(
        dataset_id=request.dataset_id,
        x_axis_key=plan.x_axis_key,
        y_axis_keys=plan.y_axis_keys,
        aggregation=plan.aggregation,
        chart_type=plan.chart_type if plan.x_axis_key is not None else "bar",
        filters=effective_filters or None,
        limit=request.limit,
        sort_by=request.sort_by,
        group_others=request.group_others,
    )

    # Close the gap between validation and run_aggregate's own dataset lease.
    _ensure_dataset_unchanged(request.dataset_id, dataset, dataset_version)
    try:
        chart = run_aggregate(aggregate_request)
    except HTTPException as error:
        if error.status_code == 400:
            return _clarification(str(error.detail))
        raise
    # A replacement can race with aggregation after its lease is acquired. If it
    # did, return a conflict instead of exposing a chart for a different snapshot.
    _ensure_dataset_unchanged(request.dataset_id, dataset, dataset_version)

    chart.title = plan.title or chart.title
    chart.x_axis_label = plan.x_axis_label
    if plan.y_axis_label:
        chart.y_axis_label = plan.y_axis_label
    chart.filters = effective_filters or None
    chart.llm_filters = ai_filters or None
    # Planner text is not an analysis of the computed result.
    chart.analysis = None
    return chart
