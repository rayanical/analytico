"""Present an ungrouped engine result through the existing chart interface."""
from models import AggregateRequest, ChartResponse


def result_keys(columns, measures):
    used = set(columns)
    def fresh(base):
        name = base
        while name in used:
            name += "_"
        used.add(name)
        return name
    return fresh("__overall__"), measures or [fresh("__row_count__")]


def overall_chart(request: AggregateRequest, axis: str, measures: list[str], values: list, applied_filters: list[str]) -> ChartResponse:
    return ChartResponse(
        aggregation_scope="overall", count_rows=not request.y_axis_keys,
        data=[{axis: "Overall", **dict(zip(measures, values))}],
        x_axis_key=axis, y_axis_keys=measures, chart_type="bar",
        title="Row count" if not request.y_axis_keys else f"Overall {request.aggregation}: {', '.join(measures)}",
        aggregation=request.aggregation, row_count=1,
        x_axis_label="", y_axis_label="Rows" if not request.y_axis_keys else ("Non-missing values" if request.aggregation == "count" else None),
        filters=request.filters or None, applied_filters=applied_filters or None,
    )
