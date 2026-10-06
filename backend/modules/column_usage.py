"""Shared interpretation of factual column profiles, independent of storage engine.

Cardinality controls presentation elsewhere; it never establishes business meaning.
Header hints are proposals, not proof of units or an additive business measure.
"""
from dataclasses import asdict, dataclass

from modules.column_statistics import ColumnStatistics


_KEY_TOKENS = {"id", "code", "key", "zip", "postal", "ssn", "account", "serial", "ref", "reference", "phone"}
_DIMENSION_TOKENS = {"type", "category", "status", "flag", "group", "class", "segment"}
_QUANTITY_TOKENS = {
    "count", "cnt", "quantity", "amount", "price", "cost", "revenue", "income",
    "salary", "expense", "balance", "tax", "fee", "surcharge", "distance",
    "duration", "age", "weight", "height", "length", "temperature", "value",
    "total", "score", "ratio", "rate", "percent", "percentage", "pct",
    "average", "mean", "expectancy",
}
# Calendar domains are syntax constraints, not cardinality thresholds.
_CALENDAR_DOMAINS = {"month": (1, 12), "hour": (0, 23), "day": (1, 31),
                     "weekday": (0, 6), "quarter": (1, 4)}


@dataclass(frozen=True)
class ColumnUsage:
    physical_type: str
    role: str
    evidence: str
    can_group: bool
    can_measure: bool
    automatic_measure: bool
    preferred_aggregation: str | None

    def metadata(self):
        return asdict(self)


def infer_column_usage(name: str, physical_type: str, statistics: ColumnStatistics,
                       *, role_override=None, aggregation_override=None) -> ColumnUsage:
    """Interpret complete-column facts; an unknown role retains numeric capability."""
    tokens = set(name.lower().split("_"))
    numeric = physical_type == "number"
    if role_override is not None:
        role, evidence = role_override, "explicit_role"
    elif not statistics.unique_count:
        role, evidence = "unknown", "no_observations"
    elif physical_type == "datetime":
        role, evidence = "temporal", "validated_datetime"
    elif tokens & _KEY_TOKENS or name.lower().endswith("_no"):
        role, evidence = "identifier", "key_name_hint"
    elif physical_type == "boolean" or tokens & _DIMENSION_TOKENS:
        role, evidence = "categorical", "dimension_type_or_name"
    elif statistics.numeric_year and not tokens & _QUANTITY_TOKENS:
        role, evidence = "temporal", "validated_calendar_year"
    elif numeric and statistics.row_position_sequence:
        role, evidence = "unknown", "row_position_sequence"
    elif numeric and not tokens & _QUANTITY_TOKENS and any(
        token in tokens and statistics.numeric_min is not None
        and not statistics.has_fractional_values
        and low <= statistics.numeric_min <= statistics.numeric_max <= high
        for token, (low, high) in _CALENDAR_DOMAINS.items()
    ):
        role, evidence = "categorical", "validated_calendar_component"
    elif numeric and tokens & _QUANTITY_TOKENS:
        role, evidence = "metric", "quantity_name_hint"
    elif numeric and statistics.has_fractional_values:
        role, evidence = "metric", "fractional_numeric_candidate"
    elif numeric:
        role, evidence = "unknown", "ambiguous_integer"
    else:
        role, evidence = "categorical", "text_grouping_default"
    can_measure = numeric and role != "identifier"
    # An explicit aggregation can make a numeric field useful without relabeling it.
    automatic = can_measure and (role == "metric" or aggregation_override in {"sum", "mean", "median", "min", "max"})
    automatic = automatic and (not statistics.row_position_sequence or role_override == "metric")
    if aggregation_override == "none":
        automatic = False
    preferred = aggregation_override if aggregation_override is not None else ("mean" if automatic else None)
    return ColumnUsage(physical_type, role, evidence, bool(statistics.unique_count),
                       can_measure, automatic, preferred)


def choose_grouping(columns, usages, statistics):
    """Prefer a varying dimension; cardinality affects presentation only."""
    candidates = [column for column in columns if usages[column].can_group
                  and usages[column].role not in {"metric", "identifier", "temporal"}
                  and (not statistics[column].row_position_sequence
                       or usages[column].evidence == "explicit_role")]
    return next((c for c in candidates if statistics[c].unique_count > 1),
                candidates[0] if candidates else None)


def default_chart_plan(columns, usages, statistics):
    temporal = next((c for c in columns if usages[c].role == "temporal"
                     and (usages[c].physical_type == "datetime" or statistics[c].numeric_year)), None)
    dimension = choose_grouping(columns, usages, statistics)
    measure = next((c for c in columns if usages[c].automatic_measure), None)
    axis = temporal or dimension
    if measure is not None and axis is not None and measure != axis:
        aggregation = usages[measure].preferred_aggregation
        return {"x_axis_key": axis, "y_axis_keys": [measure],
                "chart_type": "line" if temporal else "bar", "aggregation": aggregation,
                "title": f"{aggregation.title()} {measure} by {axis}".replace("_", " ").title(),
                "analysis": f"Shows the {aggregation} of {measure} for each observed {axis} value."}
    # A frequency view avoids inventing a business measure. Count explicitly refers
    # to non-missing values of the selected source field, matching both engines.
    axis = dimension or temporal
    if axis is None:
        axis = next((c for c in columns if usages[c].can_group
                     and not statistics[c].row_position_sequence), None)
    counted = next((c for c in columns if c != axis and usages[c].can_group), None)
    if axis is None or counted is None:
        return None
    return {"x_axis_key": axis, "y_axis_keys": [counted], "chart_type": "bar",
            "aggregation": "count",
            "title": f"Non-missing {counted} by {axis}".replace("_", " ").title(),
            "analysis": f"Counts non-missing {counted} values grouped by {axis}; no business aggregation is assumed."}
