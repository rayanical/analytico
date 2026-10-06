"""Pydantic models for the local analytics API."""

import math
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator
from modules.import_policy import ImportSettings, ColumnOverride


class ImportPreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    settings: ImportSettings


class ImportConfirmRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ai_column_analysis: bool = False
    settings: Optional[ImportSettings] = None
    column_overrides: list[ColumnOverride] = Field(default_factory=list, max_length=256)


class SchemaApplyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_version: StrictStr = Field(min_length=1, max_length=128)
    column_overrides: list[ColumnOverride] = Field(max_length=256)
    settings: Optional[ImportSettings] = None


Aggregation = Literal["sum", "mean", "median", "count", "min", "max"]
ChartType = Literal["bar", "line", "area", "pie", "composed"]
FilterOperator = Literal["eq", "gt", "lt", "gte", "lte", "contains"]
TimeBucket = Literal["year", "month", "week"]
ColumnFormat = Literal["currency", "percentage", "number", "date", "identifier", "general"]
SemanticType = Literal["metric", "identifier", "temporal", "categorical", "unknown"]


def _is_primitive(value: Any) -> bool:
    if value is None or type(value) in (str, int, bool):
        return True
    return type(value) is float and math.isfinite(value)


class MetricSummary(BaseModel):
    name: str
    total: float
    average: float
    min: float
    max: float
    aggregation: Literal["sum", "mean"] = "sum"


class TimeRange(BaseModel):
    column: str
    start: str
    end: str


class DataProfile(BaseModel):
    top_metrics: list[MetricSummary]
    time_range: Optional[TimeRange]
    row_count: int
    column_count: int


class DataHealth(BaseModel):
    missing_values: dict[str, int]
    cleaning_actions: list[str]
    quality_score: float


class DefaultChart(BaseModel):
    x_axis_key: str
    y_axis_keys: list[str]
    chart_type: ChartType
    aggregation: Aggregation
    title: str
    analysis: str


class ColumnInterpretationMetadata(BaseModel):
    status: Literal["ok", "disabled", "unavailable", "error", "uncertain", "skipped"]
    runtime_status: Literal["applied", "clarification"]
    provider: str
    model: Optional[str] = None
    prompt_version: Optional[str] = None
    latency_ms: Optional[float] = None
    usage: dict[str, Any] = Field(default_factory=dict)
    confidence: Optional[dict[str, Any]] = None
    error_code: Optional[str] = None
    decision: Optional[dict[str, Any]] = None
    cache_hit: bool = False


class ColumnSummary(BaseModel):
    name: str
    original_name: Optional[str] = None
    display_name: Optional[str] = None
    dtype: str
    is_numeric: bool
    is_datetime: bool
    semantic_type: SemanticType
    format: ColumnFormat
    unique_count: int
    sample_values: list[Any]
    interpretation: Optional[ColumnInterpretationMetadata] = None


class UploadResponse(BaseModel):
    dataset_id: str
    filename: str
    row_count: int
    columns: list[ColumnSummary]
    column_formats: dict[str, ColumnFormat]
    data_health: DataHealth
    profile: DataProfile
    default_chart: Optional[DefaultChart]
    suggestions: list[str]
    summary: Optional[str] = None
    enrichment_status: Literal["pending", "running", "done", "error", "disabled"] = "disabled"
    version: Optional[str] = None
    column_schema: list[dict[str, Any]] = Field(default_factory=list)


class FilterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    column: StrictStr = Field(min_length=1, max_length=256)
    operator: Optional[FilterOperator] = None
    value: Any = None
    values: Optional[list[Any]] = Field(default=None, max_length=500)
    min_val: Any = None
    max_val: Any = None

    @model_validator(mode="after")
    def validate_filter(self):
        if self.operator is not None and self.value is None:
            raise ValueError("A filter operator requires a non-null value.")
        if self.operator is None and self.value is not None:
            raise ValueError("A filter value requires an operator.")
        if self.values is not None and not self.values:
            raise ValueError("A values filter must contain at least one value.")

        values = [self.value, self.min_val, self.max_val]
        if self.values is not None:
            values.extend(self.values)
        if not all(_is_primitive(value) for value in values):
            raise ValueError("Filter values must be finite JSON primitives.")
        if all(value is None for value in (self.value, self.values, self.min_val, self.max_val)):
            raise ValueError("A filter must include an operator value, values list, or range bound.")
        if self.operator == "contains" and not isinstance(self.value, str):
            raise ValueError("The contains operator requires a string value.")
        return self


class AggregateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    dataset_id: StrictStr = Field(min_length=1, max_length=128)
    x_axis_key: Optional[StrictStr] = Field(default=None, min_length=1, max_length=256)
    y_axis_keys: list[StrictStr] = Field(max_length=16)
    aggregation: Aggregation = "sum"
    chart_type: ChartType = "bar"
    filters: Optional[list[FilterConfig]] = Field(default=None, max_length=50)
    limit: Optional[int] = Field(default=None, ge=0, le=10_000)
    sort_by: Optional[Literal["value", "label"]] = "value"
    group_others: Optional[bool] = True
    include_analysis: bool = False
    time_bucket: Optional[TimeBucket] = None

    @model_validator(mode="after")
    def validate_measure_columns(self):
        if not self.y_axis_keys and (self.x_axis_key is not None or self.aggregation != "count"):
            raise ValueError("Measures are required except for an overall row count.")
        if self.x_axis_key is None and self.time_bucket is not None:
            raise ValueError("Overall aggregations cannot use time bucketing.")
        if len(set(self.y_axis_keys)) != len(self.y_axis_keys):
            raise ValueError("Measure columns must be unique.")
        return self


class ChartResponse(BaseModel):
    aggregation_scope: Literal["grouped", "overall"] = "grouped"
    count_rows: bool = False
    data: list[dict[str, Any]]
    x_axis_key: str
    y_axis_keys: list[str]
    chart_type: Literal["bar", "line", "area", "pie", "composed", "empty"]
    title: str
    aggregation: Optional[Aggregation] = None
    x_axis_label: Optional[str] = None
    y_axis_label: Optional[str] = None
    row_count: int
    analysis: Optional[str] = None
    warnings: Optional[list[str]] = None
    applied_filters: Optional[list[str]] = None
    filters: Optional[list[FilterConfig]] = None
    llm_filters: Optional[list[FilterConfig]] = None
    source_x_axis_key: Optional[str] = None
    time_bucket: Optional[TimeBucket] = None
    others_label: Optional[StrictStr] = None
    answer: Optional[str] = None


class QueryFilterPlan(BaseModel):
    """Narrow filter shape accepted from the AI query planner."""

    model_config = ConfigDict(extra="forbid", strict=True)

    column: StrictStr = Field(min_length=1, max_length=256)
    operator: FilterOperator
    value: StrictStr | int | float | bool

    @model_validator(mode="after")
    def validate_value(self):
        if type(self.value) is float and not math.isfinite(self.value):
            raise ValueError("The filter value must be finite.")
        return self


class QueryPlan(BaseModel):
    """Strict structured output from the AI query planner."""

    model_config = ConfigDict(extra="forbid", strict=True)

    kind: Literal["chart", "clarification"]
    x_axis_key: Optional[StrictStr]
    y_axis_keys: list[StrictStr] = Field(max_length=16)
    aggregation: Aggregation
    chart_type: ChartType
    title: StrictStr = Field(max_length=160)
    x_axis_label: Optional[StrictStr]
    y_axis_label: Optional[StrictStr]
    filters: list[QueryFilterPlan] = Field(max_length=50)
    clarification: Optional[StrictStr] = Field(max_length=500)

    @model_validator(mode="after")
    def validate_plan_shape(self):
        if len(set(self.y_axis_keys)) != len(self.y_axis_keys):
            raise ValueError("Measure columns must be unique.")
        if self.kind == "chart" and not self.y_axis_keys and (self.x_axis_key is not None or self.aggregation != "count"):
            raise ValueError("Chart plans require measures except for an overall row count.")
        if self.x_axis_key is not None and not self.x_axis_key.strip():
            raise ValueError("A grouping column must not be blank.")
        if self.kind == "clarification" and not self.clarification:
            raise ValueError("Clarification plans require a user-facing explanation.")
        return self


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    dataset_id: StrictStr = Field(min_length=1, max_length=128)
    user_prompt: StrictStr = Field(min_length=1, max_length=2_000)
    filters: Optional[list[FilterConfig]] = Field(default=None, max_length=50)
    limit: Optional[int] = Field(default=None, ge=0, le=10_000)
    sort_by: Optional[Literal["value", "label"]] = None
    group_others: Optional[bool] = None


class DrillDownRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    dataset_id: StrictStr = Field(min_length=1, max_length=128)
    filters: Optional[list[FilterConfig]] = Field(default=None, max_length=50)
    limit: int = Field(default=50, ge=1, le=5_000)
