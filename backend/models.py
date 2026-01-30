"""
Analytico Backend - Pydantic Models
All request/response schemas
"""

from typing import Any, Optional
from pydantic import BaseModel


class MetricSummary(BaseModel):
    name: str
    total: float
    average: float
    min: float
    max: float


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
    chart_type: str
    aggregation: str
    title: str
    reasoning: str


class ColumnSummary(BaseModel):
    name: str
    dtype: str
    is_numeric: bool
    is_datetime: bool
    semantic_type: str
    format: str
    unique_count: int
    sample_values: list[Any]


class UploadResponse(BaseModel):
    dataset_id: str
    filename: str
    row_count: int
    columns: list[ColumnSummary]
    column_formats: dict[str, str]
    data_health: DataHealth
    profile: DataProfile
    default_chart: Optional[DefaultChart]
    suggestions: list[str]
    summary: Optional[str] = None


class FilterConfig(BaseModel):
    column: str
    values: Optional[list[Any]] = None
    min_val: Optional[Any] = None
    max_val: Optional[Any] = None


class AggregateRequest(BaseModel):
    dataset_id: str
    x_axis_key: str
    y_axis_keys: list[str]
    aggregation: str = "sum"
    chart_type: str = "bar"
    filters: Optional[list[FilterConfig]] = None
    limit: Optional[int] = None
    sort_by: Optional[str] = "value"  # "value" or "label"
    group_others: Optional[bool] = True


class ChartResponse(BaseModel):
    data: list[dict[str, Any]]
    x_axis_key: str
    y_axis_keys: list[str]
    chart_type: str
    title: str
    x_axis_label: Optional[str] = None
    y_axis_label: Optional[str] = None
    row_count: int
    reasoning: Optional[str] = None
    warnings: Optional[list[str]] = None
    applied_filters: Optional[list[str]] = None
    answer: Optional[str] = None  # For scalar results from Python queries


class QueryRequest(BaseModel):
    dataset_id: str
    user_prompt: str
    filters: Optional[list[FilterConfig]] = None
    limit: Optional[int] = None
    group_others: Optional[bool] = None


class DrillDownRequest(BaseModel):
    dataset_id: str
    filters: Optional[list[FilterConfig]] = None
    limit: int = 50
