import type { ColumnFormat, DatasetState, DefaultChartType, SemanticType } from '@/types';

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

const columnFormats: ColumnFormat[] = ['currency', 'percentage', 'number', 'date', 'identifier', 'general'];
const semanticTypes: SemanticType[] = ['metric', 'identifier', 'temporal', 'categorical', 'unknown'];
const defaultChartTypes: DefaultChartType[] = ['bar', 'line', 'area', 'pie', 'composed'];
const aggregations = ['sum', 'mean', 'median', 'count', 'min', 'max'] as const;

function isColumnSummary(value: unknown): boolean {
  return isRecord(value)
    && typeof value.name === 'string'
    && typeof value.dtype === 'string'
    && typeof value.is_numeric === 'boolean'
    && typeof value.is_datetime === 'boolean'
    && semanticTypes.includes(value.semantic_type as SemanticType)
    && columnFormats.includes(value.format as ColumnFormat)
    && typeof value.unique_count === 'number'
    && Array.isArray(value.sample_values);
}

function isMetricSummary(value: unknown): boolean {
  return isRecord(value)
    && typeof value.name === 'string'
    && typeof value.total === 'number'
    && typeof value.average === 'number'
    && typeof value.min === 'number'
    && typeof value.max === 'number'
    && (value.aggregation === undefined || value.aggregation === 'sum' || value.aggregation === 'mean');
}

function isTimeRange(value: unknown): boolean {
  return isRecord(value)
    && typeof value.column === 'string'
    && typeof value.start === 'string'
    && typeof value.end === 'string';
}

function isDefaultChart(value: unknown): boolean {
  return isRecord(value)
    && typeof value.x_axis_key === 'string'
    && Array.isArray(value.y_axis_keys) && value.y_axis_keys.every(key => typeof key === 'string')
    && defaultChartTypes.includes(value.chart_type as DefaultChartType)
    && aggregations.includes(value.aggregation as typeof aggregations[number])
    && typeof value.title === 'string'
    && typeof value.analysis === 'string';
}

export function isDatasetState(value: unknown): value is DatasetState {
  return isRecord(value)
    && typeof value.datasetId === 'string'
    && typeof value.filename === 'string'
    && typeof value.rowCount === 'number'
    && Array.isArray(value.columns) && value.columns.every(isColumnSummary)
    && isRecord(value.columnFormats) && Object.values(value.columnFormats).every(format =>
      columnFormats.includes(format as ColumnFormat))
    && isRecord(value.dataHealth) && isRecord(value.dataHealth.missing_values)
    && Object.values(value.dataHealth.missing_values).every(count => typeof count === 'number')
    && Array.isArray(value.dataHealth.cleaning_actions)
    && value.dataHealth.cleaning_actions.every(action => typeof action === 'string')
    && typeof value.dataHealth.quality_score === 'number'
    && isRecord(value.profile) && Array.isArray(value.profile.top_metrics)
    && value.profile.top_metrics.every(isMetricSummary)
    && (value.profile.time_range === null || isTimeRange(value.profile.time_range))
    && typeof value.profile.row_count === 'number' && typeof value.profile.column_count === 'number'
    && (value.defaultChart === null || isDefaultChart(value.defaultChart))
    && Array.isArray(value.suggestions) && value.suggestions.every(suggestion => typeof suggestion === 'string')
    && (value.summary === undefined || value.summary === null || typeof value.summary === 'string');
}
