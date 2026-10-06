import type { ColumnFormat, DatasetState, DefaultChartType, SemanticType } from '@/types';

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

const columnFormats: ColumnFormat[] = ['currency', 'percentage', 'number', 'date', 'identifier', 'general'];
const semanticTypes: SemanticType[] = ['metric', 'identifier', 'temporal', 'categorical', 'unknown'];
const defaultChartTypes: DefaultChartType[] = ['bar', 'line', 'area', 'pie', 'composed'];
const enrichmentStatuses = ['pending', 'running', 'done', 'error', 'disabled'] as const;
const aggregations = ['sum', 'mean', 'median', 'count', 'min', 'max'] as const;

function isColumnSummary(value: unknown): boolean {
  return isRecord(value)
    && typeof value.name === 'string'
    && (value.display_name === undefined || value.display_name === null || typeof value.display_name === 'string')
    && (value.original_name === undefined || value.original_name === null || typeof value.original_name === 'string')
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

function isInterpretationDecision(value: unknown): boolean {
  return isRecord(value)
    && (value.display_name === undefined || value.display_name === null || typeof value.display_name === 'string')
    && (value.label_evidence_strength === undefined || ['strong', 'tentative', 'unknown'].includes(String(value.label_evidence_strength)))
    && semanticTypes.includes(value.role as SemanticType)
    && typeof value.unit === 'string'
    && typeof value.parsing_policy === 'string'
    && typeof value.recommended_aggregation === 'string'
    && typeof value.needs_clarification === 'boolean';
}

function isInterpretationProposal(value: unknown): boolean {
  return isRecord(value) && typeof value.status === 'string'
    && (value.label_runtime_status === undefined || ['applied', 'clarification'].includes(String(value.label_runtime_status)))
    && (value.decision === null || isInterpretationDecision(value.decision));
}

export function isDatasetState(value: unknown): value is DatasetState {
  return isRecord(value)
    && typeof value.datasetId === 'string'
    && (value.version === undefined || typeof value.version === 'string')
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
    && (value.summary === undefined || value.summary === null || typeof value.summary === 'string')
    && (value.enrichmentStatus === undefined || enrichmentStatuses.includes(value.enrichmentStatus as typeof enrichmentStatuses[number]))
    && (value.interpretationProposals === undefined || (isRecord(value.interpretationProposals)
      && Object.values(value.interpretationProposals).every(isInterpretationProposal)));
}
