import type { ChartResponse, FilterConfig } from '@/types';

/** Combine filters while keeping distinct constraints on the same column. */
export function mergeFilters(...groups: Array<FilterConfig[] | null | undefined>): FilterConfig[] {
  const seen = new Set<string>();
  const merged: FilterConfig[] = [];
  for (const group of groups) {
    for (const filter of group ?? []) {
      const key = JSON.stringify({
        column: filter.column,
        operator: filter.operator,
        value: filter.value,
        values: filter.values,
        min_val: filter.min_val,
        max_val: filter.max_val,
      });
      if (seen.has(key)) continue;
      seen.add(key);
      merged.push(filter);
    }
  }
  return merged;
}

export function preserveQueryProvenance(
  response: ChartResponse,
  previous: ChartResponse,
  effectiveFilters: FilterConfig[],
): ChartResponse {
  return {
    ...response,
    ...(previous.aggregation_scope === 'overall' ? { title: previous.title, x_axis_label: previous.x_axis_label, y_axis_label: previous.y_axis_label } : {}),
    source_x_axis_key: previous.aggregation_scope === 'overall' ? undefined : previous.source_x_axis_key || previous.x_axis_key,
    time_bucket: response.time_bucket ?? null,
    filters: response.filters ?? effectiveFilters,
    llm_filters: previous.llm_filters,
  };
}

export function getNextPeriodStart(value: string, bucket: 'year' | 'month' | 'week'): string | null {
  const hasTimezone = /(?:z|[+-]\d{2}:?\d{2})$/i.test(value);
  const normalizedValue = /^\d{4}-\d{2}-\d{2}$/.test(value)
    ? `${value}T00:00:00.000Z`
    : hasTimezone ? value : `${value}Z`;
  const date = new Date(normalizedValue);
  if (!Number.isFinite(date.getTime())) return null;
  if (bucket === 'year') date.setUTCFullYear(date.getUTCFullYear() + 1);
  else if (bucket === 'month') date.setUTCMonth(date.getUTCMonth() + 1);
  else date.setUTCDate(date.getUTCDate() + 7);
  return date.toISOString();
}

/** Synthetic overall chart keys are presentation fields, never source columns. */
export function getChartAggregationFields(chart: ChartResponse) {
  return {
    x_axis_key: chart.aggregation_scope === 'overall' ? null : chart.source_x_axis_key || chart.x_axis_key,
    y_axis_keys: chart.count_rows ? [] : chart.y_axis_keys,
  };
}
