/**
 * Analytico V4 Type Definitions - Zero Friction Analytics
 */

// Column formats for display
export type ColumnFormat = 'currency' | 'percentage' | 'number' | 'date';

// Semantic types
export type SemanticType = 'metric' | 'identifier' | 'temporal' | 'categorical';

// Metric summary in profile
export interface MetricSummary {
  name: string;
  total: number;
  average: number;
  min: number;
  max: number;
}

// Time range
export interface TimeRange {
  column: string;
  start: string;
  end: string;
}

// Data profile (executive summary)
export interface DataProfile {
  top_metrics: MetricSummary[];
  time_range: TimeRange | null;
  row_count: number;
  column_count: number;
}

// Data health
export interface DataHealth {
  missing_values: Record<string, number>;
  cleaning_actions: string[];
  quality_score: number;
}

// Default chart config
export interface DefaultChart {
  x_axis_key: string;
  y_axis_keys: string[];
  chart_type: ChartType;
  aggregation: AggregationType;
  title: string;
  analysis: string; // Renamed from reasoning
}

// Column metadata
export interface ColumnSummary {
  name: string;
  dtype: string;
  is_numeric: boolean;
  is_datetime: boolean;
  semantic_type: SemanticType;
  format: ColumnFormat;
  unique_count: number;
  sample_values: unknown[];
}

// Upload response with all V4 features
export interface UploadResponse {
  dataset_id: string;
  filename: string;
  row_count: number;
  columns: ColumnSummary[];
  column_formats: Record<string, ColumnFormat>;
  data_health: DataHealth;
  profile: DataProfile;
  default_chart: DefaultChart | null;
  suggestions: string[];
  summary?: string;
}

// Filter configuration
export interface FilterConfig {
  column: string;
  values?: unknown[];
  min_val?: unknown;
  max_val?: unknown;
}

// Aggregation request
export interface AggregateRequest {
  dataset_id: string;
  x_axis_key: string;
  y_axis_keys: string[];
  aggregation: AggregationType;
  chart_type: ChartType;
  filters?: FilterConfig[];
  limit?: number;
  sort_by?: 'value' | 'label';
  group_others?: boolean;
  include_analysis?: boolean;
}

// Chart response with labels and applied filters
export interface ChartResponse {
  data: Record<string, unknown>[];
  x_axis_key: string;
  y_axis_keys: string[];
  chart_type: ChartType;
  title: string;
  aggregation?: AggregationType;
  x_axis_label?: string;
  y_axis_label?: string;
  row_count: number;
  analysis?: string; // 2-sentence business insight (renamed from reasoning)
  warnings?: string[];
  applied_filters?: string[];
  answer?: string;
}

// Drilldown request
export interface DrillDownRequest {
  dataset_id: string;
  filters?: FilterConfig[];
  limit?: number;
}

// Query request
export interface QueryRequest {
  dataset_id: string;
  user_prompt: string;
  filters?: FilterConfig[];
  limit?: number;
  group_others?: boolean;
}

// Chart types
export type ChartType = 'bar' | 'line' | 'area' | 'pie' | 'composed' | 'empty';

// Aggregation types
export type AggregationType = 'sum' | 'mean' | 'count' | 'min' | 'max';

// History item
export interface HistoryItem {
  id: string;
  query: string;
  chartResponse: ChartResponse;
  timestamp: Date;
  isManual: boolean;
}

// View mode
export type ViewMode = 'chart' | 'table';

// Builder mode
export type BuilderMode = 'ai' | 'manual';
