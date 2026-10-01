/**
 * Analytico V4 Type Definitions - Zero Friction Analytics
 */

// Column formats for display
export type ColumnFormat = 'currency' | 'percentage' | 'number' | 'date' | 'identifier' | 'general';

// Semantic types
export type SemanticType = 'metric' | 'identifier' | 'temporal' | 'categorical' | 'unknown';
export type FilterOperator = 'eq' | 'gt' | 'lt' | 'gte' | 'lte' | 'contains';
export type FilterValue = string | number | boolean;

// Metric summary in profile
export interface MetricSummary {
  name: string;
  total: number;
  average: number;
  aggregation?: 'sum' | 'mean';
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
  chart_type: DefaultChartType;
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
  summary?: string | null;
}

/** Dataset metadata retained in local storage between visits. */
export interface DatasetState {
  datasetId: string;
  filename: string;
  rowCount: number;
  columns: ColumnSummary[];
  columnFormats: Record<string, ColumnFormat>;
  dataHealth: DataHealth;
  profile: DataProfile;
  defaultChart: DefaultChart | null;
  suggestions: string[];
  summary?: string | null;
}

// Filter configuration
export interface FilterConfig {
  column: string;
  operator?: FilterOperator;
  value?: FilterValue;
  values?: (FilterValue | null)[];
  min_val?: FilterValue;
  max_val?: FilterValue;
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
  time_bucket?: 'year' | 'month' | 'week' | null;
  signal?: AbortSignal;
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
  llm_filters?: FilterConfig[];
  /** Effective query filters, including both UI and AI filters. */
  filters?: FilterConfig[];
  /** Original date column used before optional period bucketing. */
  source_x_axis_key?: string;
  time_bucket?: 'year' | 'month' | 'week' | null;
  others_label?: string;
  answer?: string;
}

/** Exact FastAPI chart payload before nullable fields are normalized for UI state. */
export interface ApiFilterConfig {
  column: string;
  operator: FilterOperator | null;
  value: FilterValue | null;
  values: (FilterValue | null)[] | null;
  min_val: FilterValue | null;
  max_val: FilterValue | null;
}

type NullableChartFields =
  | 'aggregation' | 'x_axis_label' | 'y_axis_label' | 'analysis' | 'warnings'
  | 'applied_filters' | 'filters' | 'llm_filters' | 'source_x_axis_key'
  | 'time_bucket' | 'others_label' | 'answer';

export interface ApiChartResponse extends Omit<ChartResponse, NullableChartFields> {
  aggregation: AggregationType | null;
  x_axis_label: string | null;
  y_axis_label: string | null;
  analysis: string | null;
  warnings: string[] | null;
  applied_filters: string[] | null;
  filters: ApiFilterConfig[] | null;
  llm_filters: ApiFilterConfig[] | null;
  source_x_axis_key: string | null;
  time_bucket: 'year' | 'month' | 'week' | null;
  others_label: string | null;
  answer: string | null;
}

// Drilldown request
export interface DrillDownRequest {
  dataset_id: string;
  filters?: FilterConfig[];
  limit?: number;
  signal?: AbortSignal;
}

// Query request
export interface QueryRequest {
  dataset_id: string;
  user_prompt: string;
  filters?: FilterConfig[];
  limit?: number;
  sort_by?: 'value' | 'label';
  group_others?: boolean;
  signal?: AbortSignal;
}

// Chart types
export type ChartType = 'bar' | 'line' | 'area' | 'pie' | 'composed' | 'empty';
export type DefaultChartType = Exclude<ChartType, 'empty'>;

// Aggregation types
export type AggregationType = 'sum' | 'mean' | 'median' | 'count' | 'min' | 'max';

// History item
export interface HistoryItem {
  id: string;
  datasetId: string;
  query: string;
  chartResponse: ChartResponse;
  timestamp: Date;
  isManual: boolean;
}

// View mode
export type ViewMode = 'chart' | 'table';

// Builder mode
export type BuilderMode = 'ai' | 'manual';

// Workspace mode
export type WorkspaceMode = 'explore' | 'dashboard';

// Dashboard grid item layout
export interface DashboardLayoutItem {
  i: string;
  x: number;
  y: number;
  w: number;
  h: number;
  minW?: number;
  minH?: number;
  maxW?: number;
}

export interface DashboardUiState {
  isHeaderCollapsed: boolean;
  headerCollapseMode: 'auto' | 'manual';
}

// Pinned dashboard widget
export interface DashboardWidget {
  id: string;
  title: string;
  chart: ChartResponse;
  sourceQuery: string;
  createdAt: string;
  datasetId: string;
  layout: DashboardLayoutItem;
  placementStrategy?: 'first-row-tile' | 'next-fit';
}
