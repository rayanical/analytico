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

// Optional provider provenance; absent for keyless interpretation.
export interface ColumnInterpretationMetadata {
  status: 'ok' | 'disabled' | 'unavailable' | 'error' | 'uncertain' | 'skipped';
  runtime_status: 'applied' | 'clarification';
  provider: string;
  model?: string | null;
  prompt_version?: string | null;
  latency_ms?: number | null;
  cache_hit?: boolean;
  usage: Record<string, unknown>;
  confidence?: Record<string, unknown> | null;
  error_code?: string | null;
  decision?: {
    role: SemanticType;
    unit: string;
    parsing_policy: string;
    recommended_aggregation: string;
    needs_clarification: boolean;
  } | null;
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
  interpretation?: ColumnInterpretationMetadata | null;
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
  enrichment_status?: EnrichmentStatus;
}

export type ImportDelimiter = ',' | ';' | '\t' | '|';
export type ImportEncoding = 'utf-8-sig' | 'utf-8' | 'cp1252' | 'utf-16';
export type DecimalSeparator = 'auto' | '.' | ',';
export type GroupingSeparator = ',' | '.' | ' ' | null;
export type DateOrder = 'auto' | 'ymd' | 'dmy' | 'mdy';

/** Parsing options used while previewing or confirming an imported file. */
export interface ImportSettings {
  delimiter: ImportDelimiter;
  encoding: ImportEncoding;
  decimal_separator: DecimalSeparator;
  grouping_separator: GroupingSeparator;
  date_order: DateOrder;
  null_values: string[];
}

export interface ImportPreviewColumn {
  name: string;
  original_name: string;
  dtype: string;
  semantic_type: SemanticType;
  format: ColumnFormat;
  sample_values: unknown[];
}

/** Staged import preview. Rows are a sample; full-file validation runs on confirmation. */
export interface ImportPreviewResponse {
  import_id: string;
  filename: string;
  settings: ImportSettings;
  columns: ImportPreviewColumn[];
  raw_rows: (string | null)[][];
  parsed_rows: Record<string, unknown>[];
  warnings: string[];
  sample_row_count: number;
  sample_complete: boolean;
  can_confirm: boolean;
}

export type ParseAs = 'auto' | 'text' | 'number' | 'date';
export type ColumnAggregation = 'sum' | 'mean' | 'count' | 'none';

export interface DatasetColumnSchema {
  column: string;
  original_name: string;
  parse_as?: ParseAs;
  role: SemanticType | null;
  format: ColumnFormat | null;
  unit: string | null;
  aggregation: ColumnAggregation | null;
  provenance: string;
  status?: string;
}

export interface ColumnSchemaOverride {
  column: string;
  parse_as?: ParseAs;
  role?: SemanticType | null;
  format?: ColumnFormat | null;
  unit?: string | null;
  aggregation?: ColumnAggregation | null;
}

export interface DatasetColumnProposalDecision {
  role?: SemanticType;
  unit?: string;
  parsing_policy?: string;
  recommended_aggregation?: string;
  needs_clarification?: boolean;
}

export interface DatasetColumnProposal {
  status?: string;
  decision?: DatasetColumnProposalDecision | null;
}

export interface DatasetSchemaResponse {
  dataset_id: string;
  version: string;
  settings: ImportSettings;
  columns: DatasetColumnSchema[];
  preview: {
    columns: string[];
    rows: Record<string, unknown>[];
  };
  proposals: Record<string, DatasetColumnProposal>;
}

export type VersionedUploadResponse = UploadResponse & { version: string; column_schema?: DatasetColumnSchema[] };

export type EnrichmentStatus = 'pending' | 'running' | 'done' | 'error' | 'disabled';

/** Optional model suggestions that remain separate from active column types. */
export interface InterpretationProposal {
  role: SemanticType;
  unit: string;
  parsing_policy: string;
  recommended_aggregation: string;
  needs_clarification: boolean;
}

export interface EnrichmentStatusResponse {
  dataset_id: string;
  version?: string | null;
  status: EnrichmentStatus;
  progress: number;
  summary: string | null;
  interpretation_proposals: Record<string, InterpretationProposal>;
  error: string | null;
  reason?: string | null;
  coverage?: {
    total_columns: number;
    selected_columns: number;
    completed_columns: number;
    failed_columns: number;
    skipped_columns: number;
    complete: boolean;
    stop_reason: string | null;
  } | null;
}

/** Dataset metadata retained in local storage between visits. */
export interface DatasetState {
  datasetId: string;
  version?: string;
  filename: string;
  rowCount: number;
  columns: ColumnSummary[];
  columnFormats: Record<string, ColumnFormat>;
  dataHealth: DataHealth;
  profile: DataProfile;
  defaultChart: DefaultChart | null;
  suggestions: string[];
  summary?: string | null;
  enrichmentStatus?: EnrichmentStatus;
  interpretationProposals?: Record<string, InterpretationProposal>;
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
  datasetVersion?: string;
  /** Derived by DataContext; prior-schema charts are retained but disabled. */
  isStale?: boolean;
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
