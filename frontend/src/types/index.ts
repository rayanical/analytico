/**
 * Analytico V3 Type Definitions
 */

// Semantic types
export type SemanticType = 'metric' | 'identifier' | 'temporal' | 'categorical';

// Data health from upload
export interface DataHealth {
  missing_values: Record<string, number>;
  cleaning_actions: string[];
  quality_score: number;
}

// Column metadata from upload
export interface ColumnSummary {
  name: string;
  dtype: string;
  is_numeric: boolean;
  is_datetime: boolean;
  semantic_type: SemanticType;
  unique_count: number;
  sample_values: unknown[];
  min_val?: number;
  max_val?: number;
  mean_val?: number;
}

// Upload response with health scorecard
export interface UploadResponse {
  dataset_id: string;
  filename: string;
  row_count: number;
  columns: ColumnSummary[];
  data_health: DataHealth;
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
}

// Chart response with reasoning
export interface ChartResponse {
  data: Record<string, unknown>[];
  x_axis_key: string;
  y_axis_keys: string[];
  chart_type: ChartType;
  title: string;
  row_count: number;
  reasoning?: string;
  warnings?: string[];
}

// Query request
export interface QueryRequest {
  dataset_id: string;
  user_prompt: string;
  filters?: FilterConfig[];
}

// Chart types
export type ChartType = 'bar' | 'line' | 'area' | 'pie' | 'composed';

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
