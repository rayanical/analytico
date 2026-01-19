/**
 * Analytico V2 Type Definitions
 */

// Column metadata from upload
export interface ColumnSummary {
  name: string;
  dtype: string;
  is_numeric: boolean;
  is_datetime: boolean;
  unique_count: number;
  sample_values: unknown[];
  min_val?: number;
  max_val?: number;
  mean_val?: number;
}

// Upload response (no raw data, just metadata)
export interface UploadResponse {
  dataset_id: string;
  filename: string;
  row_count: number;
  columns: ColumnSummary[];
}

// Filter configuration
export interface FilterConfig {
  column: string;
  values?: unknown[];    // For categorical
  min_val?: unknown;     // For range
  max_val?: unknown;     // For range
}

// Aggregation request
export interface AggregateRequest {
  dataset_id: string;
  x_axis_key: string;
  y_axis_keys: string[];
  aggregation: 'sum' | 'mean' | 'count' | 'min' | 'max';
  chart_type: ChartType;
  filters?: FilterConfig[];
}

// Chart response (from both /aggregate and /query)
export interface ChartResponse {
  data: Record<string, unknown>[];
  x_axis_key: string;
  y_axis_keys: string[];
  chart_type: ChartType;
  title: string;
  row_count: number;
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
