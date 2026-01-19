/**
 * Analytico Type Definitions
 */

export interface DataSummary {
  [key: string]: {
    min?: number;
    max?: number;
    mean?: number;
    type: 'numeric' | 'categorical';
    unique_values?: number;
    sample_values?: string[];
  };
}

export interface UploadResponse {
  data: Record<string, unknown>[];
  columns: string[];
  summary: DataSummary;
  row_count: number;
  truncated: boolean;
}

export interface ChartConfig {
  xAxisKey: string;
  yAxisKeys: string[];
  chartType: 'bar' | 'line' | 'area' | 'composed';
  title: string;
}

export interface QueryRequest {
  user_prompt: string;
  columns: string[];
  data_summary: DataSummary;
}

export interface HistoryItem {
  id: string;
  query: string;
  chartConfig: ChartConfig;
  timestamp: Date;
}
