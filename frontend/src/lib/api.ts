import axios, { AxiosError } from 'axios';
import {
  UploadResponse,
  QueryRequest,
  ChartResponse,
  AggregateRequest,
  DrillDownRequest,
} from '@/types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Error handler
function handleApiError(error: unknown): never {
  if (error instanceof AxiosError && error.response?.data?.detail) {
    throw new Error(error.response.data.detail);
  }
  throw error;
}

/**
 * Validate if a dataset ID still exists in backend memory
 */
export async function validateDataset(datasetId: string): Promise<boolean> {
  try {
    const response = await api.get<{ valid: boolean }>(`/validate/${datasetId}`);
    return response.data.valid;
  } catch {
    return false;
  }
}

/**
 * Upload a CSV file - returns dataset_id and column metadata (no raw data)
 */
export async function uploadCSV(file: File): Promise<UploadResponse> {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post<UploadResponse>('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  } catch (error) {
    handleApiError(error);
  }
}

/**
 * Query the AI to generate a chart configuration and get aggregated data
 */
export async function queryChart(request: QueryRequest): Promise<ChartResponse> {
  try {
    const response = await api.post<ChartResponse>('/query', {
      dataset_id: request.dataset_id,
      user_prompt: request.user_prompt,
      filters: request.filters,
      limit: request.limit,
      group_others: request.group_others,
    });
    return response.data;
  } catch (error) {
    handleApiError(error);
  }
}

/**
 * Aggregate data manually (bypasses AI)
 */
export async function aggregateData(request: AggregateRequest): Promise<ChartResponse> {
  try {
    const response = await api.post<ChartResponse>('/aggregate', {
      dataset_id: request.dataset_id,
      x_axis_key: request.x_axis_key,
      y_axis_keys: request.y_axis_keys,
      aggregation: request.aggregation,
      chart_type: request.chart_type,
      filters: request.filters,
      limit: request.limit,
      sort_by: request.sort_by,
      group_others: request.group_others,
      include_analysis: request.include_analysis,
    });
    return response.data;
  } catch (error) {
    handleApiError(error);
  }
}

/**
 * Preview dataset rows
 */
export async function previewDataset(datasetId: string, limit: number = 100): Promise<{
  data: Record<string, unknown>[];
  total_rows: number;
  showing: number;
}> {
  try {
    const response = await api.get(`/dataset/${datasetId}/preview`, {
      params: { limit },
    });
    return response.data;
  } catch (error) {
    handleApiError(error);
  }
}

/**
 * Drill down into specific data points
 */
export async function drillDown(request: DrillDownRequest): Promise<{ data: Record<string, unknown>[]; total_rows: number; limit: number }> {
  try {
    const response = await api.post('/drilldown', {
      dataset_id: request.dataset_id,
      filters: request.filters,
      limit: request.limit,
    });
    return response.data;
  } catch (error) {
    handleApiError(error);
  }
}

export default api;
