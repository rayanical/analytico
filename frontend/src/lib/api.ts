import axios, { AxiosError } from 'axios';
import type {
  ApiChartResponse,
  ApiFilterConfig,
  FilterConfig,
  UploadResponse,
  QueryRequest,
  ChartResponse,
  AggregateRequest,
  DrillDownRequest,
  EnrichmentStatusResponse,
  ImportPreviewResponse,
  ImportSettings,
  DatasetSchemaResponse,
  ColumnSchemaOverride,
  VersionedUploadResponse,
} from '@/types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

function normalizeFilter(filter: ApiFilterConfig): FilterConfig {
  return {
    column: filter.column,
    ...(filter.operator !== null ? { operator: filter.operator } : {}),
    ...(filter.value !== null ? { value: filter.value } : {}),
    ...(filter.values !== null ? { values: filter.values } : {}),
    ...(filter.min_val !== null ? { min_val: filter.min_val } : {}),
    ...(filter.max_val !== null ? { max_val: filter.max_val } : {}),
  };
}

export function normalizeChartResponse(response: ApiChartResponse): ChartResponse {
  return {
    ...response,
    aggregation: response.aggregation ?? undefined,
    x_axis_label: response.x_axis_label ?? undefined,
    y_axis_label: response.y_axis_label ?? undefined,
    analysis: response.analysis ?? undefined,
    warnings: response.warnings ?? undefined,
    applied_filters: response.applied_filters ?? undefined,
    filters: response.filters?.map(normalizeFilter),
    llm_filters: response.llm_filters?.map(normalizeFilter),
    source_x_axis_key: response.source_x_axis_key ?? undefined,
    others_label: response.others_label ?? undefined,
    answer: response.answer ?? undefined,
  };
}

// Error handler
function handleApiError(error: unknown): never {
  if (error instanceof AxiosError) {
    const detail: unknown = error.response?.data?.detail;
    if (typeof detail === 'string') throw new Error(detail);
    if (Array.isArray(detail)) {
      const messages = detail.map((item: unknown) => {
        if (typeof item === 'string') return item;
        if (!item || typeof item !== 'object') return '';
        const issue = item as { msg?: unknown; loc?: unknown };
        const message = typeof issue.msg === 'string' ? issue.msg : '';
        const location = Array.isArray(issue.loc) ? issue.loc.filter(part => typeof part === 'string' || typeof part === 'number').join('.') : '';
        return message ? (location ? `${location}: ${message}` : message) : '';
      }).filter(Boolean);
      if (messages.length > 0) throw new Error(messages.join('; '));
    }
    if (typeof detail === 'object' && detail !== null && 'message' in detail && typeof detail.message === 'string') {
      throw new Error(detail.message);
    }
    if (error.response) throw new Error(`Request failed (${error.response.status})`);
  }
  throw error;
}

/**
 * Validate if a dataset ID still exists in backend memory
 */
export type DatasetValidation = 'valid' | 'expired' | 'unavailable';

export async function validateDataset(datasetId: string): Promise<DatasetValidation> {
  try {
    const response = await api.get<{ valid: boolean }>(`/validate/${datasetId}`);
    return response.data.valid ? 'valid' : 'expired';
  } catch (error) {
    if (error instanceof AxiosError && error.response?.status === 404) return 'expired';
    return 'unavailable';
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

/** Stage a CSV and return a parsing preview before creating a dataset. */
export async function previewImport(file: File, settings?: ImportSettings): Promise<ImportPreviewResponse> {
  try {
    const formData = new FormData();
    formData.append('file', file);
    if (settings) formData.append('settings', JSON.stringify(settings));
    const response = await api.post<ImportPreviewResponse>('/imports/preview', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  } catch (error) {
    handleApiError(error);
  }
}

/** Stage a built-in demo dataset through the same review flow as file uploads. */
export async function previewDemoImport(dataset: 'taxi' | 'gapminder'): Promise<ImportPreviewResponse> {
  try {
    const response = await api.post<ImportPreviewResponse>('/imports/demo', null, { params: { dataset } });
    return response.data;
  } catch (error) {
    handleApiError(error);
  }
}

/** Re-parse a staged upload with edited settings; the original file stays on the server. */
export async function updateImportPreview(importId: string, settings: ImportSettings): Promise<ImportPreviewResponse> {
  try {
    const response = await api.post<ImportPreviewResponse>(
      `/imports/${encodeURIComponent(importId)}/preview`,
      { settings },
    );
    return response.data;
  } catch (error) {
    handleApiError(error);
  }
}

/** Confirm a staged import and create a versioned dataset. */
export async function confirmImport(
  importId: string,
  settings: ImportSettings,
): Promise<VersionedUploadResponse> {
  try {
    const response = await api.post<VersionedUploadResponse>(
      `/imports/${encodeURIComponent(importId)}/confirm`,
      { settings, column_overrides: [] },
    );
    return response.data;
  } catch (error) {
    handleApiError(error);
  }
}

/** Discard a staged import without creating a dataset. */
export async function cancelImport(importId: string): Promise<void> {
  try {
    await api.delete(`/imports/${encodeURIComponent(importId)}`);
  } catch (error) {
    handleApiError(error);
  }
}

/** Fetch active parse settings, schema, preview rows, and advisory proposals. */
export async function getDatasetSchema(datasetId: string, signal?: AbortSignal): Promise<DatasetSchemaResponse> {
  try {
    const response = await api.get<DatasetSchemaResponse>(`/datasets/${encodeURIComponent(datasetId)}/schema`, { signal });
    return response.data;
  } catch (error) {
    handleApiError(error);
  }
}

/** Apply reviewed column settings against the version the user inspected. */
export async function applyDatasetSchema(
  datasetId: string,
  expectedVersion: string,
  columnOverrides: ColumnSchemaOverride[],
  signal?: AbortSignal,
): Promise<VersionedUploadResponse> {
  try {
    const response = await api.post<VersionedUploadResponse>(
      `/datasets/${encodeURIComponent(datasetId)}/schema`,
      { expected_version: expectedVersion, column_overrides: columnOverrides },
      { signal },
    );
    return response.data;
  } catch (error) {
    handleApiError(error);
  }
}

/**
 * Load the built-in demo dataset from backend storage
 */
export async function loadDemoDataset(dataset: 'taxi' | 'gapminder' = 'taxi'): Promise<UploadResponse> {
  try {
    const response = await api.post<UploadResponse>('/load-demo', null, {
      params: { dataset },
    });
    return response.data;
  } catch (error) {
    handleApiError(error);
  }
}

/** Poll the optional, dataset-scoped enrichment job. */
export async function getEnrichmentStatus(
  datasetId: string,
  signal?: AbortSignal,
): Promise<EnrichmentStatusResponse> {
  try {
    const response = await api.get<EnrichmentStatusResponse>(
      `/enrichment/${encodeURIComponent(datasetId)}`,
      { signal },
    );
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
    const response = await api.post<ApiChartResponse>('/query', {
      dataset_id: request.dataset_id,
      user_prompt: request.user_prompt,
      filters: request.filters,
      limit: request.limit,
      sort_by: request.sort_by,
      group_others: request.group_others,
    }, { signal: request.signal });
    return normalizeChartResponse(response.data);
  } catch (error) {
    handleApiError(error);
  }
}

/**
 * Aggregate data manually (bypasses AI)
 */
export async function aggregateData(request: AggregateRequest): Promise<ChartResponse> {
  try {
    const response = await api.post<ApiChartResponse>('/aggregate', {
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
      time_bucket: request.time_bucket,
    }, { signal: request.signal });
    return normalizeChartResponse(response.data);
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
    }, { signal: request.signal });
    return response.data;
  } catch (error) {
    handleApiError(error);
  }
}

export default api;
