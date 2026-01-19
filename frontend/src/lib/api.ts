import axios from 'axios';
import { ChartConfig, QueryRequest, UploadResponse } from '@/types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Upload a CSV file and get parsed data with statistics
 */
export async function uploadCSV(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post<UploadResponse>('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
}

/**
 * Query the AI to generate a chart configuration
 */
export async function queryChart(request: QueryRequest): Promise<ChartConfig> {
  const response = await api.post<ChartConfig>('/query', request);
  return response.data;
}

export default api;
