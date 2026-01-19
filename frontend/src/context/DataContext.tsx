'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode, useCallback } from 'react';
import {
  ColumnSummary,
  ChartResponse,
  HistoryItem,
  FilterConfig,
  ViewMode,
  BuilderMode,
  DataHealth,
} from '@/types';

interface DatasetState {
  datasetId: string;
  filename: string;
  rowCount: number;
  columns: ColumnSummary[];
  dataHealth: DataHealth;
}

interface DataContextType {
  // Dataset state
  dataset: DatasetState | null;
  setDataset: (state: DatasetState | null) => void;
  
  // Current chart response
  currentChart: ChartResponse | null;
  setCurrentChart: (chart: ChartResponse | null) => void;
  
  // Filters
  filters: FilterConfig[];
  setFilters: (filters: FilterConfig[]) => void;
  addFilter: (filter: FilterConfig) => void;
  removeFilter: (column: string) => void;
  clearFilters: () => void;
  
  // View & Builder modes
  viewMode: ViewMode;
  setViewMode: (mode: ViewMode) => void;
  builderMode: BuilderMode;
  setBuilderMode: (mode: BuilderMode) => void;
  
  // History
  history: HistoryItem[];
  addToHistory: (query: string, response: ChartResponse, isManual: boolean) => void;
  selectFromHistory: (item: HistoryItem) => void;
  clearHistory: () => void;
  
  // Loading states
  isUploading: boolean;
  setIsUploading: (loading: boolean) => void;
  isQuerying: boolean;
  setIsQuerying: (loading: boolean) => void;
  
  // Column helpers
  numericColumns: ColumnSummary[];
  categoricalColumns: ColumnSummary[];
  metricColumns: ColumnSummary[];
  temporalColumns: ColumnSummary[];
  
  // Clear
  clearData: () => void;
}

const DataContext = createContext<DataContextType | undefined>(undefined);

const DATASET_KEY = 'analytico_dataset_v3';
const HISTORY_KEY = 'analytico_history_v3';

export function DataProvider({ children }: { children: ReactNode }) {
  const [dataset, setDatasetInternal] = useState<DatasetState | null>(null);
  const [currentChart, setCurrentChart] = useState<ChartResponse | null>(null);
  const [filters, setFiltersInternal] = useState<FilterConfig[]>([]);
  const [viewMode, setViewMode] = useState<ViewMode>('chart');
  const [builderMode, setBuilderMode] = useState<BuilderMode>('ai');
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
    try {
      const savedDataset = localStorage.getItem(DATASET_KEY);
      if (savedDataset) {
        setDatasetInternal(JSON.parse(savedDataset));
      }
      
      const savedHistory = localStorage.getItem(HISTORY_KEY);
      if (savedHistory) {
        const parsed = JSON.parse(savedHistory);
        setHistory(parsed.map((item: HistoryItem) => ({
          ...item,
          timestamp: new Date(item.timestamp)
        })));
      }
    } catch (error) {
      console.error('Error loading from localStorage:', error);
    }
  }, []);

  const setDataset = useCallback((state: DatasetState | null) => {
    setDatasetInternal(state);
    setCurrentChart(null);
    setFiltersInternal([]);
    if (typeof window !== 'undefined') {
      if (state) {
        localStorage.setItem(DATASET_KEY, JSON.stringify(state));
      } else {
        localStorage.removeItem(DATASET_KEY);
      }
    }
  }, []);

  useEffect(() => {
    if (isClient && history.length > 0) {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    }
  }, [history, isClient]);

  const setFilters = useCallback((newFilters: FilterConfig[]) => {
    setFiltersInternal(newFilters);
  }, []);

  const addFilter = useCallback((filter: FilterConfig) => {
    setFiltersInternal(prev => {
      const existing = prev.findIndex(f => f.column === filter.column);
      if (existing >= 0) {
        const updated = [...prev];
        updated[existing] = filter;
        return updated;
      }
      return [...prev, filter];
    });
  }, []);

  const removeFilter = useCallback((column: string) => {
    setFiltersInternal(prev => prev.filter(f => f.column !== column));
  }, []);

  const clearFilters = useCallback(() => {
    setFiltersInternal([]);
  }, []);

  const addToHistory = useCallback((query: string, response: ChartResponse, isManual: boolean) => {
    const newItem: HistoryItem = {
      id: crypto.randomUUID(),
      query,
      chartResponse: response,
      timestamp: new Date(),
      isManual,
    };
    setHistory(prev => [newItem, ...prev].slice(0, 20));
  }, []);

  const selectFromHistory = useCallback((item: HistoryItem) => {
    setCurrentChart(item.chartResponse);
    setViewMode('chart');
  }, []);

  const clearHistory = useCallback(() => {
    setHistory([]);
    if (typeof window !== 'undefined') {
      localStorage.removeItem(HISTORY_KEY);
    }
  }, []);

  const clearData = useCallback(() => {
    setDatasetInternal(null);
    setCurrentChart(null);
    setFiltersInternal([]);
    if (typeof window !== 'undefined') {
      localStorage.removeItem(DATASET_KEY);
    }
  }, []);

  // Column helpers based on semantic types
  const numericColumns = dataset?.columns.filter(c => c.is_numeric) ?? [];
  const categoricalColumns = dataset?.columns.filter(c => c.semantic_type === 'categorical') ?? [];
  const metricColumns = dataset?.columns.filter(c => c.semantic_type === 'metric') ?? [];
  const temporalColumns = dataset?.columns.filter(c => c.semantic_type === 'temporal') ?? [];

  return (
    <DataContext.Provider
      value={{
        dataset,
        setDataset,
        currentChart,
        setCurrentChart,
        filters,
        setFilters,
        addFilter,
        removeFilter,
        clearFilters,
        viewMode,
        setViewMode,
        builderMode,
        setBuilderMode,
        history,
        addToHistory,
        selectFromHistory,
        clearHistory,
        isUploading,
        setIsUploading,
        isQuerying,
        setIsQuerying,
        numericColumns,
        categoricalColumns,
        metricColumns,
        temporalColumns,
        clearData,
      }}
    >
      {children}
    </DataContext.Provider>
  );
}

export function useData() {
  const context = useContext(DataContext);
  if (context === undefined) {
    throw new Error('useData must be used within a DataProvider');
  }
  return context;
}
