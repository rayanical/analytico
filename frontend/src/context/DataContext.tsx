'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode, useCallback } from 'react';
import {
  ColumnSummary,
  ChartResponse,
  HistoryItem,
  FilterConfig,
  ViewMode,
  BuilderMode,
} from '@/types';

interface DatasetState {
  datasetId: string;
  filename: string;
  rowCount: number;
  columns: ColumnSummary[];
}

interface DataContextType {
  // Dataset state (no raw data, just metadata)
  dataset: DatasetState | null;
  setDataset: (state: DatasetState | null) => void;
  
  // Current chart response (aggregated data)
  currentChart: ChartResponse | null;
  setCurrentChart: (chart: ChartResponse | null) => void;
  
  // Filters
  filters: FilterConfig[];
  setFilters: (filters: FilterConfig[]) => void;
  addFilter: (filter: FilterConfig) => void;
  removeFilter: (column: string) => void;
  clearFilters: () => void;
  
  // View mode (chart vs table)
  viewMode: ViewMode;
  setViewMode: (mode: ViewMode) => void;
  
  // Builder mode (AI vs manual)
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
  
  // Helpers
  numericColumns: ColumnSummary[];
  categoricalColumns: ColumnSummary[];
  dateColumns: ColumnSummary[];
  
  // Clear all
  clearData: () => void;
}

const DataContext = createContext<DataContextType | undefined>(undefined);

const DATASET_KEY = 'analytico_dataset';
const HISTORY_KEY = 'analytico_history';

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

  // Load from localStorage on mount
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

  // Save dataset to localStorage
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

  // Save history to localStorage
  useEffect(() => {
    if (isClient && history.length > 0) {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    }
  }, [history, isClient]);

  // Filter management
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

  // History management
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

  // Computed column helpers
  const numericColumns = dataset?.columns.filter(c => c.is_numeric) ?? [];
  const categoricalColumns = dataset?.columns.filter(c => !c.is_numeric && !c.is_datetime) ?? [];
  const dateColumns = dataset?.columns.filter(c => c.is_datetime) ?? [];

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
        dateColumns,
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
