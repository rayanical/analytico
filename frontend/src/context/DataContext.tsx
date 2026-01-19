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
  DataProfile,
  DefaultChart,
  ColumnFormat,
} from '@/types';

interface DatasetState {
  datasetId: string;
  filename: string;
  rowCount: number;
  columns: ColumnSummary[];
  columnFormats: Record<string, ColumnFormat>;
  dataHealth: DataHealth;
  profile: DataProfile;
  defaultChart: DefaultChart | null;
  suggestions: string[];
}

interface DataContextType {
  dataset: DatasetState | null;
  setDataset: (state: DatasetState | null) => void;
  currentChart: ChartResponse | null;
  setCurrentChart: (chart: ChartResponse | null) => void;
  filters: FilterConfig[];
  setFilters: (filters: FilterConfig[]) => void;
  addFilter: (filter: FilterConfig) => void;
  removeFilter: (column: string) => void;
  clearFilters: () => void;
  viewMode: ViewMode;
  setViewMode: (mode: ViewMode) => void;
  builderMode: BuilderMode;
  setBuilderMode: (mode: BuilderMode) => void;
  history: HistoryItem[];
  addToHistory: (query: string, response: ChartResponse, isManual: boolean) => void;
  selectFromHistory: (item: HistoryItem) => void;
  clearHistory: () => void;
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

const DATASET_KEY = 'analytico_dataset_v4';
const HISTORY_KEY = 'analytico_history_v4';

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
      const saved = localStorage.getItem(DATASET_KEY);
      if (saved) setDatasetInternal(JSON.parse(saved));
      const hist = localStorage.getItem(HISTORY_KEY);
      if (hist) {
        setHistory(JSON.parse(hist).map((i: HistoryItem) => ({ ...i, timestamp: new Date(i.timestamp) })));
      }
    } catch (e) {
      console.error('Load error:', e);
    }
  }, []);

  const setDataset = useCallback((state: DatasetState | null) => {
    setDatasetInternal(state);
    setCurrentChart(null);
    setFiltersInternal([]);
    if (typeof window !== 'undefined') {
      if (state) localStorage.setItem(DATASET_KEY, JSON.stringify(state));
      else localStorage.removeItem(DATASET_KEY);
    }
  }, []);

  useEffect(() => {
    if (isClient && history.length > 0) {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    }
  }, [history, isClient]);

  const setFilters = useCallback((f: FilterConfig[]) => setFiltersInternal(f), []);
  const addFilter = useCallback((f: FilterConfig) => {
    setFiltersInternal(prev => {
      const idx = prev.findIndex(x => x.column === f.column);
      if (idx >= 0) { const u = [...prev]; u[idx] = f; return u; }
      return [...prev, f];
    });
  }, []);
  const removeFilter = useCallback((col: string) => setFiltersInternal(prev => prev.filter(f => f.column !== col)), []);
  const clearFilters = useCallback(() => setFiltersInternal([]), []);

  const addToHistory = useCallback((query: string, response: ChartResponse, isManual: boolean) => {
    setHistory(prev => [{ id: crypto.randomUUID(), query, chartResponse: response, timestamp: new Date(), isManual }, ...prev].slice(0, 20));
  }, []);
  const selectFromHistory = useCallback((item: HistoryItem) => { setCurrentChart(item.chartResponse); setViewMode('chart'); }, []);
  const clearHistory = useCallback(() => { setHistory([]); if (typeof window !== 'undefined') localStorage.removeItem(HISTORY_KEY); }, []);
  const clearData = useCallback(() => { setDatasetInternal(null); setCurrentChart(null); setFiltersInternal([]); if (typeof window !== 'undefined') localStorage.removeItem(DATASET_KEY); }, []);

  const numericColumns = dataset?.columns.filter(c => c.is_numeric) ?? [];
  const categoricalColumns = dataset?.columns.filter(c => c.semantic_type === 'categorical') ?? [];
  const metricColumns = dataset?.columns.filter(c => c.semantic_type === 'metric') ?? [];
  const temporalColumns = dataset?.columns.filter(c => c.semantic_type === 'temporal') ?? [];

  return (
    <DataContext.Provider value={{
      dataset, setDataset, currentChart, setCurrentChart, filters, setFilters, addFilter, removeFilter, clearFilters,
      viewMode, setViewMode, builderMode, setBuilderMode, history, addToHistory, selectFromHistory, clearHistory,
      isUploading, setIsUploading, isQuerying, setIsQuerying, numericColumns, categoricalColumns, metricColumns, temporalColumns, clearData,
    }}>
      {children}
    </DataContext.Provider>
  );
}

export function useData() {
  const ctx = useContext(DataContext);
  if (!ctx) throw new Error('useData must be within DataProvider');
  return ctx;
}
