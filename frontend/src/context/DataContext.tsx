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
  summary?: string;
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
  currentHistoryId: string | null;
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
  // Drill Down
  drillDownData: any[] | null;
  setDrillDownData: (data: any[] | null) => void;
  isDrillDownOpen: boolean;
  setIsDrillDownOpen: (open: boolean) => void;
  // Global Settings
  groupOthers: boolean;
  setGroupOthers: (group: boolean) => void;
  limit: number;
  setLimit: (limit: number) => void;
  sortBy: 'value' | 'label';
  setSortBy: (sort: 'value' | 'label') => void;
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
  const [currentHistoryId, setCurrentHistoryId] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [drillDownData, setDrillDownData] = useState<any[] | null>(null);
  const [isDrillDownOpen, setIsDrillDownOpen] = useState(false);
  const [groupOthers, setGroupOthers] = useState(true);
  const [limit, setLimit] = useState(20);
  const [sortBy, setSortBy] = useState<'value' | 'label'>('value');
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
    const loadAndValidate = async () => {
      try {
        const saved = localStorage.getItem(DATASET_KEY);
        const hist = localStorage.getItem(HISTORY_KEY);
        
        if (hist) {
          setHistory(JSON.parse(hist).map((i: HistoryItem) => ({ ...i, timestamp: new Date(i.timestamp) })));
        }
        
        if (saved) {
          const parsedDataset = JSON.parse(saved);
          // Validate that dataset still exists in backend
          const { validateDataset } = await import('@/lib/api');
          const isValid = await validateDataset(parsedDataset.datasetId);
          
          if (isValid) {
            setDatasetInternal(parsedDataset);
          } else {
            // Dataset expired or server restarted - clear stale data
            localStorage.removeItem(DATASET_KEY);
            console.log('Dataset expired - please re-upload');
          }
        }
      } catch (e) {
        console.error('Load error:', e);
      }
    };
    loadAndValidate();
  }, []);

  const setDataset = useCallback((state: DatasetState | null) => {
    setDatasetInternal(state);
    setCurrentChart(null);
    setCurrentHistoryId(null);
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
    const entry: HistoryItem = {
      id: crypto.randomUUID(),
      query,
      chartResponse: response,
      timestamp: new Date(),
      isManual,
    };
    setHistory(prev => [entry, ...prev].slice(0, 30));
    setCurrentHistoryId(entry.id);
  }, []);
  const selectFromHistory = useCallback((item: HistoryItem) => {
    setCurrentChart(item.chartResponse);
    setCurrentHistoryId(item.id);
    setViewMode('chart');
  }, []);
  const clearHistory = useCallback(() => {
    setHistory([]);
    setCurrentHistoryId(null);
    if (typeof window !== 'undefined') localStorage.removeItem(HISTORY_KEY);
  }, []);
  const clearData = useCallback(() => {
    setDatasetInternal(null);
    setCurrentChart(null);
    setCurrentHistoryId(null);
    setFiltersInternal([]);
    if (typeof window !== 'undefined') localStorage.removeItem(DATASET_KEY);
  }, []);

  const numericColumns = dataset?.columns.filter(c => c.is_numeric) ?? [];
  const categoricalColumns = dataset?.columns.filter(c => c.semantic_type === 'categorical') ?? [];
  const metricColumns = dataset?.columns.filter(c => c.semantic_type === 'metric') ?? [];
  const temporalColumns = dataset?.columns.filter(c => c.semantic_type === 'temporal') ?? [];

  return (
    <DataContext.Provider value={{
      dataset, setDataset, currentChart, setCurrentChart, filters, setFilters, addFilter, removeFilter, clearFilters,
      viewMode, setViewMode, builderMode, setBuilderMode, history, currentHistoryId, addToHistory, selectFromHistory, clearHistory,
      isUploading, setIsUploading, isQuerying, setIsQuerying, numericColumns, categoricalColumns, metricColumns, temporalColumns, clearData,
      drillDownData, setDrillDownData, isDrillDownOpen, setIsDrillDownOpen,
      groupOthers, setGroupOthers, limit, setLimit, sortBy, setSortBy,
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
