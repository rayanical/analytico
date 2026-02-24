'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode, useCallback } from 'react';
import {
  ColumnSummary,
  ChartResponse,
  HistoryItem,
  FilterConfig,
  ViewMode,
  BuilderMode,
  WorkspaceMode,
  DataHealth,
  DataProfile,
  DefaultChart,
  ColumnFormat,
  DashboardWidget,
  DashboardLayoutItem,
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
  workspaceMode: WorkspaceMode;
  setWorkspaceMode: (mode: WorkspaceMode) => void;
  history: HistoryItem[];
  currentHistoryId: string | null;
  addToHistory: (query: string, response: ChartResponse, isManual: boolean) => void;
  selectFromHistory: (item: HistoryItem) => void;
  clearHistory: () => void;
  dashboardWidgets: DashboardWidget[];
  pinCurrentChart: (chart: ChartResponse, sourceQuery?: string) => void;
  removeWidget: (widgetId: string) => void;
  updateWidgetLayout: (layouts: DashboardLayoutItem[]) => void;
  clearDashboard: () => void;
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
const DASHBOARD_KEY = 'analytico_dashboard_v1';

function intersects(a: DashboardLayoutItem, b: DashboardLayoutItem): boolean {
  return !(a.x + a.w <= b.x || b.x + b.w <= a.x || a.y + a.h <= b.y || b.y + b.h <= a.y);
}

function getDefaultWidgetDimensions(chartType: ChartResponse['chart_type']) {
  if (chartType === 'pie') return { w: 8, h: 10, minW: 6, minH: 9 };
  if (chartType === 'line' || chartType === 'area') return { w: 14, h: 10, minW: 8, minH: 9 };
  if (chartType === 'composed') return { w: 16, h: 11, minW: 10, minH: 9 };
  return { w: 12, h: 10, minW: 7, minH: 9 };
}

function findNextLayoutSlot(
  chartType: ChartResponse['chart_type'],
  occupied: DashboardLayoutItem[],
): DashboardLayoutItem {
  const cols = 24;
  const { w, h, minW, minH } = getDefaultWidgetDimensions(chartType);
  for (let y = 0; y < 1000; y += 1) {
    for (let x = 0; x <= cols - w; x += 1) {
      const candidate: DashboardLayoutItem = { i: '', x, y, w, h, minW, minH };
      const hasCollision = occupied.some(item => intersects(candidate, item));
      if (!hasCollision) {
        return candidate;
      }
    }
  }
  return { i: '', x: 0, y: occupied.length * h, w, h, minW, minH };
}

export function DataProvider({ children }: { children: ReactNode }) {
  const [dataset, setDatasetInternal] = useState<DatasetState | null>(null);
  const [currentChart, setCurrentChart] = useState<ChartResponse | null>(null);
  const [filters, setFiltersInternal] = useState<FilterConfig[]>([]);
  const [viewMode, setViewMode] = useState<ViewMode>('chart');
  const [builderMode, setBuilderMode] = useState<BuilderMode>('ai');
  const [workspaceMode, setWorkspaceMode] = useState<WorkspaceMode>('explore');
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [currentHistoryId, setCurrentHistoryId] = useState<string | null>(null);
  const [dashboardStore, setDashboardStore] = useState<Record<string, DashboardWidget[]>>({});
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
        const dashboard = localStorage.getItem(DASHBOARD_KEY);
        
        if (hist) {
          setHistory(JSON.parse(hist).map((i: HistoryItem) => ({ ...i, timestamp: new Date(i.timestamp) })));
        }
        if (dashboard) {
          setDashboardStore(JSON.parse(dashboard));
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
  useEffect(() => {
    if (!isClient) return;
    const timeout = setTimeout(() => {
      localStorage.setItem(DASHBOARD_KEY, JSON.stringify(dashboardStore));
    }, 250);
    return () => clearTimeout(timeout);
  }, [dashboardStore, isClient]);

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
  const dashboardWidgets = dataset ? (dashboardStore[dataset.datasetId] ?? []) : [];
  const pinCurrentChart = useCallback((chart: ChartResponse, sourceQuery?: string) => {
    if (!dataset || chart.chart_type === 'empty') return;
    const id = crypto.randomUUID();
    setDashboardStore(prev => {
      const existing = prev[dataset.datasetId] ?? [];
      const placement = findNextLayoutSlot(chart.chart_type, existing.map(widget => widget.layout));
      const widget: DashboardWidget = {
        id,
        title: chart.title,
        chart: JSON.parse(JSON.stringify(chart)) as ChartResponse,
        sourceQuery: sourceQuery?.trim() || chart.title,
        createdAt: new Date().toISOString(),
        datasetId: dataset.datasetId,
        layout: { ...placement, i: id },
      };
      return {
        ...prev,
        [dataset.datasetId]: [widget, ...existing],
      };
    });
  }, [dataset]);
  const removeWidget = useCallback((widgetId: string) => {
    if (!dataset) return;
    setDashboardStore(prev => ({
      ...prev,
      [dataset.datasetId]: (prev[dataset.datasetId] ?? []).filter(widget => widget.id !== widgetId),
    }));
  }, [dataset]);
  const updateWidgetLayout = useCallback((layouts: DashboardLayoutItem[]) => {
    if (!dataset) return;
    setDashboardStore(prev => {
      const current = prev[dataset.datasetId] ?? [];
      const next = current.map(widget => {
        const layout = layouts.find(l => l.i === widget.id);
        return layout ? { ...widget, layout: { ...widget.layout, ...layout } } : widget;
      });
      return {
        ...prev,
        [dataset.datasetId]: next,
      };
    });
  }, [dataset]);
  const clearDashboard = useCallback(() => {
    if (!dataset) return;
    setDashboardStore(prev => ({
      ...prev,
      [dataset.datasetId]: [],
    }));
  }, [dataset]);
  const clearData = useCallback(() => {
    setDatasetInternal(null);
    setCurrentChart(null);
    setCurrentHistoryId(null);
    setFiltersInternal([]);
    setWorkspaceMode('explore');
    if (typeof window !== 'undefined') localStorage.removeItem(DATASET_KEY);
  }, []);

  const numericColumns = dataset?.columns.filter(c => c.is_numeric) ?? [];
  const categoricalColumns = dataset?.columns.filter(c => c.semantic_type === 'categorical') ?? [];
  const metricColumns = dataset?.columns.filter(c => c.semantic_type === 'metric') ?? [];
  const temporalColumns = dataset?.columns.filter(c => c.semantic_type === 'temporal') ?? [];

  return (
    <DataContext.Provider value={{
      dataset, setDataset, currentChart, setCurrentChart, filters, setFilters, addFilter, removeFilter, clearFilters,
      viewMode, setViewMode, builderMode, setBuilderMode, workspaceMode, setWorkspaceMode,
      history, currentHistoryId, addToHistory, selectFromHistory, clearHistory,
      dashboardWidgets, pinCurrentChart, removeWidget, updateWidgetLayout, clearDashboard,
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
