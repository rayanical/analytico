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
  DashboardUiState,
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
  dashboardUiState: DashboardUiState;
  setDashboardHeaderCollapsed: (collapsed: boolean, mode?: 'auto' | 'manual') => void;
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
const DASHBOARD_UI_KEY = 'analytico_dashboard_ui_v1';

const DEFAULT_DASHBOARD_UI: DashboardUiState = {
  isHeaderCollapsed: false,
  headerCollapseMode: 'auto',
};

function intersects(a: DashboardLayoutItem, b: DashboardLayoutItem): boolean {
  return !(a.x + a.w <= b.x || b.x + b.w <= a.x || a.y + a.h <= b.y || b.y + b.h <= a.y);
}

function getDefaultWidgetDimensions(chartType: ChartResponse['chart_type']) {
  if (chartType === 'pie') return { w: 8, h: 9, minW: 6, minH: 8, maxW: 12 };
  if (chartType === 'line' || chartType === 'area') return { w: 12, h: 9, minW: 8, minH: 8, maxW: 16 };
  if (chartType === 'composed') return { w: 12, h: 9, minW: 9, minH: 8, maxW: 16 };
  return { w: 10, h: 9, minW: 7, minH: 8, maxW: 16 };
}

function findNextLayoutSlot(
  chartType: ChartResponse['chart_type'],
  occupied: DashboardLayoutItem[],
): DashboardLayoutItem {
  const cols = 24;
  const { w, h, minW, minH, maxW } = getDefaultWidgetDimensions(chartType);
  const sorted = [...occupied].sort((a, b) => a.y - b.y || a.x - b.x);

  if (sorted.length === 0) {
    return { i: '', x: 0, y: 0, w, h, minW, minH, maxW };
  }

  const uniqueRows = Array.from(new Set(sorted.map(item => item.y))).sort((a, b) => a - b);
  for (const rowY of uniqueRows) {
    const rowItems = sorted
      .filter(item => item.y === rowY)
      .sort((a, b) => a.x - b.x);
    const rightEdge = rowItems.reduce((max, item) => Math.max(max, item.x + item.w), 0);
    if (rightEdge + w <= cols) {
      const candidate: DashboardLayoutItem = { i: '', x: rightEdge, y: rowY, w, h, minW, minH, maxW };
      const hasCollision = sorted.some(item => intersects(candidate, item));
      if (!hasCollision) return candidate;
    }
  }

  const nextY = sorted.reduce((max, item) => Math.max(max, item.y + item.h), 0);
  return { i: '', x: 0, y: nextY, w, h, minW, minH, maxW };
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
  const [dashboardUiStore, setDashboardUiStore] = useState<Record<string, DashboardUiState>>({});
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
        const dashboardUi = localStorage.getItem(DASHBOARD_UI_KEY);
        
        if (hist) {
          setHistory(JSON.parse(hist).map((i: HistoryItem) => ({ ...i, timestamp: new Date(i.timestamp) })));
        }
        if (dashboard) {
          setDashboardStore(JSON.parse(dashboard));
        }
        if (dashboardUi) {
          setDashboardUiStore(JSON.parse(dashboardUi));
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
  useEffect(() => {
    if (!isClient) return;
    const timeout = setTimeout(() => {
      localStorage.setItem(DASHBOARD_UI_KEY, JSON.stringify(dashboardUiStore));
    }, 250);
    return () => clearTimeout(timeout);
  }, [dashboardUiStore, isClient]);

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
  const dashboardUiState = dataset ? (dashboardUiStore[dataset.datasetId] ?? DEFAULT_DASHBOARD_UI) : DEFAULT_DASHBOARD_UI;

  useEffect(() => {
    if (!dataset || workspaceMode !== 'dashboard') return;
    const state = dashboardUiStore[dataset.datasetId] ?? DEFAULT_DASHBOARD_UI;
    if (state.headerCollapseMode === 'auto' && dashboardWidgets.length >= 1 && !state.isHeaderCollapsed) {
      setDashboardUiStore(prev => ({
        ...prev,
        [dataset.datasetId]: { isHeaderCollapsed: true, headerCollapseMode: 'auto' },
      }));
    }
  }, [dataset, workspaceMode, dashboardWidgets.length, dashboardUiStore]);

  const setDashboardHeaderCollapsed = useCallback((collapsed: boolean, mode: 'auto' | 'manual' = 'manual') => {
    if (!dataset) return;
    setDashboardUiStore(prev => ({
      ...prev,
      [dataset.datasetId]: {
        isHeaderCollapsed: collapsed,
        headerCollapseMode: mode,
      },
    }));
  }, [dataset]);

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
        placementStrategy: placement.y === 0 ? 'first-row-tile' : 'next-fit',
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
      dashboardUiState, setDashboardHeaderCollapsed,
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
