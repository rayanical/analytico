'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode, useCallback, useRef } from 'react';
import {
  ColumnSummary,
  ChartResponse,
  HistoryItem,
  FilterConfig,
  ViewMode,
  BuilderMode,
  WorkspaceMode,
  DashboardWidget,
  DashboardLayoutItem,
  DashboardUiState,
  DatasetState,
} from '@/types';
import { isDatasetState } from '@/lib/storageValidation';

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
  beginQuery: (datasetId: string, controller?: AbortController) => number;
  isCurrentQuery: (requestId: number, datasetId: string) => boolean;
  finishQuery: (requestId: number) => void;
  isCurrentDataset: (datasetId: string) => boolean;
  // Column helpers
  numericColumns: ColumnSummary[];
  categoricalColumns: ColumnSummary[];
  metricColumns: ColumnSummary[];
  temporalColumns: ColumnSummary[];
  // Drill Down
  drillDownData: Record<string, unknown>[] | null;
  setDrillDownData: (data: Record<string, unknown>[] | null) => void;
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

function readStorage(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch (error) {
    console.warn(`Unable to read saved ${key}`, error);
    return null;
  }
}

function writeStorage(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch (error) {
    console.warn(`Unable to save ${key}`, error);
  }
}

function removeStorage(key: string): void {
  try {
    localStorage.removeItem(key);
  } catch (error) {
    console.warn(`Unable to remove saved ${key}`, error);
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isChartResponse(value: unknown): value is ChartResponse {
  return isRecord(value)
    && Array.isArray(value.data) && value.data.every(isRecord)
    && typeof value.x_axis_key === 'string'
    && Array.isArray(value.y_axis_keys) && value.y_axis_keys.every(key => typeof key === 'string')
    && ['bar', 'line', 'area', 'pie', 'composed', 'empty'].includes(String(value.chart_type))
    && typeof value.title === 'string'
    && typeof value.row_count === 'number';
}

function isHistoryItem(value: unknown): value is HistoryItem {
  if (!isRecord(value) || typeof value.id !== 'string' || typeof value.datasetId !== 'string'
    || typeof value.query !== 'string' || !isChartResponse(value.chartResponse)) return false;
  const timestamp = new Date(value.timestamp as string | number | Date);
  return Number.isFinite(timestamp.getTime());
}

function parseHistory(value: string | null): HistoryItem[] {
  if (!value) return [];
  try {
    const parsed: unknown = JSON.parse(value);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isHistoryItem).slice(0, 30).map(item => ({
      ...item,
      timestamp: new Date(item.timestamp),
    }));
  } catch {
    return [];
  }
}

function parseDashboardStore(value: string | null): Record<string, DashboardWidget[]> {
  if (!value) return {};
  try {
    const parsed: unknown = JSON.parse(value);
    if (!isRecord(parsed)) return {};
    return Object.fromEntries(Object.entries(parsed).map(([datasetId, widgets]) => [
      datasetId,
      Array.isArray(widgets) ? widgets.filter((widget): widget is DashboardWidget =>
        isRecord(widget) && widget.datasetId === datasetId && typeof widget.id === 'string'
        && isChartResponse(widget.chart) && isRecord(widget.layout)
        && typeof widget.layout.x === 'number' && typeof widget.layout.y === 'number'
        && typeof widget.layout.w === 'number' && typeof widget.layout.h === 'number'
      ) : [],
    ]));
  } catch {
    return {};
  }
}

function parseDashboardUiStore(value: string | null): Record<string, DashboardUiState> {
  if (!value) return {};
  try {
    const parsed: unknown = JSON.parse(value);
    if (!isRecord(parsed)) return {};
    return Object.fromEntries(Object.entries(parsed).filter(([, state]) =>
      isRecord(state) && typeof state.isHeaderCollapsed === 'boolean'
      && (state.headerCollapseMode === 'auto' || state.headerCollapseMode === 'manual')
    )) as Record<string, DashboardUiState>;
  } catch {
    return {};
  }
}

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
  const [currentChart, setCurrentChartInternal] = useState<ChartResponse | null>(null);
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
  const [drillDownData, setDrillDownData] = useState<Record<string, unknown>[] | null>(null);
  const [isDrillDownOpen, setIsDrillDownOpen] = useState(false);
  const [groupOthers, setGroupOthersInternal] = useState(true);
  const [limit, setLimitInternal] = useState(20);
  const [sortBy, setSortByInternal] = useState<'value' | 'label'>('value');
  const [storageReady, setStorageReady] = useState(false);
  const datasetIdRef = useRef<string | null>(null);
  const datasetRevisionRef = useRef(0);
  const queryRequestIdRef = useRef(0);
  const activeQueryControllerRef = useRef<AbortController | null>(null);
  const dashboardTouchedIdsRef = useRef(new Set<string>());
  const dashboardUiTouchedIdsRef = useRef(new Set<string>());
  const historyClearedDatasetIdsRef = useRef(new Set<string>());
  const historyClearedAllRef = useRef(false);

  const setCurrentChart = useCallback((chart: ChartResponse | null) => {
    setCurrentChartInternal(chart);
    setDrillDownData(null);
    setIsDrillDownOpen(false);
  }, []);

  const beginQuery = useCallback((datasetId: string, controller?: AbortController) => {
    activeQueryControllerRef.current?.abort();
    activeQueryControllerRef.current = null;
    const requestId = ++queryRequestIdRef.current;
    if (datasetIdRef.current !== datasetId) {
      controller?.abort();
      setIsQuerying(false);
      return requestId;
    }
    activeQueryControllerRef.current = controller ?? null;
    setIsQuerying(true);
    return requestId;
  }, []);

  const isCurrentQuery = useCallback((requestId: number, datasetId: string) =>
    queryRequestIdRef.current === requestId && datasetIdRef.current === datasetId, []);

  const finishQuery = useCallback((requestId: number) => {
    if (queryRequestIdRef.current === requestId) {
      activeQueryControllerRef.current = null;
      setIsQuerying(false);
    }
  }, []);

  const isCurrentDataset = useCallback((datasetId: string) => datasetIdRef.current === datasetId, []);

  const invalidateQuery = useCallback(() => {
    queryRequestIdRef.current += 1;
    activeQueryControllerRef.current?.abort();
    activeQueryControllerRef.current = null;
    setIsQuerying(false);
  }, []);

  useEffect(() => {
    const loadAndValidate = async () => {
      const startingRevision = datasetRevisionRef.current;
      try {
        const saved = readStorage(DATASET_KEY);
        const savedHistory = parseHistory(readStorage(HISTORY_KEY));
        setHistory(prev => {
          const currentIds = new Set(prev.map(item => item.id));
          const restored = historyClearedAllRef.current
            ? []
            : savedHistory.filter(item => !currentIds.has(item.id) && !historyClearedDatasetIdsRef.current.has(item.datasetId));
          return [...prev, ...restored].slice(0, 30);
        });

        const savedDashboard = parseDashboardStore(readStorage(DASHBOARD_KEY));
        setDashboardStore(prev => {
          const merged = { ...savedDashboard };
          for (const datasetId of dashboardTouchedIdsRef.current) {
            if (datasetId in prev) merged[datasetId] = prev[datasetId];
            else delete merged[datasetId];
          }
          return merged;
        });

        const savedDashboardUi = parseDashboardUiStore(readStorage(DASHBOARD_UI_KEY));
        setDashboardUiStore(prev => {
          const merged = { ...savedDashboardUi };
          for (const datasetId of dashboardUiTouchedIdsRef.current) {
            if (datasetId in prev) merged[datasetId] = prev[datasetId];
            else delete merged[datasetId];
          }
          return merged;
        });
        
        if (saved) {
          let parsedDataset: unknown;
          try {
            parsedDataset = JSON.parse(saved);
          } catch {
            removeStorage(DATASET_KEY);
            return;
          }
          if (!isDatasetState(parsedDataset)) {
            removeStorage(DATASET_KEY);
            return;
          }
          const { validateDataset } = await import('@/lib/api');
          const validation = await validateDataset(parsedDataset.datasetId);
          if (datasetRevisionRef.current !== startingRevision) return;

          if (validation === 'valid') {
            datasetIdRef.current = parsedDataset.datasetId;
            setDatasetInternal(parsedDataset);
          } else if (validation === 'expired') {
            removeStorage(DATASET_KEY);
          }
        }
      } catch (e) {
        console.error('Load error:', e);
      } finally {
        setStorageReady(true);
      }
    };
    loadAndValidate();
  }, []);

  const setDataset = useCallback((state: DatasetState | null) => {
    activeQueryControllerRef.current?.abort();
    activeQueryControllerRef.current = null;
    datasetRevisionRef.current += 1;
    queryRequestIdRef.current += 1;
    datasetIdRef.current = state?.datasetId ?? null;
    setDatasetInternal(state);
    setCurrentChartInternal(null);
    setCurrentHistoryId(null);
    setFiltersInternal([]);
    setIsQuerying(false);
    setIsDrillDownOpen(false);
    setDrillDownData(null);
    setViewMode('chart');
    setLimitInternal(20);
    setGroupOthersInternal(true);
    setSortByInternal('value');
    if (typeof window !== 'undefined') {
      if (state) writeStorage(DATASET_KEY, JSON.stringify(state));
      else removeStorage(DATASET_KEY);
    }
  }, []);

  useEffect(() => {
    if (!storageReady) return;
    writeStorage(HISTORY_KEY, JSON.stringify(history));
  }, [history, storageReady]);
  useEffect(() => {
    if (!storageReady) return;
    const timeout = setTimeout(() => {
      writeStorage(DASHBOARD_KEY, JSON.stringify(dashboardStore));
    }, 250);
    return () => clearTimeout(timeout);
  }, [dashboardStore, storageReady]);
  useEffect(() => {
    if (!storageReady) return;
    const timeout = setTimeout(() => {
      writeStorage(DASHBOARD_UI_KEY, JSON.stringify(dashboardUiStore));
    }, 250);
    return () => clearTimeout(timeout);
  }, [dashboardUiStore, storageReady]);

  const setFilters = useCallback((f: FilterConfig[]) => {
    invalidateQuery();
    setFiltersInternal(f);
  }, [invalidateQuery]);
  const addFilter = useCallback((f: FilterConfig) => {
    invalidateQuery();
    setFiltersInternal(prev => {
      const idx = prev.findIndex(x => x.column === f.column);
      if (idx >= 0) { const u = [...prev]; u[idx] = f; return u; }
      return [...prev, f];
    });
  }, [invalidateQuery]);
  const removeFilter = useCallback((col: string) => {
    invalidateQuery();
    setFiltersInternal(prev => prev.filter(f => f.column !== col));
  }, [invalidateQuery]);
  const clearFilters = useCallback(() => {
    invalidateQuery();
    setFiltersInternal([]);
  }, [invalidateQuery]);

  const setLimit = useCallback((value: number) => {
    invalidateQuery();
    setLimitInternal(value);
  }, [invalidateQuery]);
  const setGroupOthers = useCallback((value: boolean) => {
    invalidateQuery();
    setGroupOthersInternal(value);
  }, [invalidateQuery]);
  const setSortBy = useCallback((value: 'value' | 'label') => {
    invalidateQuery();
    setSortByInternal(value);
  }, [invalidateQuery]);

  const addToHistory = useCallback((query: string, response: ChartResponse, isManual: boolean) => {
    const datasetId = datasetIdRef.current;
    if (!datasetId) return;
    const entry: HistoryItem = {
      id: crypto.randomUUID(),
      datasetId,
      query,
      chartResponse: response,
      timestamp: new Date(),
      isManual,
    };
    setHistory(prev => [entry, ...prev].slice(0, 30));
    setCurrentHistoryId(entry.id);
  }, []);
  const selectFromHistory = useCallback((item: HistoryItem) => {
    if (!dataset || item.datasetId !== dataset.datasetId) return;
    invalidateQuery();
    setCurrentChart(item.chartResponse);
    setCurrentHistoryId(item.id);
    setViewMode('chart');
  }, [dataset, invalidateQuery, setCurrentChart]);
  const clearHistory = useCallback(() => {
    if (dataset) {
      historyClearedDatasetIdsRef.current.add(dataset.datasetId);
      setHistory(prev => prev.filter(item => item.datasetId !== dataset.datasetId));
    } else {
      historyClearedAllRef.current = true;
      setHistory([]);
    }
    setCurrentHistoryId(null);
  }, [dataset]);
  const visibleHistory = dataset ? history.filter(item => item.datasetId === dataset.datasetId) : [];
  const dashboardWidgets = dataset ? (dashboardStore[dataset.datasetId] ?? []) : [];
  const dashboardUiState = dataset ? (dashboardUiStore[dataset.datasetId] ?? DEFAULT_DASHBOARD_UI) : DEFAULT_DASHBOARD_UI;

  useEffect(() => {
    if (!dataset || workspaceMode !== 'dashboard') return;
    const state = dashboardUiStore[dataset.datasetId] ?? DEFAULT_DASHBOARD_UI;
    if (state.headerCollapseMode === 'auto' && dashboardWidgets.length >= 1 && !state.isHeaderCollapsed) {
      dashboardUiTouchedIdsRef.current.add(dataset.datasetId);
      setDashboardUiStore(prev => ({
        ...prev,
        [dataset.datasetId]: { isHeaderCollapsed: true, headerCollapseMode: 'auto' },
      }));
    }
  }, [dataset, workspaceMode, dashboardWidgets.length, dashboardUiStore]);

  const setDashboardHeaderCollapsed = useCallback((collapsed: boolean, mode: 'auto' | 'manual' = 'manual') => {
    if (!dataset) return;
    dashboardUiTouchedIdsRef.current.add(dataset.datasetId);
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
    dashboardTouchedIdsRef.current.add(dataset.datasetId);
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
    dashboardTouchedIdsRef.current.add(dataset.datasetId);
    setDashboardStore(prev => ({
      ...prev,
      [dataset.datasetId]: (prev[dataset.datasetId] ?? []).filter(widget => widget.id !== widgetId),
    }));
  }, [dataset]);
  const updateWidgetLayout = useCallback((layouts: DashboardLayoutItem[]) => {
    if (!dataset) return;
    dashboardTouchedIdsRef.current.add(dataset.datasetId);
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
    dashboardTouchedIdsRef.current.add(dataset.datasetId);
    setDashboardStore(prev => ({
      ...prev,
      [dataset.datasetId]: [],
    }));
  }, [dataset]);
  const clearData = useCallback(() => {
    setDataset(null);
    setWorkspaceMode('explore');
  }, [setDataset]);

  const numericColumns = dataset?.columns.filter(c => c.is_numeric) ?? [];
  const categoricalColumns = dataset?.columns.filter(c => c.semantic_type === 'categorical') ?? [];
  const metricColumns = dataset?.columns.filter(c => c.semantic_type === 'metric') ?? [];
  const temporalColumns = dataset?.columns.filter(c => c.semantic_type === 'temporal') ?? [];

  return (
    <DataContext.Provider value={{
      dataset, setDataset, currentChart, setCurrentChart, filters, setFilters, addFilter, removeFilter, clearFilters,
      viewMode, setViewMode, builderMode, setBuilderMode, workspaceMode, setWorkspaceMode,
      history: visibleHistory, currentHistoryId, addToHistory, selectFromHistory, clearHistory,
      dashboardWidgets, pinCurrentChart, removeWidget, updateWidgetLayout, clearDashboard,
      dashboardUiState, setDashboardHeaderCollapsed,
      isUploading, setIsUploading, isQuerying, beginQuery, isCurrentQuery, finishQuery, isCurrentDataset, numericColumns, categoricalColumns, metricColumns, temporalColumns, clearData,
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
