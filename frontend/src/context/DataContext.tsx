'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { DataSummary, ChartConfig, HistoryItem } from '@/types';

interface DataState {
  data: Record<string, unknown>[];
  columns: string[];
  summary: DataSummary;
  fileName: string | null;
  rowCount: number;
  truncated: boolean;
}

interface DataContextType {
  // Data state
  dataState: DataState | null;
  setDataState: (state: DataState | null) => void;
  
  // Current chart
  currentChart: ChartConfig | null;
  setCurrentChart: (chart: ChartConfig | null) => void;
  
  // History
  history: HistoryItem[];
  addToHistory: (query: string, config: ChartConfig) => void;
  selectFromHistory: (item: HistoryItem) => void;
  clearHistory: () => void;
  
  // Loading states
  isUploading: boolean;
  setIsUploading: (loading: boolean) => void;
  isQuerying: boolean;
  setIsQuerying: (loading: boolean) => void;
  
  // Clear all data
  clearData: () => void;
}

const DataContext = createContext<DataContextType | undefined>(undefined);

const STORAGE_KEY = 'analytico_data';
const HISTORY_KEY = 'analytico_history';

export function DataProvider({ children }: { children: ReactNode }) {
  const [dataState, setDataStateInternal] = useState<DataState | null>(null);
  const [currentChart, setCurrentChart] = useState<ChartConfig | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [isClient, setIsClient] = useState(false);

  // Load from localStorage on mount (client-side only)
  useEffect(() => {
    setIsClient(true);
    try {
      const savedData = localStorage.getItem(STORAGE_KEY);
      if (savedData) {
        const parsed = JSON.parse(savedData);
        setDataStateInternal(parsed);
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

  // Save data to localStorage when it changes
  const setDataState = (state: DataState | null) => {
    setDataStateInternal(state);
    if (isClient) {
      try {
        if (state) {
          localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
        } else {
          localStorage.removeItem(STORAGE_KEY);
        }
      } catch (error) {
        console.error('Error saving to localStorage:', error);
      }
    }
  };

  // Save history to localStorage when it changes
  useEffect(() => {
    if (isClient && history.length > 0) {
      try {
        localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
      } catch (error) {
        console.error('Error saving history to localStorage:', error);
      }
    }
  }, [history, isClient]);

  const addToHistory = (query: string, config: ChartConfig) => {
    const newItem: HistoryItem = {
      id: crypto.randomUUID(),
      query,
      chartConfig: config,
      timestamp: new Date(),
    };
    setHistory(prev => [newItem, ...prev].slice(0, 20)); // Keep last 20 items
  };

  const selectFromHistory = (item: HistoryItem) => {
    setCurrentChart(item.chartConfig);
  };

  const clearHistory = () => {
    setHistory([]);
    if (isClient) {
      localStorage.removeItem(HISTORY_KEY);
    }
  };

  const clearData = () => {
    setDataStateInternal(null);
    setCurrentChart(null);
    if (isClient) {
      localStorage.removeItem(STORAGE_KEY);
    }
  };

  return (
    <DataContext.Provider
      value={{
        dataState,
        setDataState,
        currentChart,
        setCurrentChart,
        history,
        addToHistory,
        selectFromHistory,
        clearHistory,
        isUploading,
        setIsUploading,
        isQuerying,
        setIsQuerying,
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
