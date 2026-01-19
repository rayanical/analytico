'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { BarChart3, Clock, Trash2, ChevronRight } from 'lucide-react';
import { useData } from '@/context/DataContext';
import { HistoryItem } from '@/types';

export function Sidebar() {
  const { history, selectFromHistory, clearHistory, currentChart } = useData();

  const formatTime = (date: Date) => {
    const now = new Date();
    const diffMs = now.getTime() - new Date(date).getTime();
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    
    const diffDays = Math.floor(diffHours / 24);
    return `${diffDays}d ago`;
  };

  const chartTypeIcon = (type: string) => {
    switch (type) {
      case 'bar':
        return '📊';
      case 'line':
        return '📈';
      case 'area':
        return '📉';
      case 'composed':
        return '📦';
      default:
        return '📊';
    }
  };

  return (
    <motion.aside
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      className="flex h-full w-72 flex-col border-r border-border/50 bg-sidebar"
    >
      {/* Logo */}
      <div className="flex items-center gap-3 border-b border-border/50 p-6">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-primary/60">
          <BarChart3 className="h-5 w-5 text-primary-foreground" />
        </div>
        <div>
          <h1 className="text-lg font-bold text-foreground">Analytico</h1>
          <p className="text-xs text-muted-foreground">AI Data Visualization</p>
        </div>
      </div>

      {/* History Header */}
      <div className="flex items-center justify-between px-6 py-4">
        <div className="flex items-center gap-2">
          <Clock className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium text-muted-foreground">History</span>
        </div>
        {history.length > 0 && (
          <button
            onClick={clearHistory}
            className="rounded-lg p-1.5 text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
            title="Clear history"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        )}
      </div>

      {/* History List */}
      <div className="flex-1 overflow-y-auto px-3 pb-4">
        {history.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <div className="mb-3 rounded-full bg-muted/50 p-4">
              <BarChart3 className="h-6 w-6 text-muted-foreground" />
            </div>
            <p className="text-sm text-muted-foreground">No charts yet</p>
            <p className="mt-1 text-xs text-muted-foreground/60">
              Upload data and ask a question
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            {history.map((item: HistoryItem, index: number) => (
              <motion.button
                key={item.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
                onClick={() => selectFromHistory(item)}
                className={`group w-full rounded-lg p-3 text-left transition-all ${
                  currentChart?.title === item.chartConfig.title
                    ? 'bg-primary/10 border border-primary/30'
                    : 'hover:bg-white/[0.03]'
                }`}
              >
                <div className="flex items-start gap-3">
                  <span className="text-lg">{chartTypeIcon(item.chartConfig.chartType)}</span>
                  <div className="flex-1 overflow-hidden">
                    <p className="truncate text-sm font-medium text-foreground">
                      {item.chartConfig.title}
                    </p>
                    <p className="mt-0.5 truncate text-xs text-muted-foreground">
                      {item.query}
                    </p>
                    <p className="mt-1 text-xs text-muted-foreground/60">
                      {formatTime(item.timestamp)}
                    </p>
                  </div>
                  <ChevronRight className="h-4 w-4 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" />
                </div>
              </motion.button>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="border-t border-border/50 p-4">
        <p className="text-center text-xs text-muted-foreground/60">
          Powered by GPT-4o-mini
        </p>
      </div>
    </motion.aside>
  );
}
