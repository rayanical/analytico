'use client';

import React from 'react';
import { motion } from 'framer-motion';
// ... imports
import { BarChart3, Clock, Trash2, ChevronRight, Wand2, Wrench, PanelLeftClose, PanelLeftOpen } from 'lucide-react';
import { useData } from '@/context/DataContext';
import { HistoryItem } from '@/types';

interface SidebarProps {
  isOpen: boolean;
  onToggle: () => void;
}

export function Sidebar({ isOpen, onToggle }: SidebarProps) {
  const { history, selectFromHistory, clearHistory, currentChart, dataset } = useData();

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
      case 'bar': return '📊';
      case 'line': return '📈';
      case 'area': return '📉';
      case 'pie': return '🥧';
      case 'composed': return '📦';
      default: return '📊';
    }
  };

  return (
    <motion.aside
      initial={false}
      animate={{ width: isOpen ? 288 : 64 }}
      className="relative flex h-full flex-col border-r border-border/50 bg-sidebar transition-all duration-300 ease-in-out"
    >
      {/* Toggle Button */}
      <button
        onClick={onToggle}
        className="absolute -right-3 top-6 z-50 flex h-6 w-6 items-center justify-center rounded-full border border-border bg-background shadow-sm hover:bg-accent"
      >
        {isOpen ? <PanelLeftClose className="h-3 w-3" /> : <PanelLeftOpen className="h-3 w-3" />}
      </button>

      {/* Logo */}
      <div className={`flex items-center gap-3 border-b border-border/50 p-6 ${!isOpen && 'justify-center px-2 py-6'}`}>
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-primary/60">
          <BarChart3 className="h-5 w-5 text-primary-foreground" />
        </div>
        {isOpen && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="overflow-hidden whitespace-nowrap">
            <h1 className="text-lg font-bold text-foreground">Analytico</h1>
            <p className="text-xs text-muted-foreground">Self-Service Analytics</p>
          </motion.div>
        )}
      </div>

      {/* Dataset Summary Card */}
      {isOpen && dataset?.summary && (
        <div className="mx-3 mt-3 rounded-lg border border-primary/20 bg-primary/5 p-3">
          <div className="flex items-center gap-2 mb-1.5">
            <Wand2 className="h-3.5 w-3.5 text-primary" />
            <span className="text-xs font-medium text-primary">Dataset Context</span>
          </div>
          <p className="text-xs text-muted-foreground leading-relaxed">{dataset.summary}</p>
        </div>
      )}

      {/* History Header */}
      {isOpen ? (
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
      ) : (
        <div className="flex justify-center py-4">
           {history.length > 0 && <Clock className="h-4 w-4 text-muted-foreground" />}
        </div>
      )}

      {/* History List */}
      <div className="flex-1 overflow-y-auto px-3 pb-4 scrollbar-hide">
        {history.length === 0 && isOpen ? (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <div className="mb-3 rounded-full bg-muted/50 p-4">
              <BarChart3 className="h-6 w-6 text-muted-foreground" />
            </div>
            <p className="text-sm text-muted-foreground">No charts yet</p>
            <p className="mt-1 text-xs text-muted-foreground/60">
              Upload data and create a chart
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
                className={`group w-full rounded-lg text-left transition-all ${
                  isOpen ? 'p-3' : 'flex justify-center p-2'
                } ${
                  currentChart?.title === item.chartResponse.title
                    ? 'bg-primary/10 border border-primary/30'
                    : 'hover:bg-white/[0.03]'
                }`}
                title={!isOpen ? item.chartResponse.title : undefined}
              >
                {isOpen ? (
                  <div className="flex items-start gap-3">
                    <span className="text-lg">{chartTypeIcon(item.chartResponse.chart_type)}</span>
                    <div className="flex-1 overflow-hidden">
                      <div className="flex items-center gap-1.5">
                        <p className="truncate text-sm font-medium text-foreground">
                          {item.chartResponse.title}
                        </p>
                        {item.isManual ? (
                          <Wrench className="h-3 w-3 text-muted-foreground" />
                        ) : (
                          <Wand2 className="h-3 w-3 text-primary" />
                        )}
                      </div>
                      <p className="mt-0.5 truncate text-xs text-muted-foreground">
                        {item.query}
                      </p>
                      <p className="mt-1 text-xs text-muted-foreground/60">
                        {formatTime(item.timestamp)}
                      </p>
                    </div>
                    <ChevronRight className="h-4 w-4 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" />
                  </div>
                ) : (
                   <div className="flex flex-col items-center gap-1">
                      <span className="text-lg">{chartTypeIcon(item.chartResponse.chart_type)}</span>
                   </div>
                )}
              </motion.button>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      {isOpen && (
        <div className="border-t border-border/50 p-4">
          <p className="text-center text-xs text-muted-foreground/60">
            Powered by GPT-4o-mini
          </p>
        </div>
      )}
    </motion.aside>
  );
}
