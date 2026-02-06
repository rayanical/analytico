'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sidebar } from '@/components/Sidebar';
import { FileUploader } from '@/components/FileUploader';
import { ChatInterface } from '@/components/ChatInterface';
import { ChartBuilder } from '@/components/ChartBuilder';
import { SmartChart } from '@/components/SmartChart';
import { ChartSkeleton } from '@/components/ChartSkeleton';
import { DataTable } from '@/components/DataTable';
import { FilterBar } from '@/components/FilterBar';
import { DrillDownModal } from '@/components/DrillDownModal';
import { useData } from '@/context/DataContext';
import { aggregateData } from '@/lib/api';
import { BarChart3, Sparkles, Wand2, Wrench, Table, LineChart, Info, AlertTriangle, X, Filter, Download, FileJson, FileImage } from 'lucide-react';
import { toPng, toSvg } from 'html-to-image';
import { toast } from 'sonner';

export default function Home() {
  const {
    dataset, currentChart, isQuerying, viewMode, setViewMode, builderMode, setBuilderMode,
    filters, setCurrentChart, limit, groupOthers, setIsQuerying,
  } = useData();

  const [showReasoning, setShowReasoning] = useState(false);
  const [expandedFilterColumn, setExpandedFilterColumn] = useState<string | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  
  // Track previous filters and settings to detect changes
  const prevFiltersRef = useRef(JSON.stringify(filters));
  const prevLimitRef = useRef(limit);
  const prevGroupOthersRef = useRef(groupOthers);
  const isInitialMount = useRef(true);
  
  // Auto-refresh chart when filters or settings change
  useEffect(() => {
    // Skip on initial mount
    if (isInitialMount.current) {
      isInitialMount.current = false;
      prevFiltersRef.current = JSON.stringify(filters);
      prevLimitRef.current = limit;
      prevGroupOthersRef.current = groupOthers;
      return;
    }
    
    const currentFiltersStr = JSON.stringify(filters);
    const filtersChanged = prevFiltersRef.current !== currentFiltersStr;
    const limitChanged = prevLimitRef.current !== limit;
    const groupOthersChanged = prevGroupOthersRef.current !== groupOthers;
    
    // Only refresh if something actually changed and we have an active chart
    if ((filtersChanged || limitChanged || groupOthersChanged) && currentChart && dataset) {
      prevFiltersRef.current = currentFiltersStr;
      prevLimitRef.current = limit;
      prevGroupOthersRef.current = groupOthers;
      
      // Only auto-refresh if we have a valid chart config (not empty/text-only response)
      if (currentChart.x_axis_key && currentChart.y_axis_keys.length > 0 && currentChart.chart_type !== 'empty') {
        const refreshChart = async () => {
          setIsQuerying(true);
          try {
            const response = await aggregateData({
              dataset_id: dataset.datasetId,
              x_axis_key: currentChart.x_axis_key,
              y_axis_keys: currentChart.y_axis_keys,
              aggregation: 'sum', // Default
              chart_type: currentChart.chart_type,
              filters: filters.length > 0 ? filters : undefined,
              limit,
              group_others: groupOthers,
            });
            setCurrentChart({ ...response, reasoning: currentChart.reasoning });
            
            // Show appropriate success message
            if (filtersChanged && (limitChanged || groupOthersChanged)) {
              toast.success('Chart updated with filters and settings');
            } else if (filtersChanged) {
              toast.success('Chart updated with filters');
            } else {
              toast.success('Chart updated with settings');
            }
          } catch (error) {
            console.error('Chart refresh error:', error);
            toast.error('Failed to update chart');
          } finally {
            setIsQuerying(false);
          }
        };
        refreshChart();
      }
    }
  }, [filters, currentChart, dataset, limit, groupOthers, setCurrentChart, setIsQuerying]);

  const downloadChart = async (format: 'png' | 'svg') => {
    const node = document.getElementById('chart-export-container');
    if (!node) {
      toast.error('Chart container not found');
      return;
    }
    
    // Show toast
    const toastId = toast.loading(`Generating ${format.toUpperCase()}...`);
    
    try {
      // Small delay to ensure render
      await new Promise(resolve => setTimeout(resolve, 100));
      
      const dataUrl = format === 'png' 
        ? await toPng(node, { backgroundColor: '#09090b', style: { borderRadius: '0' } }) 
        : await toSvg(node, { backgroundColor: '#09090b', style: { borderRadius: '0' } });
        
      const link = document.createElement('a');
      link.download = `analytico-chart-${dataset?.filename || 'export'}.${format}`;
      link.href = dataUrl;
      link.click();
      toast.dismiss(toastId);
      toast.success(`${format.toUpperCase()} downloaded`);
    } catch (error) {
      console.error(error);
      toast.error('Export failed', { id: toastId });
    }
  };

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar isOpen={isSidebarOpen} onToggle={() => setIsSidebarOpen(!isSidebarOpen)} />
      
      <main className="relative flex flex-1 flex-col overflow-hidden min-w-0">
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-primary/5" />
        
        <div className="relative flex flex-1 flex-col overflow-y-auto p-6">
          {/* Header */}
          <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-6">
            <h1 className="text-2xl font-bold">Data Visualization</h1>
            <p className="mt-1 text-sm text-muted-foreground">Drop a CSV to get instant insights — zero friction</p>
          </motion.div>

          {/* File Upload */}
          <section className="mb-6">
            <FileUploader />
            {dataset?.summary && (
              <motion.div 
                initial={{ opacity: 0, y: 10 }} 
                animate={{ opacity: 1, y: 0 }} 
                className="mt-4 rounded-lg border border-primary/20 bg-primary/5 p-4"
              >
                <div className="flex items-start gap-3">
                  <div className="rounded-full bg-primary/10 p-2 text-primary">
                     <Sparkles className="h-4 w-4" />
                  </div>
                  <div>
                    <h3 className="mb-1 text-sm font-semibold text-foreground">AI Business Summary</h3>
                    <p className="text-sm text-muted-foreground leading-relaxed">{dataset.summary}</p>
                  </div>
                </div>
              </motion.div>
            )}
          </section>

          {/* Filter Bar */}
          {dataset && (
            <section className="mb-6">
              <FilterBar 
                expandColumn={expandedFilterColumn} 
                onExpandChange={setExpandedFilterColumn} 
              />
            </section>
          )}

          {/* Mode Tabs and View Toggle */}
          {dataset && (
            <div className="mb-6 flex items-center justify-between">
              <div className="flex rounded-lg bg-muted/30 p-1">
                <button onClick={() => setBuilderMode('ai')} className={`flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-all ${builderMode === 'ai' ? 'bg-card text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}>
                  <Wand2 className="h-4 w-4" />AI Assistant
                </button>
                <button onClick={() => setBuilderMode('manual')} className={`flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-all ${builderMode === 'manual' ? 'bg-card text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}>
                  <Wrench className="h-4 w-4" />Chart Builder
                </button>
              </div>
              {currentChart && (
                <div className="flex rounded-lg bg-muted/30 p-1">
                  <button onClick={() => setViewMode('chart')} className={`flex items-center gap-2 rounded-md px-3 py-1.5 text-sm transition-all ${viewMode === 'chart' ? 'bg-card text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}>
                    <LineChart className="h-4 w-4" />Chart
                  </button>
                  <button onClick={() => setViewMode('table')} className={`flex items-center gap-2 rounded-md px-3 py-1.5 text-sm transition-all ${viewMode === 'table' ? 'bg-card text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}>
                    <Table className="h-4 w-4" />Data
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Main Content */}
          <div className="flex flex-1 gap-6 min-w-0">
            <div className="flex-1 min-w-0">
              {isQuerying ? (
                <ChartSkeleton />
              ) : currentChart ? (
                <motion.div 
                  id="chart-export-container"
                  key={viewMode} 
                  initial={{ opacity: 0 }} 
                  animate={{ opacity: 1 }} 
                  className="rounded-xl border border-border/50 bg-card/50 p-6 backdrop-blur-sm min-w-0 overflow-hidden"
                >
                  {/* Chart Header */}
                  <div className="mb-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <h3 className="text-xl font-semibold">{currentChart.title}</h3>
                        {currentChart.reasoning && (
                          <button onClick={() => setShowReasoning(!showReasoning)} className="rounded-full p-1.5 text-muted-foreground hover:bg-primary/10 hover:text-primary" title="Reasoning">
                            <Info className="h-4 w-4" />
                          </button>
                        )}
                        {currentChart.warnings?.length ? (
                          <span className="flex items-center gap-1 rounded-full bg-yellow-500/20 px-2 py-1 text-xs text-yellow-400">
                            <AlertTriangle className="h-3 w-3" />{currentChart.warnings.length}
                          </span>
                        ) : null}
                      </div>
                      
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-muted-foreground mr-2">{currentChart.row_count} points</span>
                        <div className="flex rounded-md bg-muted/30 p-0.5">
                          <button onClick={() => downloadChart('png')} className="rounded px-2.5 py-1.5 text-xs font-medium hover:bg-background hover:text-primary transition-all" title="Download PNG">
                            PNG
                          </button>
                          <div className="w-[1px] bg-border/50 my-1" />
                          <button onClick={() => downloadChart('svg')} className="rounded px-2.5 py-1.5 text-xs font-medium hover:bg-background hover:text-primary transition-all" title="Download SVG">
                            SVG
                          </button>
                        </div>
                      </div>
                    </div>
                    
                    {/* Active Filter Chips */}
                    {currentChart.applied_filters?.length ? (
                      <div className="mt-3 flex flex-wrap gap-2">
                        <Filter className="h-4 w-4 text-muted-foreground" />
                        {currentChart.applied_filters.map((f, i) => (
                          <span key={i} className="rounded-full bg-primary/10 px-2 py-1 text-xs text-primary">
                            {f}
                          </span>
                        ))}
                      </div>
                    ) : null}
                    
                    {/* Reasoning Panel */}
                    <AnimatePresence>
                      {showReasoning && currentChart.reasoning && (
                        <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }} className="mt-3 overflow-hidden">
                          <div className="flex items-start gap-3 rounded-lg border border-primary/20 bg-primary/5 p-3">
                            <Sparkles className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
                            <div className="flex-1">
                              <p className="text-sm font-medium text-primary">AI Reasoning</p>
                              <p className="mt-1 text-sm text-muted-foreground">{currentChart.reasoning}</p>
                              {currentChart.warnings?.map((w, i) => (
                                <p key={i} className="mt-1 flex items-center gap-1.5 text-xs text-yellow-400">
                                  <AlertTriangle className="h-3 w-3" />{w}
                                </p>
                              ))}
                            </div>
                            <button onClick={() => setShowReasoning(false)} className="text-muted-foreground hover:text-foreground"><X className="h-4 w-4" /></button>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                  
                  {viewMode === 'chart' ? <SmartChart /> : <DataTable />}
                </motion.div>
              ) : (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex h-[400px] flex-col items-center justify-center rounded-xl border border-dashed border-border/50 bg-card/30">
                  {dataset ? (
                    <>
                      <div className="mb-4 rounded-full bg-primary/10 p-4"><Sparkles className="h-8 w-8 text-primary" /></div>
                      <h3 className="text-lg font-medium">Ready to visualize!</h3>
                      <p className="mt-2 text-center text-sm text-muted-foreground">
                        {builderMode === 'ai' ? 'Ask a question below' : 'Use the Chart Builder on the right'}
                      </p>
                    </>
                  ) : (
                    <>
                      <div className="mb-4 rounded-full bg-muted/50 p-4"><BarChart3 className="h-8 w-8 text-muted-foreground" /></div>
                      <h3 className="text-lg font-medium text-muted-foreground">No data loaded</h3>
                      <p className="mt-2 text-sm text-muted-foreground/60">Upload a CSV to get started</p>
                    </>
                  )}
                </motion.div>
              )}
            </div>

            {/* Manual Builder */}
            {builderMode === 'manual' && dataset && (
              <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="w-80 shrink-0 rounded-xl border border-border/50 bg-card/50 p-6 backdrop-blur-sm">
                <h3 className="mb-4 flex items-center gap-2 text-lg font-semibold"><Wrench className="h-5 w-5" />Chart Builder</h3>
                <ChartBuilder />
              </motion.div>
            )}
          </div>
        </div>

        {/* Chat Input */}
        {builderMode === 'ai' && (
          <div className="relative border-t border-border/50 bg-background/80 p-6 backdrop-blur-xl">
            <div className="mx-auto max-w-4xl"><ChatInterface /></div>
          </div>
        )}
      </main>
      
      {/* Modals */}
      <DrillDownModal />
    </div>
  );
}
