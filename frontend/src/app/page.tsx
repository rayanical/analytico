'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sidebar } from '@/components/Sidebar';
import { FileUploader } from '@/components/FileUploader';
import { ChatInterface } from '@/components/ChatInterface';
import { ChartBuilder } from '@/components/ChartBuilder';
import { SmartChart } from '@/components/SmartChart';
import { ChartSkeleton } from '@/components/ChartSkeleton';
import { DataTable } from '@/components/DataTable';
import { FilterBar } from '@/components/FilterBar';
import { useData } from '@/context/DataContext';
import { BarChart3, Sparkles, Wand2, Wrench, Table, LineChart, Info, AlertTriangle, X } from 'lucide-react';

export default function Home() {
  const {
    dataset,
    currentChart,
    isQuerying,
    viewMode,
    setViewMode,
    builderMode,
    setBuilderMode,
  } = useData();

  const [showReasoning, setShowReasoning] = useState(false);

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      
      <main className="relative flex flex-1 flex-col overflow-hidden">
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-primary/5" />
        
        <div className="relative flex flex-1 flex-col overflow-y-auto p-6">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6"
          >
            <h1 className="text-2xl font-bold text-foreground">
              Data Visualization
            </h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Upload a CSV and use AI or the manual builder to create charts
            </p>
          </motion.div>

          {/* File Upload */}
          <section className="mb-6">
            <FileUploader />
          </section>

          {/* Filter Bar */}
          {dataset && (
            <section className="mb-6">
              <FilterBar />
            </section>
          )}

          {/* Mode Tabs and View Toggle */}
          {dataset && (
            <div className="mb-6 flex items-center justify-between">
              <div className="flex rounded-lg bg-muted/30 p-1">
                <button
                  onClick={() => setBuilderMode('ai')}
                  className={`flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-all ${
                    builderMode === 'ai'
                      ? 'bg-card text-foreground shadow-sm'
                      : 'text-muted-foreground hover:text-foreground'
                  }`}
                >
                  <Wand2 className="h-4 w-4" />
                  AI Assistant
                </button>
                <button
                  onClick={() => setBuilderMode('manual')}
                  className={`flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-all ${
                    builderMode === 'manual'
                      ? 'bg-card text-foreground shadow-sm'
                      : 'text-muted-foreground hover:text-foreground'
                  }`}
                >
                  <Wrench className="h-4 w-4" />
                  Chart Builder
                </button>
              </div>

              {currentChart && (
                <div className="flex rounded-lg bg-muted/30 p-1">
                  <button
                    onClick={() => setViewMode('chart')}
                    className={`flex items-center gap-2 rounded-md px-3 py-1.5 text-sm transition-all ${
                      viewMode === 'chart'
                        ? 'bg-card text-foreground shadow-sm'
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    <LineChart className="h-4 w-4" />
                    Chart
                  </button>
                  <button
                    onClick={() => setViewMode('table')}
                    className={`flex items-center gap-2 rounded-md px-3 py-1.5 text-sm transition-all ${
                      viewMode === 'table'
                        ? 'bg-card text-foreground shadow-sm'
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    <Table className="h-4 w-4" />
                    Data
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Main Content */}
          <div className="flex flex-1 gap-6">
            <div className="flex-1">
              {isQuerying ? (
                <ChartSkeleton />
              ) : currentChart ? (
                <motion.div
                  key={viewMode}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="rounded-xl border border-border/50 bg-card/50 p-6 backdrop-blur-sm"
                >
                  {/* Chart Header with Title, Reasoning, and Warnings */}
                  <div className="mb-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <h3 className="text-xl font-semibold text-foreground">
                          {currentChart.title}
                        </h3>
                        
                        {/* Reasoning Info Icon */}
                        {currentChart.reasoning && (
                          <button
                            onClick={() => setShowReasoning(!showReasoning)}
                            className="rounded-full p-1.5 text-muted-foreground transition-colors hover:bg-primary/10 hover:text-primary"
                            title="View AI reasoning"
                          >
                            <Info className="h-4 w-4" />
                          </button>
                        )}
                        
                        {/* Warnings */}
                        {currentChart.warnings && currentChart.warnings.length > 0 && (
                          <span className="flex items-center gap-1 rounded-full bg-yellow-500/20 px-2 py-1 text-xs text-yellow-400">
                            <AlertTriangle className="h-3 w-3" />
                            {currentChart.warnings.length} adjustment{currentChart.warnings.length > 1 ? 's' : ''}
                          </span>
                        )}
                      </div>
                      
                      <span className="text-sm text-muted-foreground">
                        {currentChart.row_count} data points
                      </span>
                    </div>
                    
                    {/* Reasoning Panel */}
                    <AnimatePresence>
                      {showReasoning && currentChart.reasoning && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          exit={{ opacity: 0, height: 0 }}
                          className="mt-3 overflow-hidden"
                        >
                          <div className="flex items-start gap-3 rounded-lg border border-primary/20 bg-primary/5 p-3">
                            <Sparkles className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
                            <div className="flex-1">
                              <p className="text-sm font-medium text-primary">AI Reasoning</p>
                              <p className="mt-1 text-sm text-muted-foreground">
                                {currentChart.reasoning}
                              </p>
                              {currentChart.warnings && currentChart.warnings.length > 0 && (
                                <div className="mt-2 space-y-1">
                                  {currentChart.warnings.map((warning, idx) => (
                                    <p key={idx} className="flex items-center gap-1.5 text-xs text-yellow-400">
                                      <AlertTriangle className="h-3 w-3" />
                                      {warning}
                                    </p>
                                  ))}
                                </div>
                              )}
                            </div>
                            <button
                              onClick={() => setShowReasoning(false)}
                              className="text-muted-foreground hover:text-foreground"
                            >
                              <X className="h-4 w-4" />
                            </button>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                  
                  {viewMode === 'chart' ? <SmartChart /> : <DataTable />}
                </motion.div>
              ) : (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex h-[400px] flex-col items-center justify-center rounded-xl border border-dashed border-border/50 bg-card/30"
                >
                  {dataset ? (
                    <>
                      <div className="mb-4 rounded-full bg-primary/10 p-4">
                        <Sparkles className="h-8 w-8 text-primary" />
                      </div>
                      <h3 className="text-lg font-medium text-foreground">
                        Ready to visualize!
                      </h3>
                      <p className="mt-2 text-center text-sm text-muted-foreground">
                        {builderMode === 'ai' 
                          ? 'Ask a question below to generate a chart'
                          : 'Use the Chart Builder on the right to create a chart'}
                      </p>
                    </>
                  ) : (
                    <>
                      <div className="mb-4 rounded-full bg-muted/50 p-4">
                        <BarChart3 className="h-8 w-8 text-muted-foreground" />
                      </div>
                      <h3 className="text-lg font-medium text-muted-foreground">
                        No data loaded
                      </h3>
                      <p className="mt-2 text-center text-sm text-muted-foreground/60">
                        Upload a CSV file above to get started
                      </p>
                    </>
                  )}
                </motion.div>
              )}
            </div>

            {/* Manual Builder Panel */}
            {builderMode === 'manual' && dataset && (
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                className="w-80 shrink-0 rounded-xl border border-border/50 bg-card/50 p-6 backdrop-blur-sm"
              >
                <h3 className="mb-4 flex items-center gap-2 text-lg font-semibold">
                  <Wrench className="h-5 w-5" />
                  Chart Builder
                </h3>
                <ChartBuilder />
              </motion.div>
            )}
          </div>
        </div>

        {/* Chat Input (AI mode only) */}
        {builderMode === 'ai' && (
          <div className="relative border-t border-border/50 bg-background/80 p-6 backdrop-blur-xl">
            <div className="mx-auto max-w-4xl">
              <ChatInterface />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
