'use client';

import { motion } from 'framer-motion';
import { Sidebar } from '@/components/Sidebar';
import { FileUploader } from '@/components/FileUploader';
import { ChatInterface } from '@/components/ChatInterface';
import { ChartBuilder } from '@/components/ChartBuilder';
import { SmartChart } from '@/components/SmartChart';
import { ChartSkeleton } from '@/components/ChartSkeleton';
import { DataTable } from '@/components/DataTable';
import { FilterBar } from '@/components/FilterBar';
import { useData } from '@/context/DataContext';
import { BarChart3, Sparkles, Wand2, Wrench, Table, LineChart } from 'lucide-react';

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

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <Sidebar />
      
      {/* Main Content */}
      <main className="relative flex flex-1 flex-col overflow-hidden">
        {/* Background gradient */}
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-primary/5" />
        
        {/* Content area */}
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

          {/* File Upload Section */}
          <section className="mb-6">
            <FileUploader />
          </section>

          {/* Filter Bar (only show if data loaded) */}
          {dataset && (
            <section className="mb-6">
              <FilterBar />
            </section>
          )}

          {/* Mode Tabs and View Toggle */}
          {dataset && (
            <div className="mb-6 flex items-center justify-between">
              {/* Mode Tabs */}
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

              {/* View Toggle (only show if chart exists) */}
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

          {/* Main Content Area */}
          <div className="flex flex-1 gap-6">
            {/* Chart/Table Display */}
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
                  <div className="mb-4 flex items-center justify-between">
                    <h3 className="text-xl font-semibold text-foreground">
                      {currentChart.title}
                    </h3>
                    <span className="text-sm text-muted-foreground">
                      {currentChart.row_count} data points
                    </span>
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

            {/* Manual Builder Panel (only in manual mode) */}
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

        {/* Chat Input (only in AI mode) */}
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
