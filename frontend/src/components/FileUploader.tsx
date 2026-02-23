'use client';

import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, FileSpreadsheet, AlertCircle, Loader2, Database, Sparkles, AlertTriangle, TrendingUp, Info } from 'lucide-react';
import { useData } from '@/context/DataContext';
import { uploadCSV, loadDemoDataset, aggregateData } from '@/lib/api';
import { UploadResponse } from '@/types';
import { toast } from 'sonner';
import { CleaningReportDrawer } from '@/components/CleaningReportDrawer';

// Smart number formatter with proper K/M/B scaling
function formatCompact(value: number, format: string = 'number'): string {
  const abs = Math.abs(value);
  const prefix = format === 'currency' ? '$' : '';
  const suffix = format === 'percentage' ? '%' : '';
  
  if (abs >= 1e9) return `${prefix}${(value / 1e9).toFixed(1)}B${suffix}`;
  if (abs >= 1e6) return `${prefix}${(value / 1e6).toFixed(1)}M${suffix}`;
  if (abs >= 1e3) return `${prefix}${(value / 1e3).toFixed(1)}K${suffix}`;
  
  if (format === 'percentage') return `${(value * 100).toFixed(0)}%`;
  return `${prefix}${value.toFixed(0)}${suffix}`;
}

// Format column name for display
function formatName(name: string): string {
  return name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase());
}

export function FileUploader() {
  const { dataset, setDataset, setCurrentChart, addToHistory, setIsUploading, isUploading, clearData } = useData();
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [isDemoLoading, setIsDemoLoading] = useState(false);
  const [isCleaningReportOpen, setIsCleaningReportOpen] = useState(false);
  const [isQualityInfoOpen, setIsQualityInfoOpen] = useState(false);

  const applyUploadResponse = useCallback(async (response: UploadResponse) => {
    setDataset({
      datasetId: response.dataset_id,
      filename: response.filename,
      rowCount: response.row_count,
      columns: response.columns,
      columnFormats: response.column_formats,
      dataHealth: response.data_health,
      profile: response.profile,
      defaultChart: response.default_chart,
      suggestions: response.suggestions,
      summary: response.summary,
    });

    // Auto-render default chart if available
    if (response.default_chart) {
      try {
        const chartData = await aggregateData({
          dataset_id: response.dataset_id,
          x_axis_key: response.default_chart.x_axis_key,
          y_axis_keys: response.default_chart.y_axis_keys,
          aggregation: response.default_chart.aggregation,
          chart_type: response.default_chart.chart_type,
        });
        
        chartData.analysis = response.default_chart.analysis;
        setCurrentChart(chartData);
        addToHistory('Auto-generated insight', chartData, false);
        
        toast.success('Data loaded with instant insight!', {
          description: response.default_chart.title,
        });
      } catch (e) {
        console.error('Default chart error:', e);
        toast.success(`Loaded ${response.row_count.toLocaleString()} rows`);
      }
    } else if (response.data_health.cleaning_actions.length > 0) {
      toast.success(`Data cleaned: ${response.data_health.cleaning_actions.length} improvements`);
    } else {
      toast.success(`Loaded ${response.row_count.toLocaleString()} rows`);
    }
  }, [setDataset, setCurrentChart, addToHistory]);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setUploadError(null);
    setIsUploading(true);

    try {
      const response = await uploadCSV(file);
      await applyUploadResponse(response);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Upload failed';
      setUploadError(message);
      toast.error(message);
    } finally {
      setIsUploading(false);
    }
  }, [setIsUploading, applyUploadResponse]);

  const handleDemoLoad = useCallback(async () => {
    setUploadError(null);
    setIsDemoLoading(true);
    setIsUploading(true);

    try {
      const response = await loadDemoDataset();
      await applyUploadResponse(response);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Demo load failed';
      setUploadError(message);
      toast.error(message);
    } finally {
      setIsDemoLoading(false);
      setIsUploading(false);
    }
  }, [setIsUploading, applyUploadResponse]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { 'text/csv': ['.csv'] }, maxFiles: 1, disabled: isUploading || isDemoLoading,
  });

  if (dataset) {
    const { dataHealth, profile } = dataset;
    const hasCleaning = dataHealth.cleaning_actions.length > 0;
    const hasWarning = dataHealth.quality_score < 90;
    const missingCells = Object.values(dataHealth.missing_values).reduce((sum, count) => sum + count, 0);
    const totalCells = dataset.rowCount * dataset.columns.length;
    
    return (
      <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="w-full">
        <div className={`rounded-xl border p-4 ${hasWarning ? 'border-yellow-500/30 bg-yellow-500/5' : 'border-emerald-500/30 bg-emerald-500/5'}`}>
          <div className="flex items-start justify-between">
            <div className="flex items-start gap-4">
              <div className={`flex h-12 w-12 items-center justify-center rounded-full ${hasWarning ? 'bg-yellow-500/20' : 'bg-emerald-500/20'}`}>
                {hasWarning ? <AlertTriangle className="h-6 w-6 text-yellow-400" /> : <Database className="h-6 w-6 text-emerald-400" />}
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h3 className={`font-semibold ${hasWarning ? 'text-yellow-300' : 'text-emerald-300'}`}>{dataset.filename}</h3>
                  {hasCleaning && (
                    <span className="flex items-center gap-1 rounded-full bg-primary/20 px-2 py-0.5 text-xs font-medium text-primary">
                      <Sparkles className="h-3 w-3" />Data Cleaned
                    </span>
                  )}
                  <button
                    type="button"
                    onClick={() => setIsCleaningReportOpen(true)}
                    className="rounded-md border border-border/60 bg-card/40 px-2 py-0.5 text-xs font-medium text-foreground hover:border-primary/40 hover:bg-primary/10"
                  >
                    View Cleaning Report
                  </button>
                </div>
                <p className="text-sm text-muted-foreground">
                  {dataset.rowCount.toLocaleString()} rows • {dataset.columns.length} cols • {dataHealth.quality_score.toFixed(0)}% quality
                  <button
                    type="button"
                    onClick={() => setIsQualityInfoOpen((open) => !open)}
                    className="ml-2 inline-flex align-middle text-muted-foreground/80 hover:text-foreground"
                    aria-label="Show quality score details"
                  >
                    <Info className="h-3.5 w-3.5" />
                  </button>
                </p>
                {isQualityInfoOpen && (
                  <div className="mt-2 rounded-md border border-border/50 bg-card/40 p-2 text-xs text-muted-foreground">
                    <p>Quality = 100 - (missing cells / total cells), measured before imputations.</p>
                    <p className="mt-1">Missing cells: {missingCells.toLocaleString()} / {totalCells.toLocaleString()}</p>
                    <p className="mt-1 text-muted-foreground/80">
                      This score currently reflects completeness, not outliers or duplicates.
                    </p>
                  </div>
                )}
                {/* Executive Summary with smart formatting */}
                {profile.top_metrics.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-3">
                    {profile.top_metrics.slice(0, 2).map(m => (
                      <div key={m.name} className="flex items-center gap-1.5 text-xs text-muted-foreground">
                        <TrendingUp className="h-3 w-3 text-primary" />
                        <span className="font-medium">{formatName(m.name)}:</span>
                        <span>{formatCompact(m.total, dataset.columnFormats[m.name] || 'number')} total</span>
                      </div>
                    ))}
                    {profile.time_range && (
                      <div className="text-xs text-muted-foreground">
                        📅 {profile.time_range.start.slice(0, 10)} → {profile.time_range.end.slice(0, 10)}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
            <button onClick={clearData} className="rounded-lg px-4 py-2 text-sm font-medium text-muted-foreground hover:bg-white/5 hover:text-white">
              Upload New
            </button>
          </div>
        </div>
        <CleaningReportDrawer
          open={isCleaningReportOpen}
          onClose={() => setIsCleaningReportOpen(false)}
          dataHealth={dataHealth}
          rowCount={dataset.rowCount}
          columnCount={dataset.columns.length}
        />
      </motion.div>
    );
  }

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="w-full">
      <div {...getRootProps()} className={`relative cursor-pointer rounded-xl border-2 border-dashed p-8 text-center transition-all duration-300
        ${isDragActive ? 'border-primary bg-primary/5 scale-[1.02]' : 'border-border/50 hover:border-primary/50 hover:bg-white/[0.02]'}
        ${isUploading ? 'pointer-events-none opacity-60' : ''} ${uploadError ? 'border-destructive/50' : ''}`}>
        <input {...getInputProps()} />
        <AnimatePresence mode="wait">
          {isUploading ? (
            <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col items-center gap-3">
              <Loader2 className="h-10 w-10 animate-spin text-primary" />
              <p className="font-medium">Processing dataset...</p>
              <p className="text-xs text-muted-foreground animate-pulse">Please wait, preparing your dashboard...</p>
            </motion.div>
          ) : uploadError ? (
            <motion.div key="error" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col items-center gap-3">
              <AlertCircle className="h-10 w-10 text-destructive" />
              <p className="font-medium text-destructive">{uploadError}</p>
            </motion.div>
          ) : isDragActive ? (
            <motion.div key="drag" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col items-center gap-3">
              <FileSpreadsheet className="h-10 w-10 text-primary" />
              <p className="font-medium text-primary">Drop to analyze</p>
            </motion.div>
          ) : (
            <motion.div key="default" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col items-center gap-3">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-primary/20 to-primary/5">
                <Upload className="h-7 w-7 text-primary" />
              </div>
              <div>
                <p className="font-medium">Drop any CSV — we&apos;ll handle the rest</p>
                <p className="mt-1 text-sm text-muted-foreground">Auto-clean, profile, and generate instant insights</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
      <button
        type="button"
        onClick={handleDemoLoad}
        disabled={isDemoLoading || isUploading}
        className="mt-4 inline-flex w-full items-center justify-center gap-2 rounded-xl border border-primary/40 bg-gradient-to-r from-primary/15 via-primary/10 to-transparent px-4 py-3 text-sm font-medium text-primary transition-all hover:border-primary/60 hover:from-primary/20 hover:via-primary/15 disabled:cursor-not-allowed disabled:opacity-60"
      >
        {isDemoLoading && <Loader2 className="h-4 w-4 animate-spin" />}
        {isDemoLoading ? 'Loading demo dataset...' : '🚀 Try 1 Million Rows Demo (NYC Taxi)'}
      </button>
    </motion.div>
  );
}
