'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, FileSpreadsheet, AlertCircle, Loader2, Database, Sparkles, AlertTriangle, TrendingUp, Info } from 'lucide-react';
import { useData } from '@/context/DataContext';
import { previewImport, previewDemoImport, updateImportPreview, confirmImport, cancelImport, aggregateData } from '@/lib/api';
import { ImportPreviewResponse, ImportSettings, VersionedUploadResponse } from '@/types';
import { toast } from 'sonner';
import { formatValue } from '@/lib/formatValue';
import { getColumnDisplayName, getColumnSourceName } from '@/lib/columnLabels';
import { ImportPreview } from '@/components/ImportPreview';
import { DataReview } from '@/components/DataReview';

type DemoDataset = 'taxi' | 'gapminder';

export function FileUploader() {
  const { dataset, enrichment, setDataset, setCurrentChart, addToHistory, setIsUploading, isUploading, clearData, beginQuery, isCurrentQuery, finishQuery } = useData();
  const [aiColumnAnalysis, setAiColumnAnalysis] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [isDemoLoading, setIsDemoLoading] = useState(false);
  const [importPreview, setImportPreview] = useState<ImportPreviewResponse | null>(null);
  const [previewAction, setPreviewAction] = useState<'recheck' | 'confirm' | 'cancel' | null>(null);
  const [previewActionError, setPreviewActionError] = useState<string | null>(null);
  const [isDataReviewOpen, setIsDataReviewOpen] = useState(false);
  const [isImportReviewOpen, setIsImportReviewOpen] = useState(false);
  const [isQualityInfoOpen, setIsQualityInfoOpen] = useState(false);
  const uploadRequestRef = useRef(0);
  const stagedImportRef = useRef<string | null>(null);
  useEffect(() => () => {
    uploadRequestRef.current += 1;
    if (stagedImportRef.current) void cancelImport(stagedImportRef.current).catch(() => {});
  }, []);

  const applyUploadResponse = useCallback(async (response: VersionedUploadResponse, uploadRequestId: number) => {
    if (uploadRequestRef.current !== uploadRequestId) return;
    setDataset({
      datasetId: response.dataset_id,
      version: response.version,
      filename: response.filename,
      rowCount: response.row_count,
      columns: response.columns,
      columnFormats: response.column_formats,
      dataHealth: response.data_health,
      profile: response.profile,
      defaultChart: response.default_chart,
      suggestions: response.suggestions,
      summary: response.summary,
      enrichmentStatus: response.enrichment_status,
    });

    // Auto-render default chart if available
    if (response.default_chart) {
      const controller = new AbortController();
      const queryRequestId = beginQuery(response.dataset_id, controller);
      try {
        const chartData = await aggregateData({
          dataset_id: response.dataset_id,
          x_axis_key: response.default_chart.x_axis_key,
          y_axis_keys: response.default_chart.y_axis_keys,
          aggregation: response.default_chart.aggregation,
          chart_type: response.default_chart.chart_type,
          signal: controller.signal,
        });
        
        if (uploadRequestRef.current !== uploadRequestId || controller.signal.aborted
          || !isCurrentQuery(queryRequestId, response.dataset_id)) return;
        chartData.analysis = response.default_chart.analysis;
        setCurrentChart(chartData);
        addToHistory('Auto-generated insight', chartData, false);
        
        if (chartData.chart_type === 'empty') {
          toast.message('The default chart needs clarification. See the response below.');
        } else {
          toast.success('Data loaded with instant insight!', {
            description: response.default_chart.title,
          });
        }
      } catch (e) {
        if (uploadRequestRef.current === uploadRequestId && !controller.signal.aborted
          && isCurrentQuery(queryRequestId, response.dataset_id)) {
          console.error('Default chart error:', e);
          toast.success(`Loaded ${response.row_count.toLocaleString()} rows`);
        }
      } finally {
        finishQuery(queryRequestId);
      }
    } else if (response.data_health.cleaning_actions.length > 0) {
      toast.success(`Data cleaned: ${response.data_health.cleaning_actions.length} improvements`);
    } else {
      toast.success(`Loaded ${response.row_count.toLocaleString()} rows`);
    }
  }, [setDataset, setCurrentChart, addToHistory, beginQuery, isCurrentQuery, finishQuery]);

  const importAutomatically = useCallback(async (preview: ImportPreviewResponse, requestId: number) => {
    if (uploadRequestRef.current !== requestId) {
      void cancelImport(preview.import_id).catch(() => {});
      return;
    }
    stagedImportRef.current = preview.import_id;
    setImportPreview(preview);
    if (!preview.can_confirm) {
      setUploadError('This file needs a parsing change. Open Review data to adjust its settings.');
      return;
    }
    try {
      const response = await confirmImport(preview.import_id, preview.settings, aiColumnAnalysis);
      if (uploadRequestRef.current !== requestId) return;
      stagedImportRef.current = null;
      setImportPreview(null);
      await applyUploadResponse(response, requestId);
    } catch (error) {
      if (uploadRequestRef.current === requestId) {
        setUploadError(error instanceof Error ? error.message : 'The file could not be prepared.');
      }
    }
  }, [applyUploadResponse, aiColumnAnalysis]);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setUploadError(null);
    setPreviewActionError(null);
    setIsImportReviewOpen(false);
    if (stagedImportRef.current) void cancelImport(stagedImportRef.current).catch(() => {});
    stagedImportRef.current = null;
    setImportPreview(null);
    setIsUploading(true);
    const uploadRequestId = ++uploadRequestRef.current;

    try {
      const response = await previewImport(file);
      await importAutomatically(response, uploadRequestId);
    } catch (error) {
      if (uploadRequestRef.current === uploadRequestId) {
        const message = error instanceof Error ? error.message : 'Upload failed';
        setUploadError(message);
        toast.error(message);
      }
    } finally {
      if (uploadRequestRef.current === uploadRequestId) setIsUploading(false);
    }
  }, [setIsUploading, importAutomatically]);

  const handleDemoLoad = useCallback(async (dataset: DemoDataset) => {
    setUploadError(null);
    setPreviewActionError(null);
    setIsImportReviewOpen(false);
    if (stagedImportRef.current) void cancelImport(stagedImportRef.current).catch(() => {});
    stagedImportRef.current = null;
    setImportPreview(null);
    setIsDemoLoading(true);
    setIsUploading(true);
    const uploadRequestId = ++uploadRequestRef.current;

    try {
      const response = await previewDemoImport(dataset);
      await importAutomatically(response, uploadRequestId);
    } catch (error) {
      if (uploadRequestRef.current === uploadRequestId) {
        const message = error instanceof Error ? error.message : 'Demo load failed';
        setUploadError(message);
        toast.error(message);
      }
    } finally {
      if (uploadRequestRef.current === uploadRequestId) {
        setIsDemoLoading(false);
        setIsUploading(false);
      }
    }
  }, [setIsUploading, importAutomatically]);

  const handleRecheckPreview = useCallback(async (settings: ImportSettings) => {
    if (!importPreview || previewAction) return;
    const requestId = uploadRequestRef.current;
    const importId = importPreview.import_id;
    setPreviewAction('recheck');
    setPreviewActionError(null);
    try {
      const updated = await updateImportPreview(importId, settings);
      if (uploadRequestRef.current === requestId && importId === updated.import_id) setImportPreview(updated);
    } catch (error) {
      if (uploadRequestRef.current === requestId) setPreviewActionError(error instanceof Error ? error.message : 'Could not recheck the sample.');
    } finally {
      if (uploadRequestRef.current === requestId) setPreviewAction(null);
    }
  }, [importPreview, previewAction]);

  const handleConfirmPreview = useCallback(async (settings: ImportSettings) => {
    if (!importPreview || previewAction || !importPreview.can_confirm) return;
    const requestId = uploadRequestRef.current;
    setPreviewAction('confirm');
    setPreviewActionError(null);
    try {
      const response = await confirmImport(importPreview.import_id, settings, aiColumnAnalysis);
      if (uploadRequestRef.current !== requestId) return;
      stagedImportRef.current = null;
      setImportPreview(null);
      setIsImportReviewOpen(false);
      setUploadError(null);
      setIsDataReviewOpen(false);
      await applyUploadResponse(response, requestId);
    } catch (error) {
      if (uploadRequestRef.current === requestId) setPreviewActionError(error instanceof Error ? error.message : 'Could not confirm this import.');
    } finally {
      if (uploadRequestRef.current === requestId) setPreviewAction(null);
    }
  }, [importPreview, previewAction, applyUploadResponse, aiColumnAnalysis]);

  const handleCancelPreview = useCallback(async () => {
    if (!importPreview || previewAction) return;
    const stagedImportId = importPreview.import_id;
    stagedImportRef.current = null;
    const requestId = ++uploadRequestRef.current;
    setPreviewAction('cancel');
    setIsUploading(true);
    setImportPreview(null);
    setPreviewActionError(null);
    setIsImportReviewOpen(false);
    setUploadError(null);
    try {
      await cancelImport(stagedImportId);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Could not remove the staged upload.';
      setUploadError(message);
      toast.error(message);
    } finally {
      if (uploadRequestRef.current === requestId) {
        setPreviewAction(null);
        setIsUploading(false);
      }
    }
  }, [importPreview, previewAction, setIsUploading]);

  const handleSchemaApplied = useCallback(async (response: VersionedUploadResponse, previousVersion?: string) => {
    const uploadRequestId = ++uploadRequestRef.current;
    setIsUploading(true);
    try {
      await applyUploadResponse(response, uploadRequestId);
      if (previousVersion !== response.version) {
        toast.message('Data settings updated', {
          description: 'Dashboard snapshots from the previous schema were cleared. History entries from that version are marked old and disabled.',
        });
      }
    } finally {
      setIsUploading(false);
    }
  }, [applyUploadResponse, setIsUploading]);

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
                  {(dataset.enrichmentStatus === 'pending' || dataset.enrichmentStatus === 'running') && (
                    <span className="text-xs text-muted-foreground">AI insights are being prepared</span>
                  )}
                  {dataset.enrichmentStatus === 'error' && !dataset.summary && (
                    <span className="text-xs text-muted-foreground">AI insights could not be prepared</span>
                  )}
                  {enrichment?.dataset_id === dataset.datasetId && enrichment.coverage && (
                    <span className="text-xs text-muted-foreground" aria-live="polite">
                      AI reviewed {enrichment.coverage.completed_columns} of {enrichment.coverage.total_columns} columns
                      {enrichment.status === 'done' && !enrichment.coverage.complete ? ' · Some insights are unavailable' : ''}
                    </span>
                  )}
                  <button
                    type="button"
                    onClick={() => setIsDataReviewOpen(true)}
                    className="rounded-md border border-border/60 bg-card/40 px-2 py-0.5 text-xs font-medium text-foreground hover:border-primary/40 hover:bg-primary/10"
                  >
                    Review data
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
                    <p>Quality = 100 - (missing cells / total cells), based on source missing values. Missing values are preserved.</p>
                    <p className="mt-1">Missing cells: {missingCells.toLocaleString()} / {totalCells.toLocaleString()}</p>
                    <p className="mt-1 text-muted-foreground/80">
                      This score currently reflects completeness, not outliers or duplicates.
                    </p>
                  </div>
                )}
                {/* Executive Summary with smart formatting */}
                {profile.top_metrics.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-3">
                    {profile.top_metrics.slice(0, 2).map(m => {
                      const isAverage = m.aggregation === 'mean';
                      const value = isAverage ? m.average : m.total;
                      const column = dataset.columns.find(item => item.name === m.name) ?? { name: m.name };
                      return (
                        <div key={m.name} className="flex items-center gap-1.5 text-xs text-muted-foreground">
                          <TrendingUp className="h-3 w-3 text-primary" />
                          <span className="font-medium" title={getColumnSourceName(column)}>{getColumnDisplayName(column)}:</span>
                          <span>{formatValue(value, dataset.columnFormats[m.name] || 'number', { compact: true })} {isAverage ? 'average' : 'total'}</span>
                        </div>
                      );
                    })}
                    {profile.time_range && (
                      <div className="text-xs text-muted-foreground">
                        📅 {profile.time_range.start.slice(0, 10)} → {profile.time_range.end.slice(0, 10)}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
            <button onClick={() => { uploadRequestRef.current += 1; setIsDataReviewOpen(false); clearData(); }} className="rounded-lg px-4 py-2 text-sm font-medium text-muted-foreground hover:bg-white/5 hover:text-white">
              Upload New
            </button>
          </div>
        </div>
        <DataReview
          open={isDataReviewOpen}
          datasetId={dataset.datasetId}
          datasetVersion={dataset.version}
          currentColumns={dataset.columns}
          dataHealth={dataHealth}
          rowCount={dataset.rowCount}
          onClose={() => setIsDataReviewOpen(false)}
          onApplied={handleSchemaApplied}
        />
      </motion.div>
    );
  }

  if (importPreview && isImportReviewOpen) {
    return (
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="w-full">
        <button type="button" onClick={() => setIsImportReviewOpen(false)} className="mb-3 text-sm text-muted-foreground">Close review</button>
        <ImportPreview
          key={`${importPreview.import_id}:${JSON.stringify(importPreview.settings)}`}
          preview={importPreview}
          busy={previewAction !== null}
          actionError={previewActionError}
          onRecheck={handleRecheckPreview}
          onConfirm={handleConfirmPreview}
          onCancel={handleCancelPreview}
        />
      </motion.div>
    );
  }

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="w-full">
      <div className="mb-3 rounded-xl border border-border/50 bg-card/40 p-3">
        <button
          type="button"
          role="switch"
          aria-checked={aiColumnAnalysis}
          aria-describedby="ai-column-analysis-description"
          disabled={isUploading || isDemoLoading || importPreview !== null}
          onClick={() => setAiColumnAnalysis(enabled => !enabled)}
          className="flex w-full items-center justify-between gap-3 text-left text-sm font-medium disabled:opacity-60"
        >
          <span className="inline-flex items-center gap-2"><Sparkles className="h-4 w-4 text-primary" />AI column semantic analysis</span>
          <span className={`rounded-full px-3 py-1 text-xs ${aiColumnAnalysis ? 'bg-primary/20 text-primary' : 'bg-muted text-muted-foreground'}`}>{aiColumnAnalysis ? 'On' : 'Off'}</span>
        </button>
        <p id="ai-column-analysis-description" className="mt-1 text-xs text-muted-foreground">
          {aiColumnAnalysis ? 'Luna interprets columns in the background and applies compatible roles and readable labels automatically. Sampled values are sent to your AI provider. Review data to make changes.' : 'Use automatic local column detection. Turn on AI for additional column interpretation suggestions.'}
        </p>
      </div>
      <div {...getRootProps()} className={`relative cursor-pointer rounded-xl border-2 border-dashed p-8 text-center transition-all duration-300
        ${isDragActive ? 'border-primary bg-primary/5 scale-[1.02]' : 'border-border/50 hover:border-primary/50 hover:bg-white/[0.02]'}
        ${isUploading ? 'pointer-events-none opacity-60' : ''} ${uploadError ? 'border-destructive/50' : ''}`}>
        <input {...getInputProps()} />
        <AnimatePresence mode="wait">
          {isUploading ? (
            <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col items-center gap-3">
              <Loader2 className="h-10 w-10 animate-spin text-primary" />
              <p className="font-medium">Preparing your dataset…</p>
              <p className="text-xs text-muted-foreground animate-pulse">Reading, validating and preparing columns for charting.</p>
            </motion.div>
          ) : uploadError ? (
            <motion.div key="error" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col items-center gap-3">
              <AlertCircle className="h-10 w-10 text-destructive" />
              <p className="font-medium text-destructive">{uploadError}</p>
              {importPreview && <button type="button" onClick={event => { event.stopPropagation(); setIsImportReviewOpen(true); }} className="rounded-lg border border-border px-4 py-2 text-sm text-foreground">Review data</button>}
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
                <p className="font-medium">Drop a CSV to start charting</p>
                <p className="mt-1 text-sm text-muted-foreground">Prepared automatically. Review data settings whenever you need.</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
      <div className="mt-4 space-y-2">
        <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Try a demo dataset</p>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          <button
            type="button"
            onClick={() => handleDemoLoad('taxi')}
            disabled={isDemoLoading || isUploading}
            className="inline-flex items-center justify-center gap-2 rounded-xl border border-primary/40 bg-gradient-to-r from-primary/15 via-primary/10 to-transparent px-4 py-3 text-sm font-medium text-primary transition-all hover:border-primary/60 hover:from-primary/20 hover:via-primary/15 disabled:cursor-not-allowed disabled:opacity-60"
          >
            NYC Taxi (1M)
          </button>
          <button
            type="button"
            onClick={() => handleDemoLoad('gapminder')}
            disabled={isDemoLoading || isUploading}
            className="inline-flex items-center justify-center gap-2 rounded-xl border border-primary/40 bg-gradient-to-r from-primary/15 via-primary/10 to-transparent px-4 py-3 text-sm font-medium text-primary transition-all hover:border-primary/60 hover:from-primary/20 hover:via-primary/15 disabled:cursor-not-allowed disabled:opacity-60"
          >
            Gapminder
          </button>
        </div>
      </div>
    </motion.div>
  );
}
