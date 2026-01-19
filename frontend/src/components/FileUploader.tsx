'use client';

import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, FileSpreadsheet, CheckCircle, AlertCircle, Loader2, Database, Sparkles, AlertTriangle } from 'lucide-react';
import { useData } from '@/context/DataContext';
import { uploadCSV } from '@/lib/api';
import { toast } from 'sonner';

export function FileUploader() {
  const { dataset, setDataset, setIsUploading, isUploading, clearData } = useData();
  const [uploadError, setUploadError] = useState<string | null>(null);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setUploadError(null);
    setIsUploading(true);

    try {
      const response = await uploadCSV(file);
      
      setDataset({
        datasetId: response.dataset_id,
        filename: response.filename,
        rowCount: response.row_count,
        columns: response.columns,
        dataHealth: response.data_health,
      });

      // Show cleaning actions if any
      if (response.data_health.cleaning_actions.length > 0) {
        toast.success(`Data cleaned: ${response.data_health.cleaning_actions.length} improvements made`, {
          description: response.data_health.cleaning_actions[0],
        });
      } else {
        toast.success(`Loaded ${response.row_count.toLocaleString()} rows`);
      }
    } catch (error) {
      console.error('Upload error:', error);
      const message = error instanceof Error ? error.message : 'Failed to upload file';
      setUploadError(message);
      toast.error(message);
    } finally {
      setIsUploading(false);
    }
  }, [setDataset, setIsUploading]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'] },
    maxFiles: 1,
    disabled: isUploading,
  });

  // If we have data, show success state with health info
  if (dataset) {
    const { dataHealth } = dataset;
    const hasCleaning = dataHealth.cleaning_actions.length > 0;
    const missingCount = Object.values(dataHealth.missing_values).reduce((a, b) => a + b, 0);
    const totalCells = dataset.rowCount * dataset.columns.length;
    const missingPercent = totalCells > 0 ? (missingCount / totalCells) * 100 : 0;
    const hasWarning = missingPercent > 10;
    
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="w-full"
      >
        <div className={`relative rounded-xl border p-4 ${
          hasWarning 
            ? 'border-yellow-500/30 bg-yellow-500/5' 
            : 'border-emerald-500/30 bg-emerald-500/5'
        }`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className={`flex h-12 w-12 items-center justify-center rounded-full ${
                hasWarning ? 'bg-yellow-500/20' : 'bg-emerald-500/20'
              }`}>
                {hasWarning ? (
                  <AlertTriangle className="h-6 w-6 text-yellow-400" />
                ) : (
                  <Database className="h-6 w-6 text-emerald-400" />
                )}
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h3 className={`font-semibold ${hasWarning ? 'text-yellow-300' : 'text-emerald-300'}`}>
                    {dataset.filename}
                  </h3>
                  {hasCleaning && (
                    <span className="flex items-center gap-1 rounded-full bg-primary/20 px-2 py-0.5 text-xs font-medium text-primary">
                      <Sparkles className="h-3 w-3" />
                      Data Cleaned
                    </span>
                  )}
                </div>
                <p className="text-sm text-muted-foreground">
                  {dataset.rowCount.toLocaleString()} rows • {dataset.columns.length} columns
                  {dataHealth.quality_score < 100 && (
                    <span className={`ml-2 ${hasWarning ? 'text-yellow-400' : ''}`}>
                      • Quality: {dataHealth.quality_score.toFixed(0)}%
                    </span>
                  )}
                </p>
                {hasCleaning && (
                  <p className="mt-1 text-xs text-muted-foreground/70">
                    {dataHealth.cleaning_actions[0]}
                    {dataHealth.cleaning_actions.length > 1 && ` +${dataHealth.cleaning_actions.length - 1} more`}
                  </p>
                )}
              </div>
            </div>
            <button
              onClick={clearData}
              className="rounded-lg px-4 py-2 text-sm font-medium text-muted-foreground transition-colors hover:bg-white/5 hover:text-white"
            >
              Upload New
            </button>
          </div>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full"
    >
      <div
        {...getRootProps()}
        className={`
          relative cursor-pointer rounded-xl border-2 border-dashed p-8 text-center transition-all duration-300
          ${isDragActive ? 'border-primary bg-primary/5 scale-[1.02]' : 'border-border/50 hover:border-primary/50 hover:bg-white/[0.02]'}
          ${isUploading ? 'pointer-events-none opacity-60' : ''}
          ${uploadError ? 'border-destructive/50' : ''}
        `}
      >
        <input {...getInputProps()} />
        
        <AnimatePresence mode="wait">
          {isUploading ? (
            <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col items-center gap-3">
              <Loader2 className="h-10 w-10 animate-spin text-primary" />
              <p className="font-medium">Processing & cleaning data...</p>
            </motion.div>
          ) : uploadError ? (
            <motion.div key="error" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col items-center gap-3">
              <AlertCircle className="h-10 w-10 text-destructive" />
              <p className="font-medium text-destructive">{uploadError}</p>
            </motion.div>
          ) : isDragActive ? (
            <motion.div key="drag" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col items-center gap-3">
              <FileSpreadsheet className="h-10 w-10 text-primary" />
              <p className="font-medium text-primary">Drop your CSV here</p>
            </motion.div>
          ) : (
            <motion.div key="default" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col items-center gap-3">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-primary/20 to-primary/5">
                <Upload className="h-7 w-7 text-primary" />
              </div>
              <div>
                <p className="font-medium">Drag & drop your CSV file</p>
                <p className="mt-1 text-sm text-muted-foreground">We'll auto-clean and validate your data</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
}
