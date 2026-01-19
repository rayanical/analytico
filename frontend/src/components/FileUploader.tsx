'use client';

import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, FileSpreadsheet, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import { useData } from '@/context/DataContext';
import { uploadCSV } from '@/lib/api';
import { toast } from 'sonner';

export function FileUploader() {
  const { dataState, setDataState, setIsUploading, isUploading, clearData } = useData();
  const [uploadError, setUploadError] = useState<string | null>(null);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setUploadError(null);
    setIsUploading(true);

    try {
      const response = await uploadCSV(file);
      
      setDataState({
        data: response.data,
        columns: response.columns,
        summary: response.summary,
        fileName: file.name,
        rowCount: response.row_count,
        truncated: response.truncated,
      });

      if (response.truncated) {
        toast.warning(`File truncated to ${response.row_count} rows to optimize performance.`);
      } else {
        toast.success(`Successfully loaded ${response.row_count} rows from ${file.name}`);
      }
    } catch (error) {
      console.error('Upload error:', error);
      const message = error instanceof Error ? error.message : 'Failed to upload file';
      setUploadError(message);
      toast.error(message);
    } finally {
      setIsUploading(false);
    }
  }, [setDataState, setIsUploading]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
    },
    maxFiles: 1,
    disabled: isUploading,
  });

  // If we have data, show success state
  if (dataState) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="w-full"
      >
        <div className="relative rounded-xl border border-emerald-500/30 bg-emerald-500/5 p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-emerald-500/20">
                <CheckCircle className="h-6 w-6 text-emerald-400" />
              </div>
              <div>
                <h3 className="font-semibold text-emerald-300">{dataState.fileName}</h3>
                <p className="text-sm text-muted-foreground">
                  {dataState.rowCount.toLocaleString()} rows • {dataState.columns.length} columns
                  {dataState.truncated && ' (truncated)'}
                </p>
              </div>
            </div>
            <button
              onClick={clearData}
              className="rounded-lg px-4 py-2 text-sm font-medium text-muted-foreground transition-colors hover:bg-white/5 hover:text-white"
            >
              Upload New File
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
          relative cursor-pointer rounded-xl border-2 border-dashed p-12 text-center transition-all duration-300
          ${isDragActive 
            ? 'border-primary bg-primary/5 scale-[1.02]' 
            : 'border-border/50 hover:border-primary/50 hover:bg-white/[0.02]'
          }
          ${isUploading ? 'pointer-events-none opacity-60' : ''}
          ${uploadError ? 'border-destructive/50' : ''}
        `}
      >
        <input {...getInputProps()} />
        
        <AnimatePresence mode="wait">
          {isUploading ? (
            <motion.div
              key="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center gap-4"
            >
              <Loader2 className="h-12 w-12 animate-spin text-primary" />
              <p className="text-lg font-medium">Processing your data...</p>
            </motion.div>
          ) : uploadError ? (
            <motion.div
              key="error"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center gap-4"
            >
              <AlertCircle className="h-12 w-12 text-destructive" />
              <p className="text-lg font-medium text-destructive">{uploadError}</p>
              <p className="text-sm text-muted-foreground">Click or drag to try again</p>
            </motion.div>
          ) : isDragActive ? (
            <motion.div
              key="drag"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center gap-4"
            >
              <FileSpreadsheet className="h-12 w-12 text-primary" />
              <p className="text-lg font-medium text-primary">Drop your CSV file here</p>
            </motion.div>
          ) : (
            <motion.div
              key="default"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center gap-4"
            >
              <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-primary/20 to-primary/5">
                <Upload className="h-8 w-8 text-primary" />
              </div>
              <div>
                <p className="text-lg font-medium">
                  Drag & drop your CSV file here
                </p>
                <p className="mt-1 text-sm text-muted-foreground">
                  or click to browse files
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
}
