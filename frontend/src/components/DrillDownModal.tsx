'use client';

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, ExternalLink, Download } from 'lucide-react';
import { useData } from '@/context/DataContext';

export function DrillDownModal() {
  const { drillDownData, setDrillDownData, isDrillDownOpen, setIsDrillDownOpen } = useData();

  if (!isDrillDownOpen || !drillDownData) return null;

  const columns = drillDownData.length > 0 ? Object.keys(drillDownData[0]) : [];

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 backdrop-blur-sm"
        onClick={() => setIsDrillDownOpen(false)}
      >
        <motion.div
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.95, opacity: 0 }}
          className="flex h-[80vh] w-full max-w-5xl flex-col overflow-hidden rounded-xl border border-border bg-card shadow-2xl"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-center justify-between border-b border-border p-4 bg-muted/30">
            <div className="flex items-center gap-2">
              <ExternalLink className="h-5 w-5 text-primary" />
              <h3 className="font-semibold">Drill-Down Analysis</h3>
              <span className="rounded-full bg-primary/10 px-2 py-0.5 text-xs text-primary">
                {drillDownData.length} records
              </span>
            </div>
            <button
              onClick={() => setIsDrillDownOpen(false)}
              className="rounded-full p-1 hover:bg-muted"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          {/* Table */}
          <div className="flex-1 overflow-auto p-0">
            <table className="w-full text-sm">
              <thead className="bg-muted/50 sticky top-0 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider backdrop-blur-sm">
                <tr>
                  {columns.map((col) => (
                    <th key={col} className="px-4 py-3 font-medium border-b border-border">
                      {col.replace(/_/g, ' ')}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-border/50 bg-card">
                {drillDownData.map((row, i) => (
                  <tr key={i} className="hover:bg-muted/50 transition-colors">
                    {columns.map((col) => (
                      <td key={col} className="px-4 py-2.5 whitespace-nowrap text-foreground/80">
                        {String(row[col] ?? '-')}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          {/* Footer */}
          <div className="border-t border-border p-3 bg-muted/30 text-xs text-muted-foreground text-center">
            Showing top 50 records matching selected data point.
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
