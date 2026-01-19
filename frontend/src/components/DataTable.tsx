'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useData } from '@/context/DataContext';

const ROWS_PER_PAGE = 10;

export function DataTable() {
  const { currentChart } = useData();
  const [page, setPage] = useState(0);

  if (!currentChart || !currentChart.data.length) {
    return (
      <div className="flex h-64 items-center justify-center rounded-xl border border-dashed border-border/50">
        <p className="text-muted-foreground">No data to display</p>
      </div>
    );
  }

  const { data, x_axis_key, y_axis_keys, title } = currentChart;
  const columns = [x_axis_key, ...y_axis_keys];
  
  const totalPages = Math.ceil(data.length / ROWS_PER_PAGE);
  const startIdx = page * ROWS_PER_PAGE;
  const endIdx = Math.min(startIdx + ROWS_PER_PAGE, data.length);
  const pageData = data.slice(startIdx, endIdx);

  const formatValue = (value: unknown, column: string): string => {
    if (value === null || value === undefined) return '—';
    
    const num = Number(value);
    if (isNaN(num)) return String(value);
    
    // Currency formatting
    const lowerCol = column.toLowerCase();
    if (/sales|revenue|rev|cost|price|amount|profit/.test(lowerCol)) {
      return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        notation: num >= 1000000 ? 'compact' : 'standard',
        maximumFractionDigits: num >= 1000 ? 0 : 2,
      }).format(num);
    }
    
    // Percentage formatting
    if (/rate|percent|ratio|pct/.test(lowerCol)) {
      return `${num.toFixed(1)}%`;
    }
    
    // Compact number formatting
    if (num >= 1000) {
      return new Intl.NumberFormat('en-US', {
        notation: 'compact',
        maximumFractionDigits: 1,
      }).format(num);
    }
    
    return num.toLocaleString();
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-xl border border-border/50 bg-card/50 overflow-hidden"
    >
      {/* Header */}
      <div className="border-b border-border/50 px-6 py-4">
        <h3 className="text-lg font-semibold">{title}</h3>
        <p className="text-sm text-muted-foreground">
          Showing {startIdx + 1}–{endIdx} of {data.length} rows
        </p>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-border/50 bg-muted/30">
              {columns.map(col => (
                <th
                  key={col}
                  className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-muted-foreground"
                >
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-border/30">
            {pageData.map((row, idx) => (
              <motion.tr
                key={idx}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: idx * 0.02 }}
                className="hover:bg-white/[0.02]"
              >
                {columns.map(col => (
                  <td key={col} className="whitespace-nowrap px-6 py-3 text-sm">
                    {formatValue(row[col], col)}
                  </td>
                ))}
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between border-t border-border/50 px-6 py-3">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setPage(p => Math.max(0, p - 1))}
            disabled={page === 0}
          >
            <ChevronLeft className="mr-1 h-4 w-4" />
            Previous
          </Button>
          
          <div className="flex items-center gap-1">
            {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
              let pageNum = i;
              if (totalPages > 5) {
                if (page < 3) pageNum = i;
                else if (page > totalPages - 4) pageNum = totalPages - 5 + i;
                else pageNum = page - 2 + i;
              }
              return (
                <button
                  key={pageNum}
                  onClick={() => setPage(pageNum)}
                  className={`h-8 w-8 rounded text-sm ${
                    page === pageNum
                      ? 'bg-primary text-primary-foreground'
                      : 'hover:bg-white/5'
                  }`}
                >
                  {pageNum + 1}
                </button>
              );
            })}
          </div>
          
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
            disabled={page === totalPages - 1}
          >
            Next
            <ChevronRight className="ml-1 h-4 w-4" />
          </Button>
        </div>
      )}
    </motion.div>
  );
}
