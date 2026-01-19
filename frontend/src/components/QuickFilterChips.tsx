'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { Plus, X } from 'lucide-react';
import { useData } from '@/context/DataContext';

interface QuickFilterChipsProps {
  /** Columns currently being plotted (will be excluded from quick filter suggestions) */
  plottedColumns?: string[];
  /** Callback when a filter chip is clicked to add a filter */
  onFilterChipClick?: (columnName: string) => void;
}

export function QuickFilterChips({ plottedColumns = [], onFilterChipClick }: QuickFilterChipsProps) {
  const { dataset, filters, removeFilter, categoricalColumns } = useData();

  if (!dataset) return null;

  // Get categorical columns that are NOT currently being plotted
  const availableForFiltering = categoricalColumns
    .filter(col => !plottedColumns.includes(col.name))
    .filter(col => col.unique_count >= 2 && col.unique_count <= 50) // Reasonable filter candidates
    .slice(0, 2); // Top 2 only

  // Format filter values for display
  const formatFilterValues = (values: unknown[] | undefined): string => {
    if (!values || values.length === 0) return '';
    const stringVals = values.map(String);
    if (stringVals.length === 1) return stringVals[0];
    if (stringVals.length === 2) return stringVals.join(', ');
    return `${stringVals[0]}, +${stringVals.length - 1} more`;
  };

  const hasActiveFilters = filters.length > 0;
  const hasAvailableFilters = availableForFiltering.length > 0;

  if (!hasActiveFilters && !hasAvailableFilters) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: -5 }}
      animate={{ opacity: 1, y: 0 }}
      className="mb-4 flex flex-wrap items-center gap-2"
    >
      {/* Active Filter Badges */}
      {hasActiveFilters && (
        <>
          <span className="text-xs font-medium text-muted-foreground">Active:</span>
          {filters.map(filter => (
            <motion.button
              key={filter.column}
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={() => removeFilter(filter.column)}
              className="group flex items-center gap-1.5 rounded-full bg-primary/20 px-3 py-1 text-xs font-medium text-primary transition-all hover:bg-primary/30"
            >
              <span>{filter.column}:</span>
              <span className="font-normal">{formatFilterValues(filter.values)}</span>
              <X className="h-3 w-3 opacity-60 transition-opacity group-hover:opacity-100" />
            </motion.button>
          ))}
        </>
      )}

      {/* Separator when both sections are visible */}
      {hasActiveFilters && hasAvailableFilters && (
        <span className="mx-1 h-4 w-px bg-border/50" />
      )}

      {/* Quick Filter Add Chips */}
      {hasAvailableFilters && (
        <>
          <span className="text-xs font-medium text-muted-foreground">
            {hasActiveFilters ? 'Add:' : 'Filter by:'}
          </span>
          {availableForFiltering.map(col => (
            <button
              key={col.name}
              onClick={() => onFilterChipClick?.(col.name)}
              className="flex items-center gap-1 rounded-full border border-border/50 bg-white/5 px-3 py-1 text-xs font-medium text-muted-foreground transition-all hover:border-primary/50 hover:bg-primary/10 hover:text-foreground"
            >
              <Plus className="h-3 w-3" />
              {col.name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
            </button>
          ))}
        </>
      )}
    </motion.div>
  );
}
