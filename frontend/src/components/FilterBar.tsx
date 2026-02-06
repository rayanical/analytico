'use client';

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Filter, X, ChevronDown, Check } from 'lucide-react';
import { useData } from '@/context/DataContext';
import { FilterConfig } from '@/types';
import { Button } from '@/components/ui/button';

interface FilterBarProps {
  /** Externally controlled column to expand */
  expandColumn?: string | null;
  /** Callback when expansion state changes */
  onExpandChange?: (column: string | null) => void;
}

export function FilterBar({ expandColumn, onExpandChange }: FilterBarProps) {
  const { dataset, filters, setFilters, clearFilters, categoricalColumns } = useData();
  const [expandedColumn, setExpandedColumn] = useState<string | null>(null);
  const [pendingFilters, setPendingFilters] = useState<FilterConfig[]>([]);
  const containerRef = useRef<HTMLDivElement>(null);

  // Initialize pending filters from applied filters
  useEffect(() => {
    setPendingFilters(filters);
  }, [filters]);

  // Compute dirty state (pending !== applied)
  const isDirty = JSON.stringify(pendingFilters) !== JSON.stringify(filters);

  // Sync with external control
  useEffect(() => {
    if (expandColumn !== undefined) {
      setExpandedColumn(expandColumn);
    }
  }, [expandColumn]);

  // Click outside to close dropdown
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        handleExpandChange(null);
      }
    };

    if (expandedColumn) {
      // Add listener with a small delay to prevent immediate close from the click that opened it
      const timeoutId = setTimeout(() => {
        document.addEventListener('mousedown', handleClickOutside);
      }, 100);
      
      return () => {
        clearTimeout(timeoutId);
        document.removeEventListener('mousedown', handleClickOutside);
      };
    }
  }, [expandedColumn]);

  const handleExpandChange = (column: string | null) => {
    setExpandedColumn(column);
    onExpandChange?.(column);
  };

  // Local filter mutation helpers
  const addPendingFilter = (filter: FilterConfig) => {
    setPendingFilters(prev => {
      const idx = prev.findIndex(f => f.column === filter.column);
      if (idx >= 0) {
        const updated = [...prev];
        updated[idx] = filter;
        return updated;
      }
      return [...prev, filter];
    });
  };

  const removePendingFilter = (column: string) => {
    setPendingFilters(prev => prev.filter(f => f.column !== column));
  };

  const handleApplyFilters = () => {
    setFilters(pendingFilters);
  };

  const handleClearAll = () => {
    setPendingFilters([]);
    clearFilters();
  };

  if (!dataset || categoricalColumns.length === 0) {
    return null;
  }

  const activeFilterCount = filters.length;

  // Format column name for display (convert snake_case to Title Case)
  const formatColumnName = (name: string) => {
    return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  };

  return (
    <motion.div
      ref={containerRef}
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-xl border border-border/50 bg-card/30 p-4"
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="relative">
            <Filter className="h-4 w-4 text-muted-foreground" />
            {isDirty && (
              <span className="absolute -right-1 -top-1 h-2 w-2 rounded-full bg-orange-500" title="Unapplied changes" />
            )}
          </div>
          <span className="text-sm font-medium">Filters</span>
          {activeFilterCount > 0 && (
            <span className="rounded-full bg-primary/20 px-2 py-0.5 text-xs font-medium text-primary">
              {activeFilterCount} active
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {isDirty && (
            <Button
              size="sm"
              onClick={handleApplyFilters}
              className="h-7 text-xs"
            >
              <Check className="h-3 w-3 mr-1" />
              Apply Filters
            </Button>
          )}
          {(activeFilterCount > 0 || pendingFilters.length > 0) && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClearAll}
              className="h-7 text-xs text-muted-foreground hover:text-foreground"
            >
              Clear all
            </Button>
          )}
        </div>
      </div>

      {/* Filter Chips */}
      <div className="flex flex-wrap gap-2">
        {categoricalColumns.slice(0, 6).map(column => {
          const pendingFilter = pendingFilters.find(f => f.column === column.name);
          const appliedFilter = filters.find(f => f.column === column.name);
          const isExpanded = expandedColumn === column.name;

          return (
            <div key={column.name} className="relative">
              <button
                onClick={() => handleExpandChange(isExpanded ? null : column.name)}
                className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm transition-all ${
                  appliedFilter
                    ? 'bg-primary text-primary-foreground'
                    : pendingFilter
                    ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30'
                    : 'bg-white/5 text-muted-foreground hover:bg-white/10'
                }`}
              >
                {formatColumnName(column.name)}
                {pendingFilter && (
                  <span className="text-xs opacity-75">
                    ({(pendingFilter.values?.length ?? 0)})
                  </span>
                )}
                <ChevronDown className={`h-3 w-3 transition-transform ${isExpanded ? 'rotate-180' : ''}`} />
              </button>

              {/* Dropdown */}
              <AnimatePresence>
                {isExpanded && (
                  <motion.div
                    initial={{ opacity: 0, y: -5 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -5 }}
                    className="absolute left-0 top-full z-50 mt-1 w-56 rounded-lg border border-border bg-card p-2 shadow-xl"
                  >
                    <div className="max-h-64 space-y-1 overflow-y-auto">
                      {column.sample_values.length === 0 ? (
                        <p className="px-2 py-1 text-sm text-muted-foreground">No values available</p>
                      ) : (
                        column.sample_values.map((value, idx) => {
                          const stringValue = String(value);
                          const isSelected = pendingFilter?.values?.includes(stringValue);

                          return (
                            <label
                              key={idx}
                              className="flex cursor-pointer items-center gap-2 rounded px-2 py-1.5 text-sm hover:bg-white/5"
                            >
                              <input
                                type="checkbox"
                                checked={isSelected}
                                onChange={() => {
                                  const currentValues = (pendingFilter?.values ?? []) as string[];
                                  const newValues = isSelected
                                    ? currentValues.filter(v => v !== stringValue)
                                    : [...currentValues, stringValue];

                                  if (newValues.length === 0) {
                                    removePendingFilter(column.name);
                                  } else {
                                    addPendingFilter({ column: column.name, values: newValues });
                                  }
                                }}
                                className="rounded border-border"
                              />
                              <span className="truncate">{stringValue}</span>
                            </label>
                          );
                        })
                      )}
                    </div>
                    {column.unique_count > 20 && (
                      <p className="mt-2 border-t border-border/50 pt-2 text-xs text-muted-foreground">
                        Showing 20 of {column.unique_count} values
                      </p>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          );
        })}
      </div>

      {/* Active Filter Tags */}
      {filters.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-2 border-t border-border/50 pt-3">
          {filters.map(filter => (
            <span
              key={filter.column}
              className="flex items-center gap-1 rounded-full bg-primary/10 px-2 py-1 text-xs text-primary"
            >
              {formatColumnName(filter.column)}: {(filter.values as string[])?.slice(0, 2).join(', ')}
              {((filter.values as string[])?.length ?? 0) > 2 && ` +${((filter.values as string[])?.length ?? 0) - 2}`}
              <button
                onClick={() => {
                  removePendingFilter(filter.column);
                  // Also remove from applied immediately for filter tags
                  setFilters(filters.filter(f => f.column !== filter.column));
                }}
                className="ml-1 rounded-full p-0.5 hover:bg-primary/20"
              >
                <X className="h-3 w-3" />
              </button>
            </span>
          ))}
        </div>
      )}
    </motion.div>
  );
}
