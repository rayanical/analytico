'use client';

import React, { useMemo, useState } from 'react';
import { Search, X } from 'lucide-react';
import { DataHealth } from '@/types';

interface CleaningReportDrawerProps {
  open: boolean;
  onClose: () => void;
  dataHealth: DataHealth;
  rowCount: number;
  columnCount: number;
}

export function CleaningReportDrawer({
  open,
  onClose,
  dataHealth,
  rowCount,
  columnCount,
}: CleaningReportDrawerProps) {
  const [query, setQuery] = useState('');

  const totalCells = rowCount * columnCount;
  const missingCells = Object.values(dataHealth.missing_values).reduce((sum, count) => sum + count, 0);

  const missingByColumn = useMemo(
    () =>
      Object.entries(dataHealth.missing_values)
        .sort((a, b) => b[1] - a[1])
        .filter(([column]) => column.toLowerCase().includes(query.toLowerCase())),
    [dataHealth.missing_values, query]
  );

  const filteredActions = useMemo(
    () => dataHealth.cleaning_actions.filter((action) => action.toLowerCase().includes(query.toLowerCase())),
    [dataHealth.cleaning_actions, query]
  );

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50">
      <button
        type="button"
        className="absolute inset-0 bg-black/50"
        aria-label="Close cleaning report"
        onClick={onClose}
      />
      <aside className="absolute right-0 top-0 h-full w-full max-w-xl border-l border-border/60 bg-background p-5 shadow-2xl">
        <div className="mb-4 flex items-start justify-between">
          <div>
            <h3 className="text-lg font-semibold">Cleaning Report</h3>
            <p className="text-sm text-muted-foreground">What changed during dataset ingestion</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md p-1.5 text-muted-foreground hover:bg-muted/40 hover:text-foreground"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="mb-4 grid grid-cols-3 gap-2">
          <div className="rounded-lg border border-border/50 bg-card/40 p-3">
            <p className="text-xs text-muted-foreground">Actions</p>
            <p className="text-lg font-semibold">{dataHealth.cleaning_actions.length}</p>
          </div>
          <div className="rounded-lg border border-border/50 bg-card/40 p-3">
            <p className="text-xs text-muted-foreground">Affected Columns</p>
            <p className="text-lg font-semibold">{Object.keys(dataHealth.missing_values).length}</p>
          </div>
          <div className="rounded-lg border border-border/50 bg-card/40 p-3">
            <p className="text-xs text-muted-foreground">Missing Cells</p>
            <p className="text-lg font-semibold">
              {missingCells.toLocaleString()} / {totalCells.toLocaleString()}
            </p>
          </div>
        </div>

        <div className="relative mb-4">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search actions or columns..."
            className="w-full rounded-lg border border-border/60 bg-card/40 py-2 pl-9 pr-3 text-sm outline-none focus:border-primary/50"
          />
        </div>

        <div className="grid h-[calc(100%-210px)] grid-cols-1 gap-4 overflow-hidden md:grid-cols-2">
          <section className="min-h-0 rounded-lg border border-border/60 bg-card/30 p-3">
            <h4 className="mb-2 text-sm font-medium">Cleaning Actions</h4>
            <div className="h-full space-y-2 overflow-y-auto pr-1">
              {filteredActions.length === 0 ? (
                <p className="text-sm text-muted-foreground">No cleaning actions matched.</p>
              ) : (
                filteredActions.map((action, idx) => (
                  <p key={`${idx}-${action}`} className="rounded-md border border-border/40 bg-background/40 p-2 text-sm">
                    {action}
                  </p>
                ))
              )}
            </div>
          </section>

          <section className="min-h-0 rounded-lg border border-border/60 bg-card/30 p-3">
            <h4 className="mb-2 text-sm font-medium">Missing Values by Column</h4>
            <div className="h-full space-y-2 overflow-y-auto pr-1">
              {missingByColumn.length === 0 ? (
                <p className="text-sm text-muted-foreground">No missing values matched.</p>
              ) : (
                missingByColumn.map(([column, count]) => (
                  <div key={column} className="flex items-center justify-between rounded-md border border-border/40 bg-background/40 p-2 text-sm">
                    <span className="truncate pr-2">{column}</span>
                    <span className="font-medium">{count.toLocaleString()}</span>
                  </div>
                ))
              )}
            </div>
          </section>
        </div>
      </aside>
    </div>
  );
}
