'use client';

import React, { useState } from 'react';
import { AlertCircle, AlertTriangle, CheckCircle2, Loader2, RefreshCw, X } from 'lucide-react';
import type { ImportPreviewResponse, ImportSettings } from '@/types';

interface ImportPreviewProps {
  preview: ImportPreviewResponse;
  busy?: boolean;
  actionError?: string | null;
  onRecheck: (settings: ImportSettings) => void;
  onConfirm: (settings: ImportSettings) => void;
  onCancel: () => void;
}

const fieldClass = 'mt-1 w-full rounded-lg border border-border/60 bg-background px-3 py-2 text-sm text-foreground outline-none focus:border-primary/60';

function showValue(value: unknown): string {
  if (value === null || value === undefined || value === '') return '(blank)';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

export function ImportPreview({
  preview,
  busy = false,
  actionError,
  onRecheck,
  onConfirm,
  onCancel,
}: ImportPreviewProps) {
  const [settings, setSettings] = useState<ImportSettings>(preview.settings);
  const [nullTokens, setNullTokens] = useState(preview.settings.null_values.filter(token => token !== '').join('\n'));
  const [includesBlank, setIncludesBlank] = useState(preview.settings.null_values.includes(''));
  const isDirty = JSON.stringify(settings) !== JSON.stringify(preview.settings);
  const rawRows = preview.raw_rows.slice(0, 12);
  const parsedRows = preview.parsed_rows.slice(0, 12);
  const shownRowCount = Math.min(rawRows.length, parsedRows.length);

  const updateSetting = <K extends keyof ImportSettings>(key: K, value: ImportSettings[K]) => {
    setSettings(current => ({ ...current, [key]: value }));
  };

  return (
    <section className="w-full rounded-2xl border border-border/60 bg-card/70 p-5 shadow-xl sm:p-6" aria-labelledby="import-preview-title">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-primary">Import review</p>
          <h2 id="import-preview-title" className="mt-1 text-xl font-semibold">Check how this file will be read</h2>
          <p className="mt-1 text-sm text-muted-foreground">{preview.filename}</p>
        </div>
        <button
          type="button"
          onClick={onCancel}
          disabled={busy}
          className="inline-flex items-center gap-2 rounded-lg px-3 py-2 text-sm text-muted-foreground hover:bg-muted/50 hover:text-foreground disabled:opacity-50"
        >
          <X className="h-4 w-4" /> Cancel import
        </button>
      </div>

      <div className={`mt-5 flex items-start gap-3 rounded-xl border p-3 text-sm ${preview.can_confirm ? 'border-emerald-500/30 bg-emerald-500/5 text-emerald-200' : 'border-destructive/40 bg-destructive/5 text-destructive'}`}>
        {preview.can_confirm ? <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" /> : <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />}
        <div>
          <p className="font-medium">
            {preview.can_confirm ? 'The sample can be read with these settings' : 'The sample needs a parsing change before confirmation'}
          </p>
          <p className="mt-0.5 text-xs opacity-80">
            {preview.sample_complete ? `Showing all ${preview.sample_row_count.toLocaleString()} preview rows.` : `Showing ${preview.sample_row_count.toLocaleString()} sample rows.`} The entire file is validated after you confirm.
          </p>
        </div>
      </div>

      {preview.warnings.length > 0 && (
        <div className="mt-3 space-y-2">
          {preview.warnings.map((warning, index) => (
            <div key={`${index}-${warning}`} className="flex items-start gap-2 rounded-lg border border-amber-500/25 bg-amber-500/5 px-3 py-2 text-sm text-amber-100">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
              <span>{warning}</span>
            </div>
          ))}
        </div>
      )}

      {actionError && <p role="alert" className="mt-3 rounded-lg border border-destructive/40 bg-destructive/5 px-3 py-2 text-sm text-destructive">{actionError}</p>}

      <div className="mt-5 grid gap-5 xl:grid-cols-[minmax(240px,0.75fr)_minmax(0,1.75fr)]">
        <div className="rounded-xl border border-border/50 bg-background/40 p-4">
          <h3 className="font-medium">Parsing settings</h3>
          <p className="mt-1 text-xs text-muted-foreground">Adjust a setting, then recheck the saved upload sample.</p>
          <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-1">
            <label className="text-xs font-medium text-muted-foreground">
              Delimiter
              <select className={fieldClass} value={settings.delimiter} disabled={busy} onChange={e => updateSetting('delimiter', e.target.value as ImportSettings['delimiter'])}>
                <option value=",">Comma</option>
                <option value=";">Semicolon</option>
                <option value="\t">Tab</option>
                <option value="|">Pipe</option>
              </select>
            </label>
            <label className="text-xs font-medium text-muted-foreground">
              Encoding
              <select className={fieldClass} value={settings.encoding} disabled={busy} onChange={e => updateSetting('encoding', e.target.value as ImportSettings['encoding'])}>
                <option value="utf-8-sig">UTF-8 with BOM</option>
                <option value="utf-8">UTF-8</option>
                <option value="utf-16">UTF-16</option>
                <option value="cp1252">Windows-1252</option>
              </select>
            </label>
            <label className="text-xs font-medium text-muted-foreground">
              Decimal separator
              <select className={fieldClass} value={settings.decimal_separator} disabled={busy} onChange={e => updateSetting('decimal_separator', e.target.value as ImportSettings['decimal_separator'])}>
                <option value="auto">Detect automatically</option>
                <option value=".">Period (1.25)</option>
                <option value=",">Comma (1,25)</option>
              </select>
            </label>
            <label className="text-xs font-medium text-muted-foreground">
              Grouping separator
              <select className={fieldClass} value={settings.grouping_separator ?? 'none'} disabled={busy} onChange={e => updateSetting('grouping_separator', e.target.value === 'none' ? null : e.target.value as ImportSettings['grouping_separator'])}>
                <option value="none">None</option>
                <option value=",">Comma (1,000)</option>
                <option value=".">Period (1.000)</option>
                <option value=" ">Space (1 000)</option>
              </select>
            </label>
            <label className="text-xs font-medium text-muted-foreground">
              Date order
              <select className={fieldClass} value={settings.date_order} disabled={busy} onChange={e => updateSetting('date_order', e.target.value as ImportSettings['date_order'])}>
                <option value="auto">Detect automatically</option>
                <option value="ymd">Year, month, day</option>
                <option value="dmy">Day, month, year</option>
                <option value="mdy">Month, day, year</option>
              </select>
            </label>
          </div>
          <label className="mt-3 block text-xs font-medium text-muted-foreground">
            Other missing-value tokens
            <textarea
              className={`${fieldClass} min-h-20 resize-y`}
              value={nullTokens}
              disabled={busy}
              onChange={e => {
                const value = e.target.value;
                setNullTokens(value);
                updateSetting('null_values', [
                  ...(includesBlank ? [''] : []),
                  ...value.split('\n').filter(token => token !== ''),
                ]);
              }}
              placeholder={'NA\nN/A\nnull'}
            />
            <span className="mt-1 block font-normal">Enter one token per line. Tokens are matched exactly, including spaces.</span>
          </label>
          <label className="mt-2 flex items-center gap-2 text-xs text-muted-foreground">
            <input
              type="checkbox"
              checked={includesBlank}
              disabled={busy}
              onChange={event => {
                const checked = event.target.checked;
                setIncludesBlank(checked);
                updateSetting('null_values', [
                  ...(checked ? [''] : []),
                  ...nullTokens.split('\n').filter(token => token !== ''),
                ]);
              }}
              className="h-4 w-4 accent-primary"
            />
            Treat blank cells as missing
          </label>
          {isDirty && <p className="mt-3 text-xs text-amber-200">Settings changed. Recheck the sample before confirming.</p>}
        </div>

        <div className="min-w-0 rounded-xl border border-border/50 bg-background/40 p-4">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <h3 className="font-medium">Original and parsed sample</h3>
              <p className="mt-1 text-xs text-muted-foreground">First {shownRowCount} rows shown. Raw headings preserve the source file&apos;s names.</p>
            </div>
            <button
              type="button"
              onClick={() => onRecheck(settings)}
              disabled={busy || !isDirty}
              className="inline-flex items-center gap-2 rounded-lg border border-border/60 px-3 py-2 text-sm font-medium hover:border-primary/50 hover:bg-primary/5 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
              Recheck sample
            </button>
          </div>

          <div className="mt-4 grid min-w-0 gap-4 2xl:grid-cols-2">
            <SampleTable
              title="Original values"
              columns={preview.columns.map(column => column.original_name)}
              rowCount={shownRowCount}
              renderCell={(rowIndex, columnIndex) => showValue(rawRows[rowIndex]?.[columnIndex])}
            />
            <SampleTable
              title="Parsed values"
              columns={preview.columns.map(column => column.name)}
              rowCount={shownRowCount}
              renderCell={(rowIndex, columnIndex) => {
                const row = parsedRows[rowIndex];
                const key = preview.columns[columnIndex]?.name;
                return showValue(key ? row?.[key] : undefined);
              }}
            />
          </div>
        </div>
      </div>

      <div className="mt-5 flex flex-wrap items-center justify-end gap-3 border-t border-border/50 pt-4">
        <button type="button" onClick={onCancel} disabled={busy} className="rounded-lg px-4 py-2 text-sm text-muted-foreground hover:bg-muted/40 disabled:opacity-50">Cancel</button>
        <button
          type="button"
          onClick={() => onConfirm(settings)}
          disabled={busy || isDirty || !preview.can_confirm}
          className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
          title={!preview.can_confirm ? 'Resolve the import warnings before confirming' : isDirty ? 'Recheck the edited settings first' : undefined}
        >
          {busy && !isDirty ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
          Confirm import
        </button>
      </div>
    </section>
  );
}

function SampleTable({
  title,
  columns,
  rowCount,
  renderCell,
}: {
  title: string;
  columns: string[];
  rowCount: number;
  renderCell: (rowIndex: number, columnIndex: number) => string;
}) {
  return (
    <div className="min-w-0 overflow-hidden rounded-lg border border-border/50">
      <p className="border-b border-border/50 bg-muted/20 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">{title}</p>
      <div className="max-h-72 overflow-auto">
        <table className="w-full min-w-max border-collapse text-left text-xs">
          <thead className="sticky top-0 bg-card text-muted-foreground">
            <tr>
              <th className="border-b border-border/50 px-3 py-2 font-medium">#</th>
              {columns.map((column, index) => <th key={`${index}-${column}`} className="max-w-48 border-b border-border/50 px-3 py-2 font-medium">{column}</th>)}
            </tr>
          </thead>
          <tbody>
            {Array.from({ length: rowCount }, (_, rowIndex) => (
              <tr key={rowIndex} className="even:bg-muted/10">
                <td className="border-b border-border/30 px-3 py-2 text-muted-foreground">{rowIndex + 1}</td>
                {columns.map((column, columnIndex) => (
                  <td key={`${rowIndex}-${columnIndex}`} className="max-w-48 truncate border-b border-border/30 px-3 py-2" title={renderCell(rowIndex, columnIndex)}>
                    {renderCell(rowIndex, columnIndex)}
                  </td>
                ))}
              </tr>
            ))}
            {rowCount === 0 && <tr><td colSpan={Math.max(columns.length + 1, 1)} className="px-3 py-5 text-center text-muted-foreground">No sample rows returned.</td></tr>}
          </tbody>
        </table>
      </div>
    </div>
  );
}
