'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { AlertCircle, Check, ChevronDown, Loader2, Sparkles, X } from 'lucide-react';
import type {
  ColumnAggregation,
  ColumnFormat,
  ColumnSchemaOverride,
  DatasetColumnSchema,
  DatasetColumnProposal,
  DatasetSchemaResponse,
  ParseAs,
  SemanticType,
  VersionedUploadResponse,
} from '@/types';
import { applyDatasetSchema, getDatasetSchema } from '@/lib/api';

interface ColumnReviewProps {
  open: boolean;
  datasetId: string;
  datasetVersion?: string;
  onClose: () => void;
  onApplied: (response: VersionedUploadResponse, previousVersion?: string) => void;
}

const parseOptions: ParseAs[] = ['auto', 'text', 'number', 'date'];
const roleOptions: SemanticType[] = ['metric', 'identifier', 'temporal', 'categorical', 'unknown'];
const formatOptions: ColumnFormat[] = ['currency', 'percentage', 'number', 'date', 'identifier', 'general'];
const aggregationOptions: ColumnAggregation[] = ['sum', 'mean', 'count', 'none'];
const selectClass = 'w-full rounded-md border border-border/60 bg-background px-2 py-1.5 text-xs text-foreground outline-none focus:border-primary/60';

function isOneOf<T extends string>(value: unknown, options: readonly T[]): value is T {
  return typeof value === 'string' && options.includes(value as T);
}

function toOverride(column: DatasetColumnSchema): ColumnSchemaOverride {
  return {
    column: column.column,
    parse_as: isOneOf(column.parse_as, parseOptions) ? column.parse_as : 'auto',
    role: column.role ?? null,
    format: column.format ?? null,
    unit: column.unit,
    aggregation: column.aggregation ?? null,
  };
}

type ProposalReview = { supported: true; value: ColumnSchemaOverride } | { supported: false; reason: string };

function reviewProposal(
  column: DatasetColumnSchema,
  current: ColumnSchemaOverride,
  proposal: DatasetColumnProposal,
): ProposalReview {
  const decision = proposal.decision;
  if (proposal.status !== 'ok' || !decision) return { supported: false, reason: 'This proposal is unavailable or uncertain.' };
  if (decision.needs_clarification) return { supported: false, reason: 'This proposal needs clarification.' };
  if (!isOneOf(decision.role, roleOptions)) return { supported: false, reason: 'The proposed role is not supported.' };

  const parseAsByPolicy: Record<string, ParseAs> = {
    preserve_lexeme: 'text',
    preserve_source: 'text',
    parse_currency_decimal: 'number',
    parse_percent_to_ratio: 'number',
    parse_decimal: 'number',
    preserve_nulls_parse_numeric: 'number',
    preserve_numeric_value: 'number',
    parse_unambiguous_date: 'date',
  };
  const parseAs = decision.parsing_policy ? parseAsByPolicy[decision.parsing_policy] : undefined;
  if (!parseAs) return { supported: false, reason: 'This parsing policy needs a conversion the form cannot express.' };
  if (typeof decision.unit !== 'string') return { supported: false, reason: 'The proposed unit is missing.' };
  if (!isOneOf(decision.recommended_aggregation, aggregationOptions)) {
    return { supported: false, reason: 'This aggregation is not supported by the form.' };
  }
  if (decision.role === 'unknown') return { supported: false, reason: 'The proposed role is still unknown.' };
  if ((decision.role === 'identifier' || decision.role === 'categorical')
    && (parseAs !== 'text' || !['preserve_lexeme', 'preserve_source'].includes(decision.parsing_policy ?? '')
      || !['count', 'none'].includes(decision.recommended_aggregation))) {
    return { supported: false, reason: 'This proposal combines text columns with an unsupported parse or aggregation.' };
  }
  if (decision.role === 'temporal'
    && (parseAs !== 'date' || decision.parsing_policy !== 'parse_unambiguous_date'
      || decision.unit !== 'calendar_date' || decision.recommended_aggregation !== 'none')) {
    return { supported: false, reason: 'This date proposal needs a locale or aggregation the form cannot express.' };
  }
  if (decision.role === 'metric'
    && (parseAs !== 'number' || !['sum', 'mean'].includes(decision.recommended_aggregation))) {
    return { supported: false, reason: 'This metric proposal needs an unsupported numeric or aggregation rule.' };
  }
  if (decision.parsing_policy === 'parse_currency_decimal' && !['USD', 'EUR'].includes(decision.unit)) {
    return { supported: false, reason: 'Currency proposals require a supported USD or EUR unit.' };
  }
  if (decision.parsing_policy === 'parse_percent_to_ratio' && decision.unit !== 'ratio') {
    return { supported: false, reason: 'This percentage proposal does not use a supported ratio unit.' };
  }

  const updated: ColumnSchemaOverride = { ...current };
  updated.parse_as = parseAs;
  updated.role = decision.role;
  updated.unit = decision.unit;
  updated.aggregation = decision.recommended_aggregation;
  if (decision.parsing_policy === 'parse_currency_decimal') updated.format = 'currency';
  else if (decision.parsing_policy === 'parse_percent_to_ratio') updated.format = 'percentage';
  else if (decision.parsing_policy === 'parse_unambiguous_date') updated.format = 'date';
  else if (decision.role === 'identifier') updated.format = 'identifier';
  else if (parseAs === 'number') updated.format = 'number';
  return { supported: true, value: { ...updated, column: column.column } };
}

export function ColumnReview({ open, datasetId, datasetVersion, onClose, onApplied }: ColumnReviewProps) {
  const [schema, setSchema] = useState<DatasetSchemaResponse | null>(null);
  const [overrides, setOverrides] = useState<Record<string, ColumnSchemaOverride>>({});
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generation = useRef(0);
  const saveController = useRef<AbortController | null>(null);

  useEffect(() => {
    generation.current += 1;
    saveController.current?.abort();
    setIsSaving(false);
    if (!open || !datasetId) return;
    const controller = new AbortController();
    setIsLoading(true);
    setError(null);
    setSchema(null);
    void getDatasetSchema(datasetId, controller.signal)
      .then(result => {
        if (controller.signal.aborted) return;
        setSchema(result);
        setOverrides(Object.fromEntries(result.columns.map(column => [column.column, toOverride(column)])));
      })
      .catch(reason => {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : 'Could not load the column schema.');
      })
      .finally(() => {
        if (!controller.signal.aborted) setIsLoading(false);
      });
    return () => { generation.current += 1; controller.abort(); saveController.current?.abort(); };
  }, [open, datasetId, datasetVersion]);

  const hasChanges = useMemo(() => {
    if (!schema) return false;
    return schema.columns.some(column => {
      const initial = toOverride(column);
      return JSON.stringify(initial) !== JSON.stringify(overrides[column.column]);
    });
  }, [schema, overrides]);

  const updateColumn = (column: string, update: Partial<ColumnSchemaOverride>) => {
    setOverrides(current => ({
      ...current,
      [column]: { ...current[column], ...update, column },
    }));
  };

  const accept = (column: DatasetColumnSchema) => {
    if (!schema) return;
    const proposal = schema.proposals[column.column];
    if (!proposal) return;
    setOverrides(current => {
      const review = reviewProposal(column, current[column.column] ?? toOverride(column), proposal);
      return review.supported ? { ...current, [column.column]: review.value } : current;
    });
  };

  const apply = async () => {
    if (!schema || isSaving || !hasChanges) return;
    setIsSaving(true);
    setError(null);
    const activeGeneration = generation.current;
    const controller = new AbortController();
    saveController.current = controller;
    try {
      const columnOverrides = schema.columns.flatMap(column => {
        const initial = toOverride(column);
        const edited = overrides[column.column] ?? initial;
        const changes = Object.fromEntries(Object.entries(edited).filter(([key, value]) => key !== 'column' && value !== initial[key as keyof ColumnSchemaOverride]));
        return Object.keys(changes).length ? [{ column: column.column, ...changes }] : [];
      });
      const response = await applyDatasetSchema(datasetId, schema.version, columnOverrides, controller.signal);
      if (controller.signal.aborted || generation.current !== activeGeneration) return;
      onApplied(response, datasetVersion);
      onClose();
    } catch (reason) {
      if (!controller.signal.aborted && generation.current === activeGeneration) setError(reason instanceof Error ? reason.message : 'Could not apply the column schema.');
    } finally {
      if (generation.current === activeGeneration) setIsSaving(false);
    }
  };

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/60 p-2 backdrop-blur-sm sm:p-5" role="dialog" aria-modal="true" aria-labelledby="column-review-title">
      <section className="flex max-h-[95vh] w-full max-w-[1500px] flex-col overflow-hidden rounded-2xl border border-border/60 bg-background shadow-2xl">
        <header className="flex items-start justify-between gap-4 border-b border-border/50 px-5 py-4 sm:px-6">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-primary">Column review</p>
            <h2 id="column-review-title" className="mt-1 text-xl font-semibold">Review how columns are interpreted</h2>
            <p className="mt-1 text-sm text-muted-foreground">Edits create a new dataset version. Proposals are suggestions until you accept them into the form.</p>
          </div>
          <button type="button" onClick={onClose} disabled={isSaving} className="rounded-lg p-2 text-muted-foreground hover:bg-muted/50 hover:text-foreground disabled:opacity-50" aria-label="Close column review">
            <X className="h-4 w-4" />
          </button>
        </header>

        {error && <div role="alert" className="mx-5 mt-4 flex items-start gap-2 rounded-lg border border-destructive/40 bg-destructive/5 px-3 py-2 text-sm text-destructive sm:mx-6"><AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />{error}</div>}
        {schema && datasetVersion && schema.version !== datasetVersion && (
          <div className="mx-5 mt-4 rounded-lg border border-amber-500/30 bg-amber-500/5 px-3 py-2 text-sm text-amber-100 sm:mx-6">
            The stored dataset has advanced to version {schema.version}. Applying edits will use that current version.
          </div>
        )}

        <main className="min-h-0 flex-1 overflow-y-auto p-5 sm:p-6">
          {isLoading ? (
            <div className="flex min-h-52 items-center justify-center gap-3 text-sm text-muted-foreground"><Loader2 className="h-5 w-5 animate-spin text-primary" />Loading schema and sample rows…</div>
          ) : schema ? (
            <>
              <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                <div className="text-sm text-muted-foreground">{schema.columns.length} columns · schema version {schema.version}</div>
                <p className="inline-flex items-center gap-2 text-xs text-muted-foreground"><Sparkles className="h-3.5 w-3.5 text-primary" />Accept a proposal to copy its values into editable fields.</p>
              </div>
              <div className="overflow-x-auto rounded-xl border border-border/50">
                <table className="w-full min-w-[1000px] border-collapse text-left text-sm">
                  <thead className="sticky top-0 z-10 bg-card text-xs uppercase tracking-wide text-muted-foreground">
                    <tr>
                      <th className="px-3 py-3">Column</th>
                      <th className="px-3 py-3">Parse as</th>
                      <th className="px-3 py-3">Role</th>
                      <th className="px-3 py-3">Display format</th>
                      <th className="px-3 py-3">Unit</th>
                      <th className="px-3 py-3">Aggregation</th>
                      <th className="px-3 py-3">Proposal</th>
                    </tr>
                  </thead>
                  <tbody>
                    {schema.columns.map(column => {
                      const value = overrides[column.column] ?? toOverride(column);
                      const proposal = schema.proposals[column.column];
                      const proposalReview = proposal ? reviewProposal(column, value, proposal) : null;
                      return (
                        <tr key={column.column} className="border-t border-border/40 align-top">
                          <td className="max-w-64 px-3 py-3">
                            <p className="truncate font-medium" title={column.column}>{column.column}</p>
                            {column.original_name !== column.column && <p className="mt-0.5 truncate text-xs text-muted-foreground" title={column.original_name}>Source: {column.original_name}</p>}
                            <p className="mt-1 text-[11px] text-muted-foreground/70">{column.provenance}</p>
                            <button type="button" disabled={isSaving} className="mt-1 text-xs text-primary" onClick={() => updateColumn(column.column, { parse_as: 'auto', role: null, format: null, unit: null, aggregation: null })}>Reset to automatic</button>
                          </td>
                          <td className="w-32 px-2 py-3">
                            <select aria-label={`Parse ${column.column} as`} className={selectClass} value={value.parse_as} disabled={isSaving} onChange={event => updateColumn(column.column, { parse_as: event.target.value as ParseAs })}>
                              {parseOptions.map(option => <option key={option} value={option}>{option === 'auto' ? 'Automatic' : option[0].toUpperCase() + option.slice(1)}</option>)}
                            </select>
                          </td>
                          <td className="w-36 px-2 py-3">
                            <select aria-label={`Role for ${column.column}`} className={selectClass} value={value.role ?? ''} disabled={isSaving} onChange={event => updateColumn(column.column, { role: event.target.value ? event.target.value as SemanticType : null })}>
                              <option value="">Automatic</option>
                              {roleOptions.map(option => <option key={option} value={option}>{option[0].toUpperCase() + option.slice(1)}</option>)}
                            </select>
                          </td>
                          <td className="w-36 px-2 py-3">
                            <select aria-label={`Format for ${column.column}`} className={selectClass} value={value.format ?? ''} disabled={isSaving} onChange={event => updateColumn(column.column, { format: event.target.value ? event.target.value as ColumnFormat : null })}>
                              <option value="">Automatic</option>
                              {formatOptions.map(option => <option key={option} value={option}>{option[0].toUpperCase() + option.slice(1)}</option>)}
                            </select>
                            {value.format === 'percentage' && <p className="mt-1 text-[10px] leading-tight text-muted-foreground">12% parses as 0.12; plain 0.12 displays as 12%.</p>}
                          </td>
                          <td className="w-36 px-2 py-3">
                            <input aria-label={`Unit for ${column.column}`} className={selectClass} value={value.unit ?? ''} disabled={isSaving} onChange={event => updateColumn(column.column, { unit: event.target.value || null })} placeholder="e.g. USD" />
                          </td>
                          <td className="w-32 px-2 py-3">
                            <select aria-label={`Aggregation for ${column.column}`} className={selectClass} value={value.aggregation ?? ''} disabled={isSaving} onChange={event => updateColumn(column.column, { aggregation: event.target.value ? event.target.value as ColumnAggregation : null })}>
                              <option value="">Automatic</option>
                              {aggregationOptions.map(option => <option key={option} value={option}>{option === 'none' ? 'None' : option[0].toUpperCase() + option.slice(1)}</option>)}
                            </select>
                          </td>
                          <td className="w-36 px-2 py-3">
                            {proposal ? (
                              <button type="button" onClick={() => accept(column)} disabled={isSaving || !proposalReview?.supported} title={proposalReview && !proposalReview.supported ? proposalReview.reason : undefined} className="inline-flex items-center gap-1.5 rounded-md border border-primary/40 px-2 py-1.5 text-xs font-medium text-primary hover:bg-primary/10 disabled:cursor-not-allowed disabled:opacity-50">
                                <Check className="h-3 w-3" />Accept proposal
                              </button>
                            ) : <span className="text-xs text-muted-foreground">No proposal</span>}
                            {proposalReview && !proposalReview.supported && <p className="mt-1 text-[11px] leading-tight text-amber-200">{proposalReview.reason}</p>}
                            {proposal && <details className="mt-1 text-xs text-muted-foreground">
                              <summary className="flex cursor-pointer list-none items-center gap-1"><ChevronDown className="h-3 w-3" />Details</summary>
                              <pre className="mt-1 max-w-44 whitespace-pre-wrap break-words">{JSON.stringify(proposal, null, 2)}</pre>
                            </details>}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              <section className="mt-5 rounded-xl border border-border/50 bg-card/30 p-4">
                <h3 className="text-sm font-medium">Current parsed sample</h3>
                <p className="mt-1 text-xs text-muted-foreground">A few rows from the active dataset version.</p>
                <div className="mt-3 max-h-56 overflow-auto rounded-lg border border-border/40">
                  <table className="w-full min-w-max border-collapse text-left text-xs">
                    <thead className="sticky top-0 bg-card text-muted-foreground"><tr>{schema.preview.columns.map(column => <th key={column} className="px-3 py-2 font-medium">{column}</th>)}</tr></thead>
                    <tbody>
                      {schema.preview.rows.slice(0, 8).map((row, rowIndex) => (
                        <tr key={rowIndex} className="border-t border-border/30 even:bg-muted/10">
                          {schema.preview.columns.map(column => <td key={column} className="max-w-56 truncate px-3 py-2" title={String(row[column] ?? '')}>{row[column] === null || row[column] === undefined || row[column] === '' ? '(blank)' : String(row[column])}</td>)}
                        </tr>
                      ))}
                      {schema.preview.rows.length === 0 && <tr><td colSpan={Math.max(schema.preview.columns.length, 1)} className="px-3 py-4 text-center text-muted-foreground">No preview rows returned.</td></tr>}
                    </tbody>
                  </table>
                </div>
              </section>
            </>
          ) : null}
        </main>

        <footer className="flex flex-wrap items-center justify-between gap-3 border-t border-border/50 px-5 py-4 sm:px-6">
          <p className="text-xs text-muted-foreground">Only your edits are applied. The original source is retained.</p>
          <div className="flex items-center gap-2">
            <button type="button" onClick={onClose} disabled={isSaving} className="rounded-lg px-4 py-2 text-sm text-muted-foreground hover:bg-muted/50 disabled:opacity-50">Close</button>
            <button type="button" onClick={() => void apply()} disabled={!schema || isLoading || isSaving || !hasChanges} className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40">
              {isSaving ? <Loader2 className="h-4 w-4 animate-spin" /> : null} Apply schema
            </button>
          </div>
        </footer>
      </section>
    </div>
  );
}
