import type { AggregationType, ColumnFormat } from '@/types';

interface FormatOptions {
  compact?: boolean;
  aggregation?: AggregationType;
}

export function formatValue(
  value: unknown,
  format: ColumnFormat = 'number',
  { compact = false, aggregation }: FormatOptions = {},
): string {
  if (value === null || value === undefined) return '—';
  if (format === 'identifier' || format === 'date') return String(value);

  const number = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(number)) return String(value);

  const effectiveFormat = aggregation === 'count' ? 'number' : format;
  const scaledNumber = effectiveFormat === 'percentage' ? number * 100 : number;
  const suffix = effectiveFormat === 'percentage' ? '%' : '';

  if (compact) {
    const absolute = Math.abs(scaledNumber);
    if (absolute >= 1_000_000_000) return `${(scaledNumber / 1_000_000_000).toFixed(1)}B${suffix}`;
    if (absolute >= 1_000_000) return `${(scaledNumber / 1_000_000).toFixed(1)}M${suffix}`;
    if (absolute >= 1_000) return `${(scaledNumber / 1_000).toFixed(1)}K${suffix}`;
  }

  const maximumFractionDigits = effectiveFormat === 'percentage' ? 1 : effectiveFormat === 'currency' ? 2 : 2;
  return `${new Intl.NumberFormat('en-US', { maximumFractionDigits }).format(scaledNumber)}${suffix}`;
}
