export interface ColumnLabelSource {
  name?: string | null;
  column?: string | null;
  original_name?: string | null;
  display_name?: string | null;
}

/** Preserve the existing snake_case-to-title-case fallback for unlabeled columns. */
export function humanizeColumnName(name: string): string {
  return name
    .replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2')
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, character => character.toUpperCase());
}

/** The original source header (or stable key) remains available as a tooltip. */
export function getColumnSourceName(column: ColumnLabelSource): string {
  return column.original_name?.trim() || column.name?.trim() || column.column?.trim() || '';
}

/** Prefer an explicit display label and otherwise use the existing humanization. */
export function getColumnDisplayName(column: ColumnLabelSource): string {
  const displayName = column.display_name?.trim();
  if (displayName) return displayName;
  // Humanize the normalized stable key for a familiar fallback. Keep the exact
  // source header available separately as a tooltip through getColumnSourceName.
  const stableName = column.name?.trim() || column.column?.trim() || column.original_name?.trim() || '';
  return humanizeColumnName(stableName);
}

/** Merge server enrichment labels by stable column key without changing that key. */
export function mergeColumnLabels<T extends { name: string; display_name?: string | null }>(
  columns: T[],
  columnLabels?: Record<string, string>,
): T[] {
  if (!columnLabels || Object.keys(columnLabels).length === 0) return columns;

  let changed = false;
  const merged = columns.map(column => {
    const displayName = columnLabels[column.name]?.trim();
    if (!displayName || displayName === column.display_name) return column;
    changed = true;
    return { ...column, display_name: displayName };
  });
  return changed ? merged : columns;
}
