export interface ColumnLabelSource {
  name?: string | null;
  column?: string | null;
  original_name?: string | null;
  display_name?: string | null;
}

/** The original source header (or stable key) remains available as a tooltip. */
export function getColumnSourceName(column: ColumnLabelSource): string {
  return column.original_name?.trim() || column.name?.trim() || column.column?.trim() || '';
}

/** Prefer an AI or user label and otherwise preserve the original source name. */
export function getColumnDisplayName(column: ColumnLabelSource): string {
  const displayName = column.display_name?.trim();
  if (displayName) return displayName;
  return getColumnSourceName(column);
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
