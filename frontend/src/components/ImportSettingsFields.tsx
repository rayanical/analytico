'use client';

import React, { useState } from 'react';
import type { ImportSettings } from '@/types';

const fieldClass = 'mt-1 w-full rounded-lg border border-border/60 bg-background px-3 py-2 text-sm text-foreground outline-none focus:border-primary/60';

export function ImportSettingsFields({ settings, onChange, busy = false }: {
  settings: ImportSettings;
  onChange: (settings: ImportSettings) => void;
  busy?: boolean;
}) {
  const [nullTokens, setNullTokens] = useState(settings.null_values.filter(token => token !== '').join('\n'));
  const [includesBlank, setIncludesBlank] = useState(settings.null_values.includes(''));
  const updateSetting = <K extends keyof ImportSettings>(key: K, value: ImportSettings[K]) => onChange({ ...settings, [key]: value });
  return <>
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

  </>;
}
