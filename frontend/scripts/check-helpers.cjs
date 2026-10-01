/* eslint-disable @typescript-eslint/no-require-imports */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const Module = require('node:module');
const path = require('node:path');
const ts = require('typescript');

function loadTypeScript(relativePath) {
  const filename = path.join(__dirname, '..', 'src', relativePath);
  const source = fs.readFileSync(filename, 'utf8');
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2020,
    },
  });
  const loaded = new Module(filename, module);
  loaded.filename = filename;
  loaded.paths = Module._nodeModulePaths(path.dirname(filename));
  loaded._compile(outputText, filename);
  return loaded.exports;
}

const { formatValue } = loadTypeScript('lib/formatValue.ts');
const { getNextPeriodStart, mergeFilters, preserveQueryProvenance } = loadTypeScript('lib/queryFilters.ts');
const { isDatasetState } = loadTypeScript('lib/storageValidation.ts');
const { normalizeChartResponse } = loadTypeScript('lib/api.ts');

assert.equal(formatValue(0.2, 'percentage'), '20%');
assert.equal(formatValue(1200, 'currency', { aggregation: 'count' }), '1,200');
assert.equal(formatValue(null), '—');
assert.equal(formatValue('001', 'identifier'), '001');
process.env.TZ = 'America/New_York';
assert.equal(getNextPeriodStart('2026-03-01T00:00:00', 'month'), '2026-04-01T00:00:00.000Z');
assert.equal(getNextPeriodStart(null, 'month'), null);

const lowerBound = { column: 'event_date', operator: 'gte', value: '2026-01-01' };
const upperBound = { column: 'event_date', operator: 'lt', value: '2026-02-01' };
const equality = { column: 'revenue', operator: 'eq', value: 150 };
const merged = mergeFilters(
  [lowerBound, upperBound, equality],
  [lowerBound, { column: 'revenue', operator: 'gt', value: 100 }],
);
assert.equal(merged.length, 4);
assert.deepEqual(merged.slice(0, 3), [lowerBound, upperBound, equality]);
assert.deepEqual(merged[3], { column: 'revenue', operator: 'gt', value: 100 });
assert.deepEqual(mergeFilters(null, [lowerBound]), [lowerBound]);

const previousChart = { x_axis_key: 'event_date', source_x_axis_key: 'event_date', time_bucket: null, llm_filters: null };
const newlyBucketed = preserveQueryProvenance({ time_bucket: 'month' }, previousChart, []);
assert.equal(newlyBucketed.time_bucket, 'month', 'use the bucket returned for the refreshed result');
const unbucketed = preserveQueryProvenance({ time_bucket: null }, { ...previousChart, time_bucket: 'month' }, []);
assert.equal(unbucketed.time_bucket, null, 'keep the server result when a refresh no longer buckets dates');

const normalizedChart = normalizeChartResponse({
  data: [],
  x_axis_key: '',
  y_axis_keys: [],
  chart_type: 'empty',
  title: 'Clarification Needed',
  row_count: 0,
  aggregation: null,
  x_axis_label: null,
  y_axis_label: null,
  analysis: null,
  warnings: null,
  applied_filters: null,
  filters: [{ column: 'region', operator: null, value: null, values: ['West', null], min_val: null, max_val: null }],
  llm_filters: null,
  source_x_axis_key: null,
  time_bucket: null,
  others_label: null,
  answer: null,
});
assert.equal(normalizedChart.aggregation, undefined);
assert.equal(normalizedChart.analysis, undefined);
assert.deepEqual(normalizedChart.filters, [{ column: 'region', values: ['West', null] }]);
assert.equal(normalizedChart.time_bucket, null);

const savedGapminder = {
  datasetId: 'gapminder',
  filename: 'gapminder.csv',
  rowCount: 1704,
  columns: [
    { name: 'country', dtype: 'object', is_numeric: false, is_datetime: false, semantic_type: 'categorical', format: 'general', unique_count: 142, sample_values: ['Afghanistan'] },
    { name: 'year', dtype: 'int64', is_numeric: true, is_datetime: false, semantic_type: 'temporal', format: 'number', unique_count: 12, sample_values: [1952] },
    { name: 'life_expectancy', dtype: 'float64', is_numeric: true, is_datetime: false, semantic_type: 'metric', format: 'number', unique_count: 1626, sample_values: [28.8] },
  ],
  columnFormats: { country: 'general', year: 'number', life_expectancy: 'number' },
  dataHealth: { missing_values: {}, cleaning_actions: [], quality_score: 100 },
  profile: {
    top_metrics: [{ name: 'life_expectancy', total: 79002, average: 58.4, aggregation: 'mean', min: 23.6, max: 82.6 }],
    time_range: { column: 'year', start: '1952', end: '2007' },
    row_count: 1704,
    column_count: 6,
  },
  defaultChart: { x_axis_key: 'year', y_axis_keys: ['life_expectancy'], chart_type: 'line', aggregation: 'mean', title: 'Mean life expectancy by year', analysis: 'Example summary.' },
  suggestions: [],
  summary: null,
};
assert.equal(isDatasetState(savedGapminder), true, 'real saved dataset shape with general format and null summary should hydrate');
assert.equal(isDatasetState({ ...savedGapminder, columns: [{ ...savedGapminder.columns[0], format: 'made-up' }] }), false);
assert.equal(isDatasetState({ ...savedGapminder, defaultChart: { ...savedGapminder.defaultChart, chart_type: 'empty' } }), false);

console.log('Frontend helper checks passed.');
