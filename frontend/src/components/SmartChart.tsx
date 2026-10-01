'use client';

import React, { memo, useEffect, useMemo, useRef, useState } from 'react';
import {
  ResponsiveContainer, BarChart, Bar, LineChart, Line, AreaChart, Area,
  PieChart, Pie, Cell, ComposedChart, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
} from 'recharts';
import { Loader2 } from 'lucide-react';
import { useData } from '@/context/DataContext';
import { ChartResponse, ColumnFormat } from '@/types';
import { aggregateData, drillDown } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';
import { formatValue } from '@/lib/formatValue';
import { getNextPeriodStart, mergeFilters, preserveQueryProvenance } from '@/lib/queryFilters';

const COLORS = [
  'hsl(252, 87%, 64%)', 'hsl(173, 80%, 40%)', 'hsl(43, 96%, 56%)',
  'hsl(346, 77%, 59%)', 'hsl(199, 89%, 48%)', 'hsl(280, 65%, 60%)',
  'hsl(150, 60%, 45%)', 'hsl(30, 90%, 55%)',
];
const EMPTY_RECORDS: ChartResponse['data'] = [];

function getDrillDownRecord(value: unknown): Record<string, unknown> | null {
  if (typeof value !== 'object' || value === null) return null;
  const record = value as Record<string, unknown>;
  const payload = record.payload;
  return typeof payload === 'object' && payload !== null && !Array.isArray(payload)
    ? payload as Record<string, unknown>
    : record;
}

function isFilterValue(value: unknown): value is string | number | boolean | null {
  return value === null
    || typeof value === 'string'
    || typeof value === 'boolean'
    || (typeof value === 'number' && Number.isFinite(value));
}

interface SmartChartProps {
  chartData?: ChartResponse;
  showAnalyzeButton?: boolean;
  compact?: boolean;
}

// ============================================================================
// CustomTooltip - Extracted as memoized component for performance
// ============================================================================

interface TooltipPayloadEntry {
  name: string;
  value: number;
  color: string;
  dataKey?: string | number;
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: TooltipPayloadEntry[];
  label?: string;
  formats: Record<string, ColumnFormat>;
  aggregation?: ChartResponse['aggregation'];
  primaryFormat: ColumnFormat;
}

const CustomTooltip = memo(function CustomTooltip({ 
  active, 
  payload, 
  label, 
  formats,
  aggregation,
  primaryFormat,
}: CustomTooltipProps) {
  if (!active || !payload?.length) return null;
  
  return (
    <div className="rounded-lg border border-border bg-card p-3 shadow-xl">
      <p className="mb-2 font-medium">{label}</p>
      {payload.map((entry, index) => (
        <p key={index} className="flex items-center gap-2 text-sm">
          <span 
            className="h-3 w-3 rounded-full" 
            style={{ backgroundColor: entry.color }} 
          />
          <span className="text-muted-foreground">{entry.name}:</span>
          <span className="font-medium">
            {formatValue(entry.value, (formats[String(entry.dataKey ?? '')] || primaryFormat) as ColumnFormat, { aggregation })}
          </span>
        </p>
      ))}
    </div>
  );
});

// ============================================================================
// SmartChart - Main chart component using composed Recharts architecture
// ============================================================================

export function SmartChart({ chartData, showAnalyzeButton = true, compact = false }: SmartChartProps) {
  const { currentChart, dataset, filters, setCurrentChart, setDrillDownData, setIsDrillDownOpen, limit, groupOthers, sortBy, beginQuery, isCurrentQuery, finishQuery, isCurrentDataset } = useData();
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const data = chartData || currentChart;
  const analysisControllerRef = useRef<AbortController | null>(null);
  const drillDownControllerRef = useRef<AbortController | null>(null);
  const drillDownRequestRef = useRef(0);
  const records = data?.data ?? EMPTY_RECORDS;
  const x_axis_key = data?.x_axis_key ?? '';
  const y_axis_keys = data?.y_axis_keys ?? [];
  const chart_type = data?.chart_type ?? 'empty';
  const y_axis_label = data?.y_axis_label;
  const answer = data?.answer;
  const analysis = data?.analysis;
  const aggregation = data?.aggregation;
  const llm_filters = data?.llm_filters;
  const formats = dataset?.columnFormats ?? {};
  const primaryFormat = (aggregation === 'count' ? 'number' : formats[y_axis_keys[0]] || 'number') as ColumnFormat;
  const tickFormatter = useMemo(() => (value: number) =>
    formatValue(value, primaryFormat, { compact: true, aggregation }), [primaryFormat, aggregation]);
  const axisStyle = useMemo(() => ({ fontSize: 11, fill: '#a1a1aa' }), []);
  const commonProps = useMemo(
    () => ({ data: records, margin: { top: 20, right: 30, left: 70, bottom: 20 } }),
    [records]
  );

  useEffect(() => {
    analysisControllerRef.current?.abort();
    drillDownControllerRef.current?.abort();
    drillDownRequestRef.current += 1;
    return () => {
      analysisControllerRef.current?.abort();
      drillDownControllerRef.current?.abort();
    };
  }, [dataset?.datasetId]);

  if (!data || (!records.length && !answer && !analysis)) return null;

  // Text-only answers include clarification responses that have no chart data.
  if (chart_type === 'empty' || (!records.length && (answer || analysis))) {
    return (
      <div className="flex h-full min-h-[400px] flex-col overflow-y-auto rounded-lg p-6">
         <div className="mb-6 rounded-lg bg-primary/10 p-4">
           <h3 className="mb-2 text-lg font-semibold text-primary">{chart_type === 'empty' ? 'Clarification needed' : answer ? 'Insight' : 'Analysis'}</h3>
           {answer && <div className="whitespace-pre-wrap font-mono text-sm leading-relaxed">{answer}</div>}
         </div>
         {analysis && (
           <div className="rounded-lg border border-border/50 bg-card/30 p-4">
             <h4 className="mb-2 text-sm font-medium text-muted-foreground">Analysis</h4>
             <p className="text-sm text-foreground/80">{analysis}</p>
           </div>
         )}
      </div>
    );
  }

  const handleAnalyze = async () => {
    if (!dataset?.datasetId || !data || !x_axis_key || !y_axis_keys.length) return;
    analysisControllerRef.current?.abort();
    const controller = new AbortController();
    analysisControllerRef.current = controller;
    const requestId = beginQuery(dataset.datasetId, controller);
    const effectiveFilters = mergeFilters(llm_filters, filters);
    setIsAnalyzing(true);
    try {
      const response = await aggregateData({
        dataset_id: dataset.datasetId,
        x_axis_key: data.source_x_axis_key || x_axis_key,
        y_axis_keys,
        aggregation: aggregation || 'sum',
        chart_type,
        filters: effectiveFilters,
        limit,
        sort_by: sortBy,
        group_others: groupOthers,
        include_analysis: true,
        time_bucket: data.time_bucket,
        signal: controller.signal,
      });
      if (!isCurrentQuery(requestId, dataset.datasetId)) return;
      setCurrentChart(preserveQueryProvenance(response, data, effectiveFilters));
      if (response.chart_type === 'empty') toast.message('The chart needs clarification. See the response below.');
      else toast.success('Analysis added');
    } catch (error) {
      if (!controller.signal.aborted && isCurrentDataset(dataset.datasetId)) {
        console.error('Analyze error:', error);
        const message = error instanceof Error ? error.message : 'Failed to analyze chart';
        toast.error(message);
      }
    } finally {
      setIsAnalyzing(false);
      finishQuery(requestId);
    }
  };
  
  const handleDrillDown = async (entry: Record<string, unknown>) => {
    if (!dataset?.datasetId || !data || !x_axis_key) return;
    
    // Extract value for the x-axis key from the clicked entry (payload)
    const sourceAxis = data.source_x_axis_key || x_axis_key;
    const xVal = entry[x_axis_key];
    if (!isFilterValue(xVal)) return;
    
    // Protection: Prevent drill-down into "Others"
    if (data.others_label && String(xVal) === data.others_label) {
      toast.warning("Cannot drill down into aggregated 'Others' group.");
      return;
    }

    const toastId = toast.loading(`Loading details for ${String(xVal)}...`);
    drillDownControllerRef.current?.abort();
    const controller = new AbortController();
    drillDownControllerRef.current = controller;
    const requestId = ++drillDownRequestRef.current;
    
    try {
      const effectiveFilters = data.filters ?? mergeFilters(filters, llm_filters);
      const nextPeriodStart = data.time_bucket ? getNextPeriodStart(String(xVal), data.time_bucket) : null;
      const clickedFilters = data.time_bucket && nextPeriodStart
        ? [
            { column: sourceAxis, operator: 'gte' as const, value: String(xVal) },
            { column: sourceAxis, operator: 'lt' as const, value: nextPeriodStart },
          ]
        : [{ column: sourceAxis, values: [xVal] }];
      const drillFilters = mergeFilters(effectiveFilters, clickedFilters);
      
      const result = await drillDown({ 
        dataset_id: dataset.datasetId, 
        filters: drillFilters,
        limit: 50,
        signal: controller.signal,
      });
      
      if (result && result.data && !controller.signal.aborted
        && requestId === drillDownRequestRef.current && isCurrentDataset(dataset.datasetId)) {
        setDrillDownData(result.data);
        setIsDrillDownOpen(true);
        toast.dismiss(toastId);
      }
    } catch (e) {
      if (!controller.signal.aborted && isCurrentDataset(dataset.datasetId)) {
        console.error(e);
        toast.error("Failed to fetch drill-down data", { id: toastId });
      }
    }
  };

  // Format legend names from snake_case to Title Case
  const formatLegendName = (value: string) => {
    return value
      .replace(/_/g, ' ')
      .replace(/\b\w/g, c => c.toUpperCase());
  };

  // Create tooltip element with formats passed as prop
  const tooltipContent = <CustomTooltip formats={formats} aggregation={aggregation} primaryFormat={primaryFormat} />;

  const renderChart = () => {
    switch (chart_type) {
      case 'line':
        return (
          <LineChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.5} />
            <XAxis dataKey={x_axis_key} tick={axisStyle} axisLine={{ stroke: '#3f3f46' }} />
            <YAxis 
              tick={axisStyle} 
              axisLine={{ stroke: '#3f3f46' }} 
              tickFormatter={tickFormatter} 
              label={y_axis_label ? { value: y_axis_label, angle: -90, position: 'insideLeft', style: { fontSize: 11 } } : undefined} 
            />
            <Tooltip content={tooltipContent} />
            {y_axis_keys.map((k, i) => (
              <Line 
                key={k} 
                type="monotone" 
                dataKey={k} 
                stroke={COLORS[i % COLORS.length]} 
                strokeWidth={2} 
                dot={{ fill: COLORS[i % COLORS.length], r: 4 }}
                activeDot={{
                  r: 8,
                  cursor: 'pointer',
                  onClick: (_event, payload) => {
                    const row = getDrillDownRecord(payload);
                    if (row) void handleDrillDown(row);
                  },
                }}
              />
            ))}
          </LineChart>
        );
      case 'area':
        return (
          <AreaChart {...commonProps}>
            <defs>
              {y_axis_keys.map((k, i) => (
                <linearGradient key={k} id={`grad-${k}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={COLORS[i % COLORS.length]} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={COLORS[i % COLORS.length]} stopOpacity={0} />
                </linearGradient>
              ))}
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.5} />
            <XAxis dataKey={x_axis_key} tick={axisStyle} />
            <YAxis tick={axisStyle} tickFormatter={tickFormatter} />
            <Tooltip content={tooltipContent} />
            {y_axis_keys.map((k, i) => (
              <Area 
                key={k} 
                type="monotone" 
                dataKey={k} 
                stroke={COLORS[i % COLORS.length]} 
                fill={`url(#grad-${k})`} 
                activeDot={{
                  r: 8,
                  cursor: 'pointer',
                  onClick: (_event, payload) => {
                    const row = getDrillDownRecord(payload);
                    if (row) void handleDrillDown(row);
                  },
                }}
              />
            ))}
          </AreaChart>
        );
      case 'pie':
        return (
          <PieChart>
            <Pie 
              data={records} 
              dataKey={y_axis_keys[0]} 
              nameKey={x_axis_key} 
              cx="50%" 
              cy="50%" 
              outerRadius={150}
              label={({ name, percent }) => `${name}: ${((percent ?? 0) * 100).toFixed(0)}%`}
              labelLine={{ stroke: 'var(--muted-foreground)' }}
            >
              {records.map((entry, i) => (
                <Cell 
                  key={i} 
                  fill={COLORS[i % COLORS.length]} 
                  onClick={() => handleDrillDown(entry)}
                  cursor="pointer"
                />
              ))}
            </Pie>
            <Tooltip content={tooltipContent} />
            <Legend formatter={formatLegendName} />
          </PieChart>
        );
      case 'composed':
        return (
          <ComposedChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.5} />
            <XAxis dataKey={x_axis_key} tick={axisStyle} />
            <YAxis tick={axisStyle} tickFormatter={tickFormatter} />
            <Tooltip content={tooltipContent} />
            {y_axis_keys.map((k, i) => i % 2 === 0
              ? <Bar key={k} dataKey={k} fill={COLORS[i % COLORS.length]} radius={[4, 4, 0, 0]} opacity={0.8} />
              : <Line key={k} type="monotone" dataKey={k} stroke={COLORS[i % COLORS.length]} strokeWidth={2} />
            )}
          </ComposedChart>
        );
      default: // bar
        return (
          <BarChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.5} />
            <XAxis dataKey={x_axis_key} tick={axisStyle} />
            <YAxis 
              tick={axisStyle} 
              tickFormatter={tickFormatter}
              label={y_axis_label ? { value: y_axis_label, angle: -90, position: 'insideLeft', style: { fontSize: 11 } } : undefined}
            />
            <Tooltip content={tooltipContent} />
            {y_axis_keys.map((k, i) => (
              <Bar 
                key={k} 
                dataKey={k} 
                fill={COLORS[i % COLORS.length]} 
                radius={[4, 4, 0, 0]} 
                onClick={(event) => {
                  const row = getDrillDownRecord(event);
                  if (row) void handleDrillDown(row);
                }}
                cursor="pointer"
              />
            ))}
          </BarChart>
        );
    }
  };

  // Calculate dynamic width for scrolling - keep container fixed, scroll inside
  const minBarWidth = 50; // Minimum pixels per bar/point
  const calculatedWidth = Math.max(records.length * minBarWidth, 800);
  const shouldScroll = records.length > 12; // Enable scroll if more than 12 items

  return (
    <div className={`relative flex w-full min-w-0 flex-col rounded-lg border border-border/50 bg-card/30 p-4 ${compact ? 'min-h-[360px]' : 'min-h-[450px]'}`}>
      {/* Fixed Legend - stays in place during horizontal scroll */}
      {chart_type !== 'pie' && y_axis_keys.length > 0 && (
        <div className="flex flex-wrap gap-4 justify-center mb-4">
          {y_axis_keys.map((key, i) => (
            <div key={key} className="flex items-center gap-2">
              <span 
                className="h-3 w-3 rounded-full" 
                style={{ backgroundColor: COLORS[i % COLORS.length] }} 
              />
              <span className="text-sm text-muted-foreground">
                {formatLegendName(key)}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Scrollable Chart Container */}
      <div 
        className="w-full max-w-full min-w-0 overflow-x-auto overflow-y-hidden" 
        style={{ height: compact ? '300px' : '380px' }}
      >
        <div 
          style={{ 
            width: shouldScroll ? `${calculatedWidth}px` : '100%', 
            height: compact ? '300px' : '380px',
            minWidth: shouldScroll ? `${calculatedWidth}px` : '100%'
          }}
        >
          <ResponsiveContainer width="100%" height="100%">{renderChart()}</ResponsiveContainer>
        </div>
      </div>

      {/* Fixed X-Axis Label - stays in place during horizontal scroll */}
      {x_axis_key && chart_type !== 'pie' && (
        <div className="text-center font-medium text-muted-foreground mt-2 text-sm">
          {formatLegendName(x_axis_key)}
        </div>
      )}

      {showAnalyzeButton && !analysis && (
        <div className="mt-3 flex justify-end">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleAnalyze}
            disabled={isAnalyzing}
            title="Generate a 2-sentence insight for this view"
          >
            {isAnalyzing && <Loader2 className="h-4 w-4 animate-spin" />}
            {isAnalyzing ? 'Analyzing...' : '✨ Analyze this view'}
          </Button>
        </div>
      )}
    </div>
  );
}
