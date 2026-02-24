'use client';

import React, { memo, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
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

const COLORS = [
  'hsl(252, 87%, 64%)', 'hsl(173, 80%, 40%)', 'hsl(43, 96%, 56%)',
  'hsl(346, 77%, 59%)', 'hsl(199, 89%, 48%)', 'hsl(280, 65%, 60%)',
  'hsl(150, 60%, 45%)', 'hsl(30, 90%, 55%)',
];

interface SmartChartProps {
  chartData?: ChartResponse;
  showAnalyzeButton?: boolean;
}

// Format value based on column format
function formatValue(value: number, format: ColumnFormat, compact: boolean = false): string {
  if (typeof value !== 'number' || isNaN(value)) return String(value);
  
  if (format === 'currency') {
    if (compact) {
      if (Math.abs(value) >= 1e6) return `$${(value / 1e6).toFixed(1)}M`;
      if (Math.abs(value) >= 1e3) return `$${(value / 1e3).toFixed(0)}K`;
      return `$${value.toFixed(0)}`;
    }
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(value);
  }
  
  if (format === 'percentage') {
    return `${(value * 100).toFixed(1)}%`;
  }
  
  // Default number
  if (compact) {
    if (Math.abs(value) >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
    if (Math.abs(value) >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  }
  return value.toLocaleString();
}

// ============================================================================
// CustomTooltip - Extracted as memoized component for performance
// ============================================================================

interface TooltipPayloadEntry {
  name: string;
  value: number;
  color: string;
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: TooltipPayloadEntry[];
  label?: string;
  formats: Record<string, ColumnFormat>;
}

const CustomTooltip = memo(function CustomTooltip({ 
  active, 
  payload, 
  label, 
  formats 
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
            {formatValue(entry.value, (formats[entry.name] || 'number') as ColumnFormat, false)}
          </span>
        </p>
      ))}
    </div>
  );
});

// ============================================================================
// SmartChart - Main chart component using composed Recharts architecture
// ============================================================================

export function SmartChart({ chartData, showAnalyzeButton = true }: SmartChartProps) {
  const { currentChart, dataset, filters, setCurrentChart, setDrillDownData, setIsDrillDownOpen, limit, groupOthers, sortBy } = useData();
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const data = chartData || currentChart;
  
  if (!data?.data.length && !data?.answer) return null;

  const { data: records, x_axis_key, y_axis_keys, chart_type, y_axis_label, answer, analysis, aggregation, llm_filters } = data;
  
  // Text-Only Answer View
  if (chart_type === 'empty' || (answer && !records.length)) {
    return (
      <div className="flex h-full min-h-[400px] flex-col overflow-y-auto rounded-lg p-6">
         <div className="mb-6 rounded-lg bg-primary/10 p-4">
           <h3 className="mb-2 text-lg font-semibold text-primary">Insight</h3>
           <div className="whitespace-pre-wrap font-mono text-sm leading-relaxed">{answer}</div>
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
  
  const formats = dataset?.columnFormats ?? {};

  const handleAnalyze = async () => {
    if (!dataset?.datasetId || !x_axis_key || !y_axis_keys?.length) return;
    setIsAnalyzing(true);
    try {
      const response = await aggregateData({
        dataset_id: dataset.datasetId,
        x_axis_key,
        y_axis_keys,
        aggregation: aggregation || 'sum',
        chart_type,
        filters: filters.length > 0 ? filters : undefined,
        limit,
        sort_by: sortBy,
        group_others: groupOthers,
        include_analysis: true,
      });
      setCurrentChart(response);
      toast.success('Analysis added');
    } catch (error) {
      console.error('Analyze error:', error);
      const message = error instanceof Error ? error.message : 'Failed to analyze chart';
      toast.error(message);
    } finally {
      setIsAnalyzing(false);
    }
  };
  
  const handleDrillDown = async (entry: any) => {
    if (!dataset?.datasetId || !x_axis_key) return;
    
    // Extract value for the x-axis key from the clicked entry (payload)
    const xVal = entry[x_axis_key];
    if (xVal === undefined) return;
    
    // Protection: Prevent drill-down into "Others"
    if (xVal === 'Others') {
      toast.warning("Cannot drill down into aggregated 'Others' group.");
      return;
    }

    const toastId = toast.loading(`Loading details for ${xVal}...`);
    
    try {
      const combinedFilters = [
        ...filters,
        ...(llm_filters ?? []),
      ];
      const drillFilters = [
        ...combinedFilters,
        { column: x_axis_key, values: [xVal] }
      ];
      
      const result = await drillDown({ 
        dataset_id: dataset.datasetId, 
        filters: drillFilters,
        limit: 50 
      });
      
      if (result && result.data) {
        setDrillDownData(result.data);
        setIsDrillDownOpen(true);
        toast.dismiss(toastId);
      }
    } catch (e) {
      console.error(e);
      toast.error("Failed to fetch drill-down data", { id: toastId });
    }
  };
  
  // Get format for primary Y axis
  const primaryFormat = (formats[y_axis_keys[0]] || 'number') as ColumnFormat;
  
  // Memoize the tick formatter to prevent recreation on each render
  const tickFormatter = useMemo(
    () => (v: number) => formatValue(v, primaryFormat, true),
    [primaryFormat]
  );
  
  // Memoize axis style object - use explicit light color for dark background visibility
  const axisStyle = useMemo(
    () => ({ fontSize: 11, fill: '#a1a1aa' }), // zinc-400 - visible on dark backgrounds
    []
  );
  
  // Memoize common chart props - increased left margin for Y-axis label
  const commonProps = useMemo(
    () => ({ data: records, margin: { top: 20, right: 30, left: 70, bottom: 20 } }),
    [records]
  );

  // Format legend names from snake_case to Title Case
  const formatLegendName = (value: string) => {
    return value
      .replace(/_/g, ' ')
      .replace(/\b\w/g, c => c.toUpperCase());
  };

  // Create tooltip element with formats passed as prop
  const tooltipContent = <CustomTooltip formats={formats} />;

  const renderChart = () => {
    switch (chart_type) {
      case 'line':
        return (
          <LineChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.5} />
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
                  onClick: (event: any, payload: any) => handleDrillDown(payload?.payload),
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
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.5} />
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
                  onClick: (event: any, payload: any) => handleDrillDown(payload?.payload),
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
              labelLine={{ stroke: 'hsl(var(--muted-foreground))' }}
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
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.5} />
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
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.5} />
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
                onClick={(e: any) => handleDrillDown(e.payload)}
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
    <div className="relative flex flex-col w-full min-w-0 min-h-[450px] rounded-lg border border-border/50 bg-card/30 p-4">
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
        style={{ height: '380px' }}
      >
        <div 
          style={{ 
            width: shouldScroll ? `${calculatedWidth}px` : '100%', 
            height: '380px',
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
