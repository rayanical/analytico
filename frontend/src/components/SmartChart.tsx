'use client';

import React from 'react';
import { motion } from 'framer-motion';
import {
  ResponsiveContainer, BarChart, Bar, LineChart, Line, AreaChart, Area,
  PieChart, Pie, Cell, ComposedChart, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
} from 'recharts';
import { useData } from '@/context/DataContext';
import { ChartResponse, ColumnFormat } from '@/types';

const COLORS = [
  'hsl(252, 87%, 64%)', 'hsl(173, 80%, 40%)', 'hsl(43, 96%, 56%)',
  'hsl(346, 77%, 59%)', 'hsl(199, 89%, 48%)', 'hsl(280, 65%, 60%)',
  'hsl(150, 60%, 45%)', 'hsl(30, 90%, 55%)',
];

interface SmartChartProps {
  chartData?: ChartResponse;
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

export function SmartChart({ chartData }: SmartChartProps) {
  const { currentChart, dataset } = useData();
  const data = chartData || currentChart;
  
  if (!data?.data.length) return null;

  const { data: records, x_axis_key, y_axis_keys, chart_type, y_axis_label } = data;
  const formats = dataset?.columnFormats ?? {};
  
  // Get format for primary Y axis
  const primaryFormat = (formats[y_axis_keys[0]] || 'number') as ColumnFormat;
  
  const tickFormatter = (v: number) => formatValue(v, primaryFormat, true);
  
  const CustomTooltip = ({ active, payload, label }: { active?: boolean; payload?: Array<{ name: string; value: number; color: string }>; label?: string }) => {
    if (!active || !payload?.length) return null;
    return (
      <div className="rounded-lg border border-border bg-card p-3 shadow-xl">
        <p className="mb-2 font-medium">{label}</p>
        {payload.map((e, i) => (
          <p key={i} className="flex items-center gap-2 text-sm">
            <span className="h-3 w-3 rounded-full" style={{ backgroundColor: e.color }} />
            <span className="text-muted-foreground">{e.name}:</span>
            <span className="font-medium">{formatValue(e.value, (formats[e.name] || 'number') as ColumnFormat, false)}</span>
          </p>
        ))}
      </div>
    );
  };

  const axisStyle = { fontSize: 11, fill: 'hsl(var(--muted-foreground))' };
  const commonProps = { data: records, margin: { top: 20, right: 30, left: 20, bottom: 20 } };

  const renderChart = () => {
    switch (chart_type) {
      case 'line':
        return (
          <LineChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.5} />
            <XAxis dataKey={x_axis_key} tick={axisStyle} axisLine={{ stroke: 'hsl(var(--border))' }} />
            <YAxis tick={axisStyle} axisLine={{ stroke: 'hsl(var(--border))' }} tickFormatter={tickFormatter} label={y_axis_label ? { value: y_axis_label, angle: -90, position: 'insideLeft', style: { fontSize: 11 } } : undefined} />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            {y_axis_keys.map((k, i) => <Line key={k} type="monotone" dataKey={k} stroke={COLORS[i % COLORS.length]} strokeWidth={2} dot={{ fill: COLORS[i % COLORS.length], r: 4 }} />)}
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
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            {y_axis_keys.map((k, i) => <Area key={k} type="monotone" dataKey={k} stroke={COLORS[i % COLORS.length]} fill={`url(#grad-${k})`} />)}
          </AreaChart>
        );
      case 'pie':
        return (
          <PieChart>
            <Pie data={records} dataKey={y_axis_keys[0]} nameKey={x_axis_key} cx="50%" cy="50%" outerRadius={150}
              label={({ name, percent }) => `${name}: ${((percent ?? 0) * 100).toFixed(0)}%`}
              labelLine={{ stroke: 'hsl(var(--muted-foreground))' }}>
              {records.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
            <Legend />
          </PieChart>
        );
      case 'composed':
        return (
          <ComposedChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.5} />
            <XAxis dataKey={x_axis_key} tick={axisStyle} />
            <YAxis tick={axisStyle} tickFormatter={tickFormatter} />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
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
            <YAxis tick={axisStyle} tickFormatter={tickFormatter} />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            {y_axis_keys.map((k, i) => <Bar key={k} dataKey={k} fill={COLORS[i % COLORS.length]} radius={[4, 4, 0, 0]} />)}
          </BarChart>
        );
    }
  };

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="h-[400px] w-full">
      <ResponsiveContainer width="100%" height="100%">{renderChart()}</ResponsiveContainer>
    </motion.div>
  );
}
