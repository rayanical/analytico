'use client';

import React from 'react';
import { motion } from 'framer-motion';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  AreaChart,
  Area,
  PieChart,
  Pie,
  Cell,
  ComposedChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import { useData } from '@/context/DataContext';
import { ChartResponse } from '@/types';

// Color palette for chart series
const CHART_COLORS = [
  'hsl(252, 87%, 64%)', // Purple
  'hsl(173, 80%, 40%)', // Teal
  'hsl(43, 96%, 56%)',  // Amber
  'hsl(346, 77%, 59%)', // Rose
  'hsl(199, 89%, 48%)', // Sky
  'hsl(280, 65%, 60%)', // Violet
  'hsl(150, 60%, 45%)', // Emerald
  'hsl(30, 90%, 55%)',  // Orange
];

interface SmartChartProps {
  chartData?: ChartResponse;
}

// Smart axis tick formatter
function formatAxisTick(value: number, columnName: string): string {
  if (typeof value !== 'number' || isNaN(value)) return String(value);
  
  const lowerCol = columnName.toLowerCase();
  
  // Currency formatting
  if (/sales|revenue|rev|cost|price|amount|profit|income/.test(lowerCol)) {
    if (Math.abs(value) >= 1000000) {
      return `$${(value / 1000000).toFixed(1)}M`;
    }
    if (Math.abs(value) >= 1000) {
      return `$${(value / 1000).toFixed(0)}K`;
    }
    return `$${value.toFixed(0)}`;
  }
  
  // Percentage formatting
  if (/rate|percent|ratio|pct/.test(lowerCol)) {
    return `${value.toFixed(1)}%`;
  }
  
  // Compact number formatting
  if (Math.abs(value) >= 1000000) {
    return `${(value / 1000000).toFixed(1)}M`;
  }
  if (Math.abs(value) >= 1000) {
    return `${(value / 1000).toFixed(1)}K`;
  }
  
  return value.toLocaleString();
}

// Smart tooltip formatter
function formatTooltipValue(value: number, columnName: string): string {
  if (typeof value !== 'number' || isNaN(value)) return String(value);
  
  const lowerCol = columnName.toLowerCase();
  
  // Currency formatting
  if (/sales|revenue|rev|cost|price|amount|profit|income/.test(lowerCol)) {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 0,
    }).format(value);
  }
  
  // Percentage formatting
  if (/rate|percent|ratio|pct/.test(lowerCol)) {
    return `${value.toFixed(2)}%`;
  }
  
  return value.toLocaleString();
}

export function SmartChart({ chartData }: SmartChartProps) {
  const { currentChart } = useData();
  
  const data = chartData || currentChart;
  
  if (!data || !data.data.length) {
    return null;
  }

  const { data: chartRecords, x_axis_key, y_axis_keys, chart_type, title } = data;

  // Common chart props
  const commonProps = {
    data: chartRecords,
    margin: { top: 20, right: 30, left: 20, bottom: 20 },
  };

  // Axis styling
  const axisStyle = {
    fontSize: 11,
    fill: 'hsl(var(--muted-foreground))',
  };

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: { active?: boolean; payload?: Array<{ name: string; value: number; color: string }>; label?: string }) => {
    if (!active || !payload?.length) return null;
    
    return (
      <div className="rounded-lg border border-border bg-card p-3 shadow-xl">
        <p className="mb-2 font-medium text-foreground">{label}</p>
        {payload.map((entry, index) => (
          <p key={index} className="flex items-center gap-2 text-sm">
            <span
              className="h-3 w-3 rounded-full"
              style={{ backgroundColor: entry.color }}
            />
            <span className="text-muted-foreground">{entry.name}:</span>
            <span className="font-medium">{formatTooltipValue(entry.value, entry.name)}</span>
          </p>
        ))}
      </div>
    );
  };

  // Determine primary Y-axis column for formatting
  const primaryYColumn = y_axis_keys[0] || '';

  // Render the appropriate chart type
  const renderChart = () => {
    switch (chart_type) {
      case 'line':
        return (
          <LineChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.5} />
            <XAxis dataKey={x_axis_key} tick={axisStyle} axisLine={{ stroke: 'hsl(var(--border))' }} />
            <YAxis
              tick={axisStyle}
              axisLine={{ stroke: 'hsl(var(--border))' }}
              tickFormatter={(v) => formatAxisTick(v, primaryYColumn)}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            {y_axis_keys.map((key, index) => (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                stroke={CHART_COLORS[index % CHART_COLORS.length]}
                strokeWidth={2}
                dot={{ fill: CHART_COLORS[index % CHART_COLORS.length], strokeWidth: 0, r: 4 }}
                activeDot={{ r: 6, strokeWidth: 0 }}
              />
            ))}
          </LineChart>
        );

      case 'area':
        return (
          <AreaChart {...commonProps}>
            <defs>
              {y_axis_keys.map((key, index) => (
                <linearGradient key={key} id={`gradient-${key}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={CHART_COLORS[index % CHART_COLORS.length]} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={CHART_COLORS[index % CHART_COLORS.length]} stopOpacity={0} />
                </linearGradient>
              ))}
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.5} />
            <XAxis dataKey={x_axis_key} tick={axisStyle} axisLine={{ stroke: 'hsl(var(--border))' }} />
            <YAxis
              tick={axisStyle}
              axisLine={{ stroke: 'hsl(var(--border))' }}
              tickFormatter={(v) => formatAxisTick(v, primaryYColumn)}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            {y_axis_keys.map((key, index) => (
              <Area
                key={key}
                type="monotone"
                dataKey={key}
                stroke={CHART_COLORS[index % CHART_COLORS.length]}
                strokeWidth={2}
                fill={`url(#gradient-${key})`}
              />
            ))}
          </AreaChart>
        );

      case 'pie':
        // For pie charts, use the first y-axis key
        const pieKey = y_axis_keys[0];
        return (
          <PieChart>
            <Pie
              data={chartRecords}
              dataKey={pieKey}
              nameKey={x_axis_key}
              cx="50%"
              cy="50%"
              outerRadius={150}
              label={({ name, percent }) => `${name}: ${((percent ?? 0) * 100).toFixed(0)}%`}
              labelLine={{ stroke: 'hsl(var(--muted-foreground))' }}
            >
              {chartRecords.map((_, index) => (
                <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
            <Legend />
          </PieChart>
        );

      case 'composed':
        return (
          <ComposedChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.5} />
            <XAxis dataKey={x_axis_key} tick={axisStyle} axisLine={{ stroke: 'hsl(var(--border))' }} />
            <YAxis
              tick={axisStyle}
              axisLine={{ stroke: 'hsl(var(--border))' }}
              tickFormatter={(v) => formatAxisTick(v, primaryYColumn)}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            {y_axis_keys.map((key, index) => {
              if (index % 2 === 0) {
                return (
                  <Bar
                    key={key}
                    dataKey={key}
                    fill={CHART_COLORS[index % CHART_COLORS.length]}
                    radius={[4, 4, 0, 0]}
                    opacity={0.8}
                  />
                );
              }
              return (
                <Line
                  key={key}
                  type="monotone"
                  dataKey={key}
                  stroke={CHART_COLORS[index % CHART_COLORS.length]}
                  strokeWidth={2}
                  dot={{ fill: CHART_COLORS[index % CHART_COLORS.length], strokeWidth: 0, r: 4 }}
                />
              );
            })}
          </ComposedChart>
        );

      case 'bar':
      default:
        return (
          <BarChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.5} />
            <XAxis dataKey={x_axis_key} tick={axisStyle} axisLine={{ stroke: 'hsl(var(--border))' }} />
            <YAxis
              tick={axisStyle}
              axisLine={{ stroke: 'hsl(var(--border))' }}
              tickFormatter={(v) => formatAxisTick(v, primaryYColumn)}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            {y_axis_keys.map((key, index) => (
              <Bar
                key={key}
                dataKey={key}
                fill={CHART_COLORS[index % CHART_COLORS.length]}
                radius={[4, 4, 0, 0]}
              />
            ))}
          </BarChart>
        );
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: 'easeOut' }}
      className="h-[400px] w-full"
    >
      <ResponsiveContainer width="100%" height="100%">
        {renderChart()}
      </ResponsiveContainer>
    </motion.div>
  );
}
