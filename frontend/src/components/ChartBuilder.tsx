'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { BarChart3, LineChart, AreaChart, PieChart, Layers, Play, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useData } from '@/context/DataContext';
import { aggregateData } from '@/lib/api';
import { ChartType, AggregationType } from '@/types';
import { toast } from 'sonner';

const CHART_TYPES: { type: ChartType; icon: React.ReactNode; label: string }[] = [
  { type: 'bar', icon: <BarChart3 className="h-5 w-5" />, label: 'Bar' },
  { type: 'line', icon: <LineChart className="h-5 w-5" />, label: 'Line' },
  { type: 'area', icon: <AreaChart className="h-5 w-5" />, label: 'Area' },
  { type: 'pie', icon: <PieChart className="h-5 w-5" />, label: 'Pie' },
  { type: 'composed', icon: <Layers className="h-5 w-5" />, label: 'Composed' },
];

const AGGREGATIONS: { type: AggregationType; label: string }[] = [
  { type: 'sum', label: 'Sum' },
  { type: 'mean', label: 'Average' },
  { type: 'count', label: 'Count' },
  { type: 'min', label: 'Min' },
  { type: 'max', label: 'Max' },
];

export function ChartBuilder() {
  const { dataset, numericColumns, filters, setCurrentChart, addToHistory, isQuerying, setIsQuerying } = useData();
  
  const [chartType, setChartType] = useState<ChartType>('bar');
  const [xAxis, setXAxis] = useState<string>('');
  const [yAxes, setYAxes] = useState<string[]>([]);
  const [aggregation, setAggregation] = useState<AggregationType>('sum');
  const [limit, setLimit] = useState<number>(20);
  const [sortBy, setSortBy] = useState<'value' | 'label'>('value');

  const allColumns = dataset?.columns ?? [];

  const handleYAxisToggle = (col: string) => {
    setYAxes(prev => 
      prev.includes(col) 
        ? prev.filter(c => c !== col)
        : [...prev, col]
    );
  };

  const handlePlot = async () => {
    if (!dataset || !xAxis || yAxes.length === 0) {
      toast.error('Please select X-axis and at least one Y-axis column');
      return;
    }

    setIsQuerying(true);

    try {
      const response = await aggregateData({
        dataset_id: dataset.datasetId,
        x_axis_key: xAxis,
        y_axis_keys: yAxes,
        aggregation,
        chart_type: chartType,
        filters: filters.length > 0 ? filters : undefined,
        limit,
        sort_by: sortBy,
      });

      setCurrentChart(response);
      addToHistory(
        `${chartType} chart: ${yAxes.join(', ')} by ${xAxis}`,
        response,
        true
      );
      toast.success(`Generated chart with ${response.row_count} data points`);
    } catch (error) {
      console.error('Aggregate error:', error);
      const message = error instanceof Error ? error.message : 'Failed to generate chart';
      toast.error(message);
    } finally {
      setIsQuerying(false);
    }
  };

  if (!dataset) {
    return (
      <div className="rounded-xl border border-dashed border-border/50 bg-card/30 p-8 text-center">
        <p className="text-muted-foreground">Upload a CSV file to use the Chart Builder</p>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      {/* Chart Type Selector */}
      <div>
        <label className="mb-2 block text-sm font-medium text-muted-foreground">Chart Type</label>
        <div className="flex flex-wrap gap-2">
          {CHART_TYPES.map(({ type, icon, label }) => (
            <button
              key={type}
              onClick={() => setChartType(type)}
              className={`flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-all ${
                chartType === type
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-white/5 text-muted-foreground hover:bg-white/10 hover:text-foreground'
              }`}
            >
              {icon}
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* X-Axis Selector */}
      <div>
        <label className="mb-2 block text-sm font-medium text-muted-foreground">X-Axis (Category)</label>
        <select
          value={xAxis}
          onChange={(e) => setXAxis(e.target.value)}
          className="w-full rounded-lg border border-border/50 bg-card/50 px-4 py-2.5 text-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
        >
          <option value="">Select column...</option>
          {allColumns.map(col => (
            <option key={col.name} value={col.name}>
              {col.name} {col.is_numeric ? '(numeric)' : col.is_datetime ? '(date)' : ''}
            </option>
          ))}
        </select>
      </div>

      {/* Y-Axis Multi-Select */}
      <div>
        <label className="mb-2 block text-sm font-medium text-muted-foreground">
          Y-Axis (Values) — Select numeric columns
        </label>
        <div className="flex flex-wrap gap-2">
          {numericColumns.length === 0 ? (
            <p className="text-sm text-muted-foreground/60">No numeric columns available</p>
          ) : (
            numericColumns.map(col => (
              <button
                key={col.name}
                onClick={() => handleYAxisToggle(col.name)}
                className={`rounded-full px-3 py-1.5 text-sm transition-all ${
                  yAxes.includes(col.name)
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-white/5 text-muted-foreground hover:bg-white/10'
                }`}
              >
                {col.name}
              </button>
            ))
          )}
        </div>
        {yAxes.length > 0 && (
          <p className="mt-2 text-xs text-muted-foreground">
            Selected: {yAxes.join(', ')}
          </p>
        )}
      </div>

      {/* Aggregation Selector */}
      <div>
        <label className="mb-2 block text-sm font-medium text-muted-foreground">Aggregation</label>
        <div className="flex flex-wrap gap-2">
          {AGGREGATIONS.map(({ type, label }) => (
            <button
              key={type}
              onClick={() => setAggregation(type)}
              className={`rounded-lg px-3 py-1.5 text-sm transition-all ${
                aggregation === type
                  ? 'bg-secondary text-secondary-foreground'
                  : 'bg-white/5 text-muted-foreground hover:bg-white/10'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Advanced Controls */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="mb-2 block text-sm font-medium text-muted-foreground">Top Results: {limit}</label>
          <input 
            type="range" 
            min="5" 
            max="50" 
            step="1"
            value={limit}
            onChange={(e) => setLimit(parseInt(e.target.value))}
            className="w-full h-2 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
          />
        </div>
        <div>
          <label className="mb-2 block text-sm font-medium text-muted-foreground">Sort By</label>
          <div className="flex rounded-lg bg-white/5 p-1">
            <button
              onClick={() => setSortBy('value')}
              className={`flex-1 rounded py-1 text-xs font-medium transition-all ${sortBy === 'value' ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}
            >
              Value
            </button>
            <button
              onClick={() => setSortBy('label')}
              className={`flex-1 rounded py-1 text-xs font-medium transition-all ${sortBy === 'label' ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}
            >
              Label (A-Z)
            </button>
          </div>
        </div>
      </div>

      {/* Plot Button */}
      <Button
        onClick={handlePlot}
        disabled={!xAxis || yAxes.length === 0 || isQuerying}
        className="w-full gap-2"
        size="lg"
      >
        {isQuerying ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" />
            Generating...
          </>
        ) : (
          <>
            <Play className="h-4 w-4" />
            Plot Chart
          </>
        )}
      </Button>
    </motion.div>
  );
}
