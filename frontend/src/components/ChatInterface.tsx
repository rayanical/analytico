'use client';

import React, { useState,FormEvent } from 'react';
import { motion } from 'framer-motion';
import { Send, Sparkles, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useData } from '@/context/DataContext';
import { queryChart } from '@/lib/api';
import { toast } from 'sonner';

export function ChatInterface() {
  const { dataState, setCurrentChart, addToHistory, isQuerying, setIsQuerying } = useData();
  const [query, setQuery] = useState('');

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    
    if (!query.trim() || !dataState || isQuerying) return;

    setIsQuerying(true);

    try {
      const chartConfig = await queryChart({
        user_prompt: query.trim(),
        columns: dataState.columns,
        data_summary: dataState.summary,
      });

      setCurrentChart(chartConfig);
      addToHistory(query.trim(), chartConfig);
      setQuery('');
      toast.success(`Generated: ${chartConfig.title}`);
    } catch (error) {
      console.error('Query error:', error);
      const message = error instanceof Error ? error.message : 'Failed to generate chart';
      toast.error(message);
    } finally {
      setIsQuerying(false);
    }
  };

  const isDisabled = !dataState || isQuerying;

  const suggestions = [
    'Show me a bar chart of revenue by month',
    'Compare sales vs costs over time',
    'Create an area chart of monthly trends',
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full"
    >
      {/* Suggestions */}
      {dataState && !isQuerying && (
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mb-4 flex flex-wrap gap-2"
        >
          {suggestions.map((suggestion, index) => (
            <button
              key={index}
              onClick={() => setQuery(suggestion)}
              className="rounded-full border border-border/50 bg-white/[0.02] px-4 py-2 text-sm text-muted-foreground transition-all hover:border-primary/50 hover:bg-primary/5 hover:text-primary"
            >
              {suggestion}
            </button>
          ))}
        </motion.div>
      )}

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="relative">
        <div className="relative flex items-center gap-3 rounded-xl border border-border/50 bg-card/50 p-2 backdrop-blur-sm transition-all focus-within:border-primary/50 focus-within:shadow-lg focus-within:shadow-primary/5">
          <Sparkles className="ml-3 h-5 w-5 text-primary/60" />
          
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={dataState ? "Ask me to create a chart... (e.g., 'Show revenue by month as a bar chart')" : "Upload a CSV file first to start creating charts"}
            disabled={isDisabled}
            className="flex-1 border-0 bg-transparent text-base placeholder:text-muted-foreground/50 focus-visible:ring-0"
          />

          <Button
            type="submit"
            disabled={isDisabled || !query.trim()}
            size="icon"
            className="h-10 w-10 shrink-0 rounded-lg bg-primary text-primary-foreground transition-all hover:bg-primary/90 disabled:opacity-50"
          >
            {isQuerying ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </Button>
        </div>
      </form>
    </motion.div>
  );
}
