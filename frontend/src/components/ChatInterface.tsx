'use client';

import React, { useState, FormEvent } from 'react';
import { motion } from 'framer-motion';
import { Send, Sparkles, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useData } from '@/context/DataContext';
import { queryChart } from '@/lib/api';
import { toast } from 'sonner';

export function ChatInterface() {
  const { dataset, filters, setCurrentChart, addToHistory, isQuerying, setIsQuerying, setViewMode, groupOthers, limit } = useData();
  const [query, setQuery] = useState('');
  const [isInputFocused, setIsInputFocused] = useState(false);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!query.trim() || !dataset || isQuerying) return;

    setIsQuerying(true);
    try {
      const response = await queryChart({
        dataset_id: dataset.datasetId,
        user_prompt: query.trim(),
        filters: filters.length > 0 ? filters : undefined,
        limit: limit,
        group_others: groupOthers,
      });
      setCurrentChart(response);
      addToHistory(query.trim(), response, false);
      setViewMode('chart');
      setQuery('');
      setIsInputFocused(false);
      toast.success(`Generated: ${response.title}`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Failed to generate chart');
    } finally {
      setIsQuerying(false);
    }
  };

  const handleSuggestion = (s: string) => {
    setQuery(s);
  };

  const isDisabled = !dataset || isQuerying;
  
  // Use dynamic suggestions from dataset, fallback to generic
  const suggestions = dataset?.suggestions ?? [
    'Show me a summary of the data',
    'What are the main trends?',
    'Compare key metrics',
  ];

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="w-full">
      {/* Dynamic Suggestions */}
      {dataset && !isQuerying && isInputFocused && (
        <motion.div
          initial={{ opacity: 0, y: 10, scale: 0.98 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{ type: 'spring', stiffness: 380, damping: 28 }}
          className="mb-4 rounded-xl border border-border/40 bg-background/40 p-2 backdrop-blur-sm"
        >
          <div className="grid grid-cols-1 gap-2 md:grid-cols-3">
            {suggestions.map((s, i) => (
              <button
                key={i}
                onMouseDown={(e) => {
                  e.preventDefault();
                  handleSuggestion(s);
                }}
                title={s}
                className="min-h-[74px] rounded-xl border border-border/50 bg-white/[0.02] px-4 py-3 text-left text-sm text-muted-foreground transition-all hover:border-primary/50 hover:bg-primary/5 hover:text-primary"
              >
                <p className="line-clamp-2 leading-relaxed">{s}</p>
              </button>
            ))}
          </div>
        </motion.div>
      )}

      <form onSubmit={handleSubmit} className="relative">
        <div className="relative flex items-center gap-3 rounded-xl border border-border/50 bg-card/50 p-2 backdrop-blur-sm transition-all focus-within:border-primary/50 focus-within:shadow-lg focus-within:shadow-primary/5">
          <Sparkles className="ml-3 h-5 w-5 text-primary/60" />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onFocus={() => setIsInputFocused(true)}
            onBlur={() => setIsInputFocused(false)}
            placeholder={dataset ? "Ask a business question (e.g., trends, top drivers, comparisons)..." : "Upload a CSV first"}
            disabled={isDisabled}
            className="flex-1 border-0 bg-transparent text-base placeholder:text-muted-foreground/50 focus-visible:ring-0"
          />
          <Button type="submit" disabled={isDisabled || !query.trim()} size="icon" className="h-10 w-10 shrink-0 rounded-lg">
            {isQuerying ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          </Button>
        </div>
      </form>
    </motion.div>
  );
}
