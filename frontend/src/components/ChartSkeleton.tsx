'use client';

import { motion } from 'framer-motion';
import { Skeleton } from '@/components/ui/skeleton';

export function ChartSkeleton() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full"
    >
      <div className="rounded-xl border border-border/50 bg-card/50 p-6 backdrop-blur-sm">
        {/* Title skeleton */}
        <Skeleton className="mb-6 h-7 w-64" />
        
        {/* Chart skeleton */}
        <div className="relative h-[400px] w-full">
          {/* Y-axis labels */}
          <div className="absolute left-0 top-0 flex h-full flex-col justify-between py-4">
            {[...Array(5)].map((_, i) => (
              <Skeleton key={i} className="h-3 w-8" />
            ))}
          </div>
          
          {/* Chart bars/lines */}
          <div className="ml-12 flex h-full items-end justify-around gap-4 pb-8">
            {[...Array(8)].map((_, i) => (
              <motion.div
                key={i}
                initial={{ scaleY: 0 }}
                animate={{ scaleY: 1 }}
                transition={{ 
                  delay: i * 0.1,
                  duration: 0.4,
                  ease: 'easeOut'
                }}
                className="flex-1 origin-bottom"
                style={{ height: `${30 + Math.random() * 60}%` }}
              >
                <Skeleton className="h-full w-full rounded-t-md" />
              </motion.div>
            ))}
          </div>
          
          {/* X-axis labels */}
          <div className="absolute bottom-0 left-12 right-0 flex justify-around">
            {[...Array(8)].map((_, i) => (
              <Skeleton key={i} className="h-3 w-12" />
            ))}
          </div>
        </div>
        
        {/* Legend skeleton */}
        <div className="mt-4 flex justify-center gap-6">
          <div className="flex items-center gap-2">
            <Skeleton className="h-3 w-3 rounded-full" />
            <Skeleton className="h-3 w-16" />
          </div>
          <div className="flex items-center gap-2">
            <Skeleton className="h-3 w-3 rounded-full" />
            <Skeleton className="h-3 w-20" />
          </div>
        </div>
      </div>
    </motion.div>
  );
}
