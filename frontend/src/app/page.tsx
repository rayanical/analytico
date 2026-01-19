'use client';

import { motion } from 'framer-motion';
import { Sidebar } from '@/components/Sidebar';
import { FileUploader } from '@/components/FileUploader';
import { ChatInterface } from '@/components/ChatInterface';
import { SmartChart } from '@/components/SmartChart';
import { ChartSkeleton } from '@/components/ChartSkeleton';
import { useData } from '@/context/DataContext';
import { BarChart3, Sparkles } from 'lucide-react';

export default function Home() {
  const { dataState, currentChart, isQuerying } = useData();

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <Sidebar />
      
      {/* Main Content */}
      <main className="relative flex flex-1 flex-col overflow-hidden">
        {/* Background gradient */}
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-primary/5" />
        
        {/* Content area */}
        <div className="relative flex flex-1 flex-col overflow-y-auto p-8">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-8"
          >
            <h1 className="text-3xl font-bold text-foreground">
              Data Visualization
            </h1>
            <p className="mt-2 text-muted-foreground">
              Upload a CSV file and use natural language to create beautiful charts
            </p>
          </motion.div>

          {/* File Upload Section */}
          <section className="mb-8">
            <FileUploader />
          </section>

          {/* Chart Display Area */}
          <section className="flex-1">
            {isQuerying ? (
              <ChartSkeleton />
            ) : currentChart ? (
              <SmartChart />
            ) : (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex h-[400px] flex-col items-center justify-center rounded-xl border border-dashed border-border/50 bg-card/30"
              >
                {dataState ? (
                  <>
                    <div className="mb-4 rounded-full bg-primary/10 p-4">
                      <Sparkles className="h-8 w-8 text-primary" />
                    </div>
                    <h3 className="text-lg font-medium text-foreground">
                      Ready to visualize!
                    </h3>
                    <p className="mt-2 text-center text-sm text-muted-foreground">
                      Ask a question below to generate a chart from your data.
                      <br />
                      Try: &quot;Show me a bar chart of revenue by month&quot;
                    </p>
                  </>
                ) : (
                  <>
                    <div className="mb-4 rounded-full bg-muted/50 p-4">
                      <BarChart3 className="h-8 w-8 text-muted-foreground" />
                    </div>
                    <h3 className="text-lg font-medium text-muted-foreground">
                      No data loaded
                    </h3>
                    <p className="mt-2 text-center text-sm text-muted-foreground/60">
                      Upload a CSV file above to get started
                    </p>
                  </>
                )}
              </motion.div>
            )}
          </section>
        </div>

        {/* Chat Input - Fixed at bottom */}
        <div className="relative border-t border-border/50 bg-background/80 p-6 backdrop-blur-xl">
          <div className="mx-auto max-w-4xl">
            <ChatInterface />
          </div>
        </div>
      </main>
    </div>
  );
}
