'use client';

import React, { useMemo } from 'react';
import { Responsive, WidthProvider, type Layout } from 'react-grid-layout';
import { ExternalLink, GripVertical, PinOff } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { SmartChart } from '@/components/SmartChart';
import { useData } from '@/context/DataContext';
import { DashboardWidget, DashboardLayoutItem } from '@/types';

const ResponsiveGridLayout = WidthProvider(Responsive);

function toGridLayout(widget: DashboardWidget): Layout {
  return {
    i: widget.id,
    x: widget.layout.x,
    y: widget.layout.y,
    w: widget.layout.w,
    h: widget.layout.h,
    minW: widget.layout.minW ?? 6,
    minH: widget.layout.minH ?? 9,
  };
}

function toDashboardLayoutItem(layout: Layout): DashboardLayoutItem {
  return {
    i: layout.i,
    x: layout.x,
    y: layout.y,
    w: layout.w,
    h: layout.h,
    minW: layout.minW,
    minH: layout.minH,
  };
}

export function DashboardCanvas() {
  const {
    dashboardWidgets,
    removeWidget,
    updateWidgetLayout,
    setCurrentChart,
    setWorkspaceMode,
    setViewMode,
  } = useData();

  const layouts = useMemo(
    () => ({
      lg: dashboardWidgets.map(toGridLayout),
    }),
    [dashboardWidgets],
  );

  if (dashboardWidgets.length === 0) {
    return (
      <div className="flex min-h-[480px] items-center justify-center rounded-xl border border-dashed border-border/50 bg-card/20 p-8">
        <div className="text-center">
          <h3 className="text-lg font-semibold">Your dashboard is empty</h3>
          <p className="mt-2 text-sm text-muted-foreground">
            Pin charts from Explore to build your report.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div
      id="dashboard-report-root"
      className="dashboard-grid-container min-h-[75vh] w-full overflow-auto rounded-xl border border-border/50 bg-card/20 p-4"
    >
      <ResponsiveGridLayout
        className="layout"
        layouts={layouts}
        breakpoints={{ lg: 1400, md: 996, sm: 768, xs: 480, xxs: 0 }}
        cols={{ lg: 24, md: 16, sm: 8, xs: 4, xxs: 2 }}
        rowHeight={56}
        margin={[16, 16]}
        containerPadding={[0, 0]}
        isDraggable
        isResizable
        resizeHandles={['se']}
        draggableHandle=".dashboard-widget-drag-handle"
        compactType="vertical"
        preventCollision={false}
        onLayoutChange={(currentLayout: Layout[]) => {
          updateWidgetLayout(currentLayout.map(toDashboardLayoutItem));
        }}
      >
        {dashboardWidgets.map((widget) => (
          <div
            key={widget.id}
            className="rounded-xl border border-border/50 bg-background/80 shadow-sm backdrop-blur-sm"
          >
            <div className="dashboard-widget-drag-handle flex items-center justify-between gap-2 border-b border-border/50 px-3 py-2">
              <div className="flex min-w-0 items-center gap-2">
                <GripVertical className="h-4 w-4 shrink-0 text-muted-foreground" />
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium">{widget.title}</p>
                  <p className="truncate text-xs text-muted-foreground">{widget.sourceQuery}</p>
                </div>
              </div>
              <div className="flex items-center gap-1">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7"
                  title="Open in Explore"
                  onClick={() => {
                    setCurrentChart(widget.chart);
                    setViewMode('chart');
                    setWorkspaceMode('explore');
                  }}
                >
                  <ExternalLink className="h-4 w-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7 text-muted-foreground hover:text-destructive"
                  title="Remove from dashboard"
                  onClick={() => removeWidget(widget.id)}
                >
                  <PinOff className="h-4 w-4" />
                </Button>
              </div>
            </div>
            <div className="h-[calc(100%-44px)] p-2">
              <SmartChart chartData={widget.chart} showAnalyzeButton={false} />
            </div>
          </div>
        ))}
      </ResponsiveGridLayout>
    </div>
  );
}
