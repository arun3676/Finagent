"use client";

import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import { cn } from "@/lib/utils";
import { MetricCardProps, ChangeDirection } from "@/types";

export function MetricCard({ metric, onCitationClick }: MetricCardProps) {
  const hasChange = metric.change_percent !== undefined && metric.change_direction;

  return (
    <div className="border rounded-lg p-4 bg-card hover:shadow-md transition-shadow">
      {/* Company Ticker */}
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-semibold text-muted-foreground tracking-wide">
          {metric.company}
        </span>
        {metric.source_citation_id && onCitationClick && (
          <button
            onClick={() => onCitationClick(metric.source_citation_id!)}
            className="text-xs text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 font-medium"
          >
            [{metric.source_citation_id.replace('cit_', '')}]
          </button>
        )}
      </div>

      {/* Metric Name */}
      <div className="text-sm text-muted-foreground mb-1">
        {metric.display_name}
      </div>

      {/* Value and Change Indicator */}
      <div className="flex items-center gap-2 mb-2">
        <span className="text-2xl font-bold">
          {metric.formatted_value}
        </span>
        {hasChange && (
          <ChangeIndicator
            direction={metric.change_direction!}
            changePercent={metric.change_percent!}
          />
        )}
      </div>

      {/* Comparison Label */}
      {metric.comparison_label && metric.change_percent !== undefined && (
        <div className="text-xs text-muted-foreground">
          {metric.comparison_label}: {metric.change_percent > 0 ? '+' : ''}
          {metric.change_percent.toFixed(1)}%
        </div>
      )}

      {/* Period and Section */}
      <div className="flex items-center gap-2 mt-2 text-xs text-muted-foreground">
        {metric.fiscal_period && (
          <span className="px-2 py-0.5 bg-muted rounded">
            {metric.fiscal_period}
          </span>
        )}
        {metric.source_section && (
          <span className="truncate" title={metric.source_section}>
            {metric.source_section}
          </span>
        )}
      </div>
    </div>
  );
}

interface ChangeIndicatorProps {
  direction: ChangeDirection;
  changePercent: number;
}

function ChangeIndicator({ direction, changePercent }: ChangeIndicatorProps) {
  const isPositive = direction === 'up';
  const isNegative = direction === 'down';
  const isFlat = direction === 'flat';

  return (
    <div
      className={cn(
        "flex items-center gap-1 text-sm font-medium",
        isPositive && "text-green-600 dark:text-green-400",
        isNegative && "text-red-600 dark:text-red-400",
        isFlat && "text-gray-500 dark:text-gray-400"
      )}
    >
      {isPositive && <TrendingUp className="h-4 w-4" />}
      {isNegative && <TrendingDown className="h-4 w-4" />}
      {isFlat && <Minus className="h-4 w-4" />}
      <span className="text-xs">
        {changePercent > 0 ? '+' : ''}{changePercent.toFixed(1)}%
      </span>
    </div>
  );
}
