"use client";

import { TrendingUp, Trophy } from "lucide-react";
import { cn } from "@/lib/utils";
import { ComparisonChartProps } from "@/types";

// Color mapping for companies (consistent across charts)
const COMPANY_COLORS: Record<string, string> = {
  AAPL: 'bg-blue-500',
  MSFT: 'bg-green-500',
  GOOGL: 'bg-red-500',
  AMZN: 'bg-orange-500',
  META: 'bg-indigo-500',
  TSLA: 'bg-purple-500',
  NVDA: 'bg-emerald-500',
  // Add more as needed
};

const DEFAULT_COLOR = 'bg-gray-500';

function getCompanyColor(company: string): string {
  return COMPANY_COLORS[company] || DEFAULT_COLOR;
}

export function ComparisonChart({ comparison, onCitationClick }: ComparisonChartProps) {
  if (!comparison || !comparison.items || comparison.items.length === 0) {
    return null;
  }

  // Find the maximum value for scaling the bars
  const maxValue = Math.max(...comparison.items.map((item) => item.value));

  // Sort items by value (descending)
  const sortedItems = [...comparison.items].sort((a, b) => b.value - a.value);

  return (
    <div className="border rounded-lg p-4 bg-card">
      {/* Header */}
      <div className="flex items-center gap-2 mb-4">
        <TrendingUp className="h-5 w-5 text-blue-600 dark:text-blue-400" />
        <h3 className="text-sm font-semibold">
          Comparison: {comparison.display_name}
        </h3>
      </div>

      {/* Bar Chart */}
      <div className="space-y-3">
        {sortedItems.map((item, index) => {
          const percentage = (item.value / maxValue) * 100;
          const isWinner = comparison.winner === item.company;

          return (
            <div
              key={item.company}
              className="animate-in slide-in-from-left-2"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <div className="flex items-center gap-3 mb-1">
                <span className="text-xs font-semibold text-muted-foreground w-12">
                  {item.company}
                </span>
                {isWinner && (
                  <Trophy className="h-4 w-4 text-yellow-500" />
                )}
              </div>

              <div className="flex items-center gap-3">
                {/* Bar */}
                <div className="flex-1 bg-muted rounded-full h-6 overflow-hidden">
                  <div
                    className={cn(
                      "h-full rounded-full transition-all duration-700 ease-out flex items-center justify-end pr-2",
                      getCompanyColor(item.company),
                      isWinner && "ring-2 ring-yellow-400 ring-offset-1"
                    )}
                    style={{ width: `${percentage}%` }}
                  >
                    <span className="text-xs font-semibold text-white drop-shadow">
                      {item.formatted_value}
                    </span>
                  </div>
                </div>

                {/* Citation Link */}
                {item.citation_id && onCitationClick && (
                  <button
                    onClick={() => onCitationClick(item.citation_id!)}
                    className="text-xs text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 font-medium flex-shrink-0"
                  >
                    [{item.citation_id.replace('cit_', '')}]
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Insight */}
      {comparison.insight && (
        <div className="mt-4 pt-3 border-t text-sm text-muted-foreground italic">
          {comparison.insight}
        </div>
      )}
    </div>
  );
}
