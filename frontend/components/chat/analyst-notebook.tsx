"use client";

import {
  ChevronDown,
  ChevronRight,
  FileBarChart,
  CheckCircle,
  AlertTriangle,
  AlertCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { AnalystNotebookProps, DataQuality } from "@/types";
import { MetricCard } from "./metric-card";
import { ComparisonChart } from "./comparison-chart";
import { KeyFindings } from "./key-findings";

export function AnalystNotebook({
  notebook,
  onCitationClick,
  isExpanded,
  onToggle,
}: AnalystNotebookProps) {
  if (!notebook) {
    return null;
  }

  const hasMetrics = notebook.metrics && notebook.metrics.length > 0;
  const hasComparisons = notebook.comparisons && notebook.comparisons.length > 0;
  const hasFindings = notebook.key_findings && notebook.key_findings.length > 0;

  if (!hasMetrics && !hasComparisons && !hasFindings) {
    return null;
  }

  return (
    <div className="border rounded-lg overflow-hidden bg-card shadow-sm mb-4">
      {/* Header */}
      <button
        onClick={onToggle}
        className={cn(
          "w-full flex items-center justify-between px-4 py-3",
          "hover:bg-muted/50 transition-colors",
          "text-left"
        )}
      >
        <div className="flex items-center gap-3">
          {isExpanded ? (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronRight className="h-4 w-4 text-muted-foreground" />
          )}

          <FileBarChart className="h-5 w-5 text-blue-600 dark:text-blue-400" />

          <span className="text-sm font-semibold">Analyst's Notebook</span>

          <DataQualityBadge quality={notebook.data_quality} />
        </div>
      </button>

      {/* Expanded Content */}
      {isExpanded && (
        <div className="border-t bg-muted/10">
          <div className="px-4 py-4 space-y-4">
            {/* Context Info */}
            <div className="flex items-center gap-4 text-xs text-muted-foreground">
              {notebook.companies_analyzed.length > 0 && (
                <div className="flex items-center gap-2">
                  <span className="font-medium">Companies:</span>
                  <span>{notebook.companies_analyzed.join(', ')}</span>
                </div>
              )}
              {notebook.periods_covered.length > 0 && (
                <div className="flex items-center gap-2">
                  <span className="font-medium">Period:</span>
                  <span>{notebook.periods_covered.join(', ')}</span>
                </div>
              )}
            </div>

            {/* Metrics Grid */}
            {hasMetrics && (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {notebook.metrics.map((metric, index) => (
                  <MetricCard
                    key={`${metric.company}-${metric.metric_name}-${index}`}
                    metric={metric}
                    onCitationClick={onCitationClick}
                  />
                ))}
              </div>
            )}

            {/* Comparisons */}
            {hasComparisons && (
              <div className="space-y-4">
                {notebook.comparisons.map((comparison, index) => (
                  <ComparisonChart
                    key={`${comparison.metric_name}-${index}`}
                    comparison={comparison}
                    onCitationClick={onCitationClick}
                  />
                ))}
              </div>
            )}

            {/* Key Findings */}
            {hasFindings && (
              <KeyFindings findings={notebook.key_findings} />
            )}
          </div>
        </div>
      )}
    </div>
  );
}

interface DataQualityBadgeProps {
  quality: DataQuality;
}

function DataQualityBadge({ quality }: DataQualityBadgeProps) {
  const config = {
    high: {
      label: 'High Quality',
      icon: CheckCircle,
      className: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
    },
    medium: {
      label: 'Medium Quality',
      icon: AlertTriangle,
      className: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400',
    },
    low: {
      label: 'Low Quality',
      icon: AlertCircle,
      className: 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400',
    },
  };

  const { label, icon: Icon, className } = config[quality] || config.medium;

  return (
    <div
      className={cn(
        "flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium",
        className
      )}
    >
      <Icon className="h-3 w-3" />
      <span>{label}</span>
    </div>
  );
}
