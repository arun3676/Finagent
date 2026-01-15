"use client";

import { cn } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

export interface ComplexityInfo {
  level: "SIMPLE" | "MODERATE" | "COMPLEX";
  display_label: string;
  display_color: string;
  estimated_time_seconds: number;
  reasoning: string;
  features_enabled: string[];
}

interface ComplexityBadgeProps {
  complexity: ComplexityInfo | null;
  isLoading: boolean;
}

const colorClasses = {
  green: "bg-green-100 text-green-800 border-green-300",
  blue: "bg-blue-100 text-blue-800 border-blue-300",
  purple: "bg-purple-100 text-purple-800 border-purple-300",
  gray: "bg-gray-100 text-gray-600 border-gray-300",
};

export function ComplexityBadge({
  complexity,
  isLoading,
}: ComplexityBadgeProps) {
  if (!isLoading && !complexity) {
    return null;
  }

  const isAnalyzing = isLoading && !complexity;
  const displayLabel = isAnalyzing ? "Analyzing..." : complexity?.display_label;
  const colorClass = isAnalyzing
    ? colorClasses.gray
    : colorClasses[complexity?.display_color as keyof typeof colorClasses] ||
      colorClasses.gray;

  const badge = (
    <div
      className={cn(
        "inline-flex items-center gap-1.5 px-3 py-1 rounded-full border text-xs font-medium",
        "transition-all duration-300",
        colorClass,
        isAnalyzing && "animate-pulse"
      )}
    >
      <span className="relative flex h-2 w-2">
        {isAnalyzing && (
          <>
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-current opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-current"></span>
          </>
        )}
        {!isAnalyzing && (
          <span className="relative inline-flex rounded-full h-2 w-2 bg-current"></span>
        )}
      </span>
      <span>{displayLabel}</span>
    </div>
  );

  if (isAnalyzing || !complexity) {
    return badge;
  }

  return (
    <TooltipProvider delayDuration={200}>
      <Tooltip>
        <TooltipTrigger asChild>{badge}</TooltipTrigger>
        <TooltipContent side="top" className="max-w-xs">
          <div className="space-y-1.5">
            <div className="flex items-start gap-2">
              <span className="text-muted-foreground text-xs">Reasoning:</span>
              <span className="text-xs font-medium">{complexity.reasoning}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-muted-foreground text-xs">Estimated time:</span>
              <span className="text-xs font-medium">
                ~{complexity.estimated_time_seconds}s
              </span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-muted-foreground text-xs">Agents:</span>
              <span className="text-xs font-medium">
                {complexity.features_enabled.join(", ")}
              </span>
            </div>
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
