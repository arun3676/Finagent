"use client";

import { Citation } from "@/types/chat";
import { cn } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Badge } from "@/components/ui/badge";

interface CitationMarkerProps {
  number: number;
  citation: Citation;
  isActive: boolean;
  onClick: () => void;
}

export function CitationMarker({
  number,
  citation,
  isActive,
  onClick,
}: CitationMarkerProps) {
  const confidenceColor =
    citation.confidence >= 0.8
      ? "text-green-600"
      : citation.confidence >= 0.6
      ? "text-yellow-600"
      : "text-red-600";

  const confidencePercent = Math.round(citation.confidence * 100);

  return (
    <TooltipProvider delayDuration={200}>
      <Tooltip>
        <TooltipTrigger asChild>
          <sup
            className={cn(
              "inline-flex items-center cursor-pointer mx-0.5",
              "text-xs font-medium transition-all duration-200",
              "hover:scale-110",
              isActive
                ? "bg-primary/20 text-primary px-1 rounded"
                : "text-primary hover:text-primary/80"
            )}
            onClick={onClick}
          >
            [{number}]
          </sup>
        </TooltipTrigger>
        <TooltipContent side="top" className="max-w-xs">
          <div className="space-y-2">
            <p className="text-xs">
              {citation.preview_text || citation.source_text.slice(0, 100) + "..."}
            </p>
            <div className="flex items-center gap-2">
              <Badge
                variant="outline"
                className={cn("text-xs", confidenceColor)}
              >
                {confidencePercent}% confidence
              </Badge>
            </div>
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
