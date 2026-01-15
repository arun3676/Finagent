"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { TraceEvent } from "@/types";
import { TraceViewer } from "@/components/observe/trace-viewer";
import { Button } from "@/components/ui/button";

interface AgentStepsInlineProps {
  events: TraceEvent[];
  totalDuration?: number;
  isStreaming?: boolean;
}

export function AgentStepsInline({ 
  events, 
  totalDuration,
  isStreaming 
}: AgentStepsInlineProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (events.length === 0 && !isStreaming) return null;

  const completedCount = events.filter((e: TraceEvent) => e.status === "completed").length;
  const totalCount = events.length;

  return (
    <div className="border rounded-lg overflow-hidden bg-muted/20">
      <Button
        variant="ghost"
        className="w-full justify-between h-auto py-2 px-3"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          {isExpanded ? (
            <ChevronDown className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
          <span className="text-sm font-medium">
            Agent Reasoning
          </span>
          <span className="text-xs text-muted-foreground">
            ({completedCount}/{totalCount} steps)
          </span>
        </div>
        {totalDuration && (
          <span className="text-xs text-muted-foreground">
            {(totalDuration / 1000).toFixed(2)}s
          </span>
        )}
      </Button>

      {isExpanded && (
        <div className="border-t">
          <TraceViewer events={events} isLoading={isStreaming} />
        </div>
      )}
    </div>
  );
}
