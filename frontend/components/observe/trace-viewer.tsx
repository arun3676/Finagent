"use client";

import { useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  Clock,
  CheckCircle,
  XCircle,
  Loader2,
  Search,
  Brain,
  Calculator,
  FileText,
  Wand2,
  Shield,
  Wrench,
  MessageSquare,
} from "lucide-react";
import { TraceEvent, TraceEventType } from "@/types";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

interface TraceViewerProps {
  events: TraceEvent[];
  isLoading?: boolean;
}

const EVENT_ICONS: Record<TraceEventType, React.ElementType> = {
  router: Search,
  planner: Brain,
  retriever: FileText,
  analyst: Calculator,
  synthesizer: Wand2,
  validator: Shield,
  tool_call: Wrench,
  llm_call: MessageSquare,
  error: XCircle,
};

const EVENT_COLORS: Record<TraceEventType, string> = {
  router: "text-purple-500 bg-purple-500/10",
  planner: "text-blue-500 bg-blue-500/10",
  retriever: "text-green-500 bg-green-500/10",
  analyst: "text-orange-500 bg-orange-500/10",
  synthesizer: "text-pink-500 bg-pink-500/10",
  validator: "text-cyan-500 bg-cyan-500/10",
  tool_call: "text-yellow-500 bg-yellow-500/10",
  llm_call: "text-indigo-500 bg-indigo-500/10",
  error: "text-red-500 bg-red-500/10",
};

export function TraceViewer({ events, isLoading }: TraceViewerProps) {
  if (events.length === 0 && !isLoading) {
    return (
      <div className="text-center text-muted-foreground py-8">
        No trace events to display
      </div>
    );
  }

  return (
    <ScrollArea className="h-[500px]">
      <div className="space-y-2 p-4">
        {events.map((event, index) => (
          <TraceEventItem
            key={event.id}
            event={event}
            depth={0}
            isLast={index === events.length - 1}
          />
        ))}
        {isLoading && (
          <div className="flex items-center gap-2 text-muted-foreground pl-4">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span className="text-sm">Processing...</span>
          </div>
        )}
      </div>
    </ScrollArea>
  );
}

interface TraceEventItemProps {
  event: TraceEvent;
  depth: number;
  isLast: boolean;
}

function TraceEventItem({ event, depth, isLast }: TraceEventItemProps) {
  const [isExpanded, setIsExpanded] = useState(depth < 2);
  
  const Icon = EVENT_ICONS[event.type] || Wrench;
  const colorClass = EVENT_COLORS[event.type] || "text-gray-500 bg-gray-500/10";
  const hasChildren = event.children && event.children.length > 0;
  const hasDetails = event.input || event.output || event.error;

  const statusIcon: Record<string, JSX.Element> = {
    pending: <Clock className="h-3 w-3 text-muted-foreground" />,
    running: <Loader2 className="h-3 w-3 text-blue-500 animate-spin" />,
    completed: <CheckCircle className="h-3 w-3 text-green-500" />,
    failed: <XCircle className="h-3 w-3 text-red-500" />,
  };
  const statusElement = statusIcon[event.status];

  return (
    <div className={cn("relative", depth > 0 && "ml-6")}>
      {depth > 0 && (
        <div
          className={cn(
            "absolute left-[-20px] top-0 w-[20px] border-l-2 border-b-2 border-muted rounded-bl",
            isLast ? "h-[20px]" : "h-full"
          )}
        />
      )}

      <div
        className={cn(
          "border rounded-lg overflow-hidden",
          event.status === "failed" && "border-red-500/50"
        )}
      >
        <button
          className="w-full flex items-center gap-3 p-3 hover:bg-muted/50 transition-colors text-left"
          onClick={() => setIsExpanded(!isExpanded)}
        >
          {(hasChildren || hasDetails) && (
            <span className="text-muted-foreground">
              {isExpanded ? (
                <ChevronDown className="h-4 w-4" />
              ) : (
                <ChevronRight className="h-4 w-4" />
              )}
            </span>
          )}
          
          <div className={cn("p-1.5 rounded", colorClass)}>
            <Icon className="h-4 w-4" />
          </div>

          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="font-medium text-sm truncate">{event.name}</span>
              <Badge variant="outline" className="text-xs capitalize">
                {event.type.replace("_", " ")}
              </Badge>
            </div>
          </div>

          <div className="flex items-center gap-3 text-xs text-muted-foreground">
            {event.metadata?.tokens_used && (
              <span>{event.metadata.tokens_used} tokens</span>
            )}
            {event.metadata?.chunks_retrieved && (
              <span>{event.metadata.chunks_retrieved} chunks</span>
            )}
            <span>{event.duration_ms}ms</span>
            {statusElement}
          </div>
        </button>

        {isExpanded && hasDetails && (
          <div className="border-t p-3 space-y-3 bg-muted/30">
            {event.input && (
              <div>
                <p className="text-xs font-medium text-muted-foreground mb-1">Input</p>
                <pre className="text-xs bg-background p-2 rounded overflow-x-auto">
                  {JSON.stringify(event.input, null, 2)}
                </pre>
              </div>
            )}
            {event.output && (
              <div>
                <p className="text-xs font-medium text-muted-foreground mb-1">Output</p>
                <pre className="text-xs bg-background p-2 rounded overflow-x-auto max-h-32 overflow-y-auto">
                  {typeof event.output === "string"
                    ? event.output
                    : JSON.stringify(event.output, null, 2)}
                </pre>
              </div>
            )}
            {event.error && (
              <div>
                <p className="text-xs font-medium text-red-500 mb-1">Error</p>
                <pre className="text-xs bg-red-500/10 text-red-600 p-2 rounded">
                  {event.error}
                </pre>
              </div>
            )}
          </div>
        )}

        {isExpanded && hasChildren && (
          <div className="border-t p-3 pl-6 space-y-2">
            {event.children!.map((child: TraceEvent, idx: number) => (
              <TraceEventItem
                key={child.id}
                event={child}
                depth={depth + 1}
                isLast={idx === event.children!.length - 1}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
