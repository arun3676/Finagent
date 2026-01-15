"use client";

import { AgentTrace, TraceEvent } from "@/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Clock, Zap, FileText, CheckCircle, XCircle } from "lucide-react";

interface TraceSummaryProps {
  trace: AgentTrace;
}

export function TraceSummary({ trace }: TraceSummaryProps) {
  const completedEvents = trace.events.filter((e: TraceEvent) => e.status === "completed").length;
  const failedEvents = trace.events.filter((e: TraceEvent) => e.status === "failed").length;

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
            <Clock className="h-4 w-4" />
            Duration
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-2xl font-bold">
            {trace.total_duration_ms 
              ? `${(trace.total_duration_ms / 1000).toFixed(2)}s` 
              : "—"
            }
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
            <Zap className="h-4 w-4" />
            Tokens Used
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-2xl font-bold">
            {trace.total_tokens?.toLocaleString() || "—"}
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
            <FileText className="h-4 w-4" />
            Chunks Retrieved
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-2xl font-bold">
            {trace.total_chunks_retrieved || "—"}
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
            Steps
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1 text-green-500">
              <CheckCircle className="h-4 w-4" />
              <span className="font-bold">{completedEvents}</span>
            </div>
            {failedEvents > 0 && (
              <div className="flex items-center gap-1 text-red-500">
                <XCircle className="h-4 w-4" />
                <span className="font-bold">{failedEvents}</span>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
