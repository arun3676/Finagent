"use client";

import { useState } from "react";
import { Activity, Clock, Zap, CheckCircle } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { TraceViewer } from "@/components/observe/trace-viewer";
import { TraceSummary } from "@/components/observe/trace-summary";
import { MetricsCard } from "@/components/observe/metrics-card";
import { AgentTrace } from "@/types";

const MOCK_TRACE: AgentTrace = {
  id: "trace-1",
  query: "Compare Apple and Microsoft gross margins for 2023",
  start_time: new Date(),
  end_time: new Date(),
  total_duration_ms: 8420,
  status: "completed",
  total_tokens: 4521,
  total_chunks_retrieved: 12,
  events: [
    {
      id: "1",
      type: "router",
      name: "Query Classification",
      timestamp: new Date(),
      duration_ms: 245,
      status: "completed",
      input: { query: "Compare Apple and Microsoft gross margins for 2023" },
      output: { complexity: "COMPLEX", requires_comparison: true },
    },
    {
      id: "2",
      type: "planner",
      name: "Query Decomposition",
      timestamp: new Date(),
      duration_ms: 890,
      status: "completed",
      input: { query: "Compare Apple and Microsoft gross margins for 2023" },
      output: {
        steps: [
          "Retrieve AAPL 10-K for gross margin data",
          "Retrieve MSFT 10-K for gross margin data",
          "Calculate and compare margins",
        ],
      },
      metadata: { tokens_used: 342 },
    },
    {
      id: "3",
      type: "retriever",
      name: "Parallel Document Retrieval",
      timestamp: new Date(),
      duration_ms: 1240,
      status: "completed",
      metadata: { chunks_retrieved: 12 },
      children: [
        {
          id: "3a",
          type: "tool_call",
          name: "Search AAPL 10-K",
          timestamp: new Date(),
          duration_ms: 580,
          status: "completed",
          output: { chunks: 6 },
        },
        {
          id: "3b",
          type: "tool_call",
          name: "Search MSFT 10-K",
          timestamp: new Date(),
          duration_ms: 620,
          status: "completed",
          output: { chunks: 6 },
        },
      ],
    },
    {
      id: "4",
      type: "analyst",
      name: "Financial Analysis",
      timestamp: new Date(),
      duration_ms: 1890,
      status: "completed",
      input: { task: "Extract and calculate gross margins" },
      output: {
        aapl_margin: "44.1%",
        msft_margin: "69.4%",
      },
      metadata: { tokens_used: 1256 },
    },
    {
      id: "5",
      type: "synthesizer",
      name: "Response Generation",
      timestamp: new Date(),
      duration_ms: 2100,
      status: "completed",
      metadata: { tokens_used: 2123 },
    },
    {
      id: "6",
      type: "validator",
      name: "Citation Verification",
      timestamp: new Date(),
      duration_ms: 780,
      status: "completed",
      output: { citations_verified: 4, faithfulness_score: 0.96 },
    },
  ],
};

export default function ObservePage() {
  const [selectedTrace] = useState<AgentTrace>(MOCK_TRACE);

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-primary/10 rounded-lg">
          <Activity className="h-6 w-6 text-primary" />
        </div>
        <div>
          <h1 className="text-2xl font-bold">Observability</h1>
          <p className="text-muted-foreground">
            Monitor agent performance, traces, and metrics
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <MetricsCard
          title="Queries Today"
          value="127"
          change="+12%"
          icon={Activity}
          trend="up"
        />
        <MetricsCard
          title="Avg. Latency"
          value="2.4s"
          change="-8%"
          icon={Clock}
          trend="down"
        />
        <MetricsCard
          title="Tokens Used"
          value="45.2K"
          change="+5%"
          icon={Zap}
          trend="up"
        />
        <MetricsCard
          title="Success Rate"
          value="98.2%"
          change="+0.5%"
          icon={CheckCircle}
          trend="up"
        />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Latest Query Trace</CardTitle>
          <CardDescription>
            {selectedTrace.query}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <TraceSummary trace={selectedTrace} />
          <TraceViewer events={selectedTrace.events} />
        </CardContent>
      </Card>
    </div>
  );
}
