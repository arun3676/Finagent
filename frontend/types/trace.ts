export type TraceEventType = 
  | "router"
  | "planner" 
  | "retriever"
  | "analyst"
  | "synthesizer"
  | "validator"
  | "tool_call"
  | "llm_call"
  | "error";

export interface TraceEvent {
  id: string;
  type: TraceEventType;
  name: string;
  timestamp: Date;
  duration_ms: number;
  status: "pending" | "running" | "completed" | "failed";
  input?: Record<string, unknown>;
  output?: Record<string, unknown>;
  error?: string;
  children?: TraceEvent[];
  metadata?: {
    model?: string;
    tokens_used?: number;
    tool_name?: string;
    chunks_retrieved?: number;
  };
}

export interface AgentTrace {
  id: string;
  query: string;
  start_time: Date;
  end_time?: Date;
  total_duration_ms?: number;
  status: "running" | "completed" | "failed";
  events: TraceEvent[];
  final_response?: string;
  total_tokens?: number;
  total_chunks_retrieved?: number;
}
