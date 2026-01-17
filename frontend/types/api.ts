// API Response Types

export type ResponseLength = "short" | "normal" | "detailed";

export interface HealthResponse {
  status: "healthy" | "degraded" | "unhealthy";
  version: string;
  components: {
    qdrant: "up" | "down";
    llm: "up" | "down";
    embeddings: "up" | "down";
  };
  timestamp: string;
}

export interface QueryRequest {
  query: string;
  response_length?: ResponseLength;
  filters?: {
    tickers?: string[];
    filing_types?: string[];
    date_range?: {
      start?: string;
      end?: string;
    };
  };
  options?: {
    include_reasoning?: boolean;
    include_citations?: boolean;
    max_sources?: number;
  };
}

export interface QueryCitation {
  citation_id: string;
  claim: string;
  source_chunk_id: string;
  source_text: string;
  confidence: number;
  page_reference?: string;
  metadata?: {
    ticker?: string;
    filing_type?: string;
    section?: string;
    page?: number;
  };
}

export interface QueryResponse {
  answer: string;
  citations: QueryCitation[];
  reasoning?: string;
  analyst_notebook?: import('./analyst').AnalystNotebook;
  query_id?: string;
  metadata: {
    query_time_ms: number;
    model_used?: string;
    sources_consulted: number;
    confidence?: number;
  };
}

export interface IngestRequest {
  ticker: string;
  filing_types?: string[];
  years?: number[];
}

export interface IngestResponse {
  job_id: string;
  status: "queued" | "processing" | "completed" | "failed";
  ticker: string;
  message: string;
}

export interface IngestProgress {
  job_id: string;
  ticker: string;
  status: "fetching" | "parsing" | "chunking" | "embedding" | "storing" | "completed" | "failed";
  progress_percent: number;
  current_step: string;
  documents_processed: number;
  documents_total: number;
  chunks_created: number;
  error_message?: string;
}

// Generic API result wrapper
export interface ApiResult<T> {
  success: boolean;
  data?: T;
  error?: {
    message: string;
    code?: string;
    details?: unknown;
  };
}

// ========== SSE Event Types ==========

export type SSEEventType =
  | 'step'
  | 'step_detail'
  | 'complexity'
  | 'sub_query'
  | 'retrieval_progress'
  | 'analysis_insight'
  | 'token'
  | 'citations'
  | 'analyst_notebook'
  | 'validation'
  | 'done'
  | 'error';

export interface BaseSSEEvent {
  type: SSEEventType;
}

export interface TokenSSEEvent extends BaseSSEEvent {
  type: 'token';
  content: string;
}

export interface ComplexitySSEEvent extends BaseSSEEvent {
  type: 'complexity';
  data: import('./chat').ComplexityInfo;
}

export interface StepSSEEvent extends BaseSSEEvent {
  type: 'step';
  step: string;
  status: 'started' | 'completed' | 'error';
}

export interface StepDetailSSEEvent extends BaseSSEEvent {
  type: 'step_detail';
  agent: string;
  data: {
    complexity?: string;
    reasoning?: string;
    [key: string]: unknown;
  };
}

export interface SubQuerySSEEvent extends BaseSSEEvent {
  type: 'sub_query';
  index: number;
  total: number;
  query: string;
  intent?: string;
}

export interface RetrievalProgressSSEEvent extends BaseSSEEvent {
  type: 'retrieval_progress';
  status: string;
  source?: string;
  chunks_found?: number;
}

export interface AnalysisInsightSSEEvent extends BaseSSEEvent {
  type: 'analysis_insight';
  metric?: string;
  value?: string;
  company?: string;
  [key: string]: unknown;
}

export interface CitationsSSEEvent extends BaseSSEEvent {
  type: 'citations';
  citations: import('./chat').Citation[];
}

export interface AnalystNotebookSSEEvent extends BaseSSEEvent {
  type: 'analyst_notebook';
  notebook?: import('./analyst').AnalystNotebook;
  data?: import('./analyst').AnalystNotebook; // Backend sends 'data' key
}

export interface ValidationSSEEvent extends BaseSSEEvent {
  type: 'validation';
  validation?: import('./validation').ValidationResult;
  data?: import('./validation').ValidationResult; // Backend sends 'data' key
}

export interface DoneSSEEvent extends BaseSSEEvent {
  type: 'done';
  metadata?: {
    query_time_ms: number;
    model_used: string;
    sources_consulted: number;
  };
}

export interface ErrorSSEEvent extends BaseSSEEvent {
  type: 'error';
  message: string;
  code?: string;
}

export type SSEEvent =
  | TokenSSEEvent
  | ComplexitySSEEvent
  | StepSSEEvent
  | StepDetailSSEEvent
  | SubQuerySSEEvent
  | RetrievalProgressSSEEvent
  | AnalysisInsightSSEEvent
  | CitationsSSEEvent
  | AnalystNotebookSSEEvent
  | ValidationSSEEvent
  | DoneSSEEvent
  | ErrorSSEEvent;
