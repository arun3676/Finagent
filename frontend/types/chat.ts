export interface CitationSourceMetadata {
  ticker: string;
  company_name: string;
  document_type: string;
  filing_date: string;
  section: string;
  page_number: number;
  source_url: string;
}

export interface Citation {
  citation_id: string;
  citation_number: number;
  claim: string;

  // Source identification
  source_chunk_id: string;
  source_document_id?: string;

  // Source content
  source_text: string;
  source_context: string;  // 2-3 sentences before/after for context
  highlight_start: number; // Character position where relevant text starts
  highlight_end: number;   // Character position where relevant text ends

  // Source metadata for display
  source_metadata: Partial<CitationSourceMetadata>;

  // Confidence and validation
  confidence: number;
  validation_method: 'exact_match' | 'semantic_similarity' | 'llm_verified';

  // For UI display
  preview_text: string;    // Short preview for tooltip (50 chars)

  // Legacy fields (backward compatibility)
  page_reference?: string;
  source_url?: string;
  metadata?: {
    ticker?: string;
    company_name?: string;
    filing_type?: string;
    document_type?: string;
    filing_date?: string;
    section?: string;
    page?: number;
    page_number?: number;
    source_url?: string;
  };
}

export type AgentStepStatus = "pending" | "running" | "completed" | "failed";

/**
 * Legacy AgentStep format used in message agent_steps.
 * For real-time stepper visualization, use AgentStep from types/stepper.ts
 */
export interface MessageAgentStep {
  step_name: string;
  description: string;
  status: AgentStepStatus;
  duration_ms: number;
  output?: string;
}

export interface MessageMetadata {
  query_complexity?: "simple" | "moderate" | "complex";
  total_duration_ms?: number;
  query_time_ms?: number;
  model_used?: string;
  tokens_used?: number;
  confidence?: number;
  sources_consulted?: number;
}

export interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: Date;
  citations?: Citation[];
  agent_steps?: MessageAgentStep[];
  analyst_notebook?: import('./analyst').AnalystNotebook;
  validation?: import('./validation').ValidationResult;
  metadata?: MessageMetadata;
  is_streaming?: boolean;
  error?: string;
}

export interface ComplexityInfo {
  level: "SIMPLE" | "MODERATE" | "COMPLEX";
  display_label: string;
  display_color: string;
  estimated_time_seconds: number;
  reasoning: string;
  features_enabled: string[];
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  created_at: Date;
  updated_at: Date;
  metadata?: {
    company_ticker?: string;
    filing_types?: string[];
  };
}
