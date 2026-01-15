// Agent Stepper Types

export type AgentName =
  | 'router'
  | 'planner'
  | 'retriever'
  | 'analyst'
  | 'synthesizer'
  | 'validator';

export type StepStatus = 'pending' | 'active' | 'completed' | 'error';

export interface AgentStep {
  agent: AgentName;
  displayName: string;
  status: StepStatus;
  detail?: string;           // "Classified as COMPLEX"
  subItems?: string[];       // Sub-queries, retrieved docs, insights
  duration?: number;         // ms
  startTime?: number;        // timestamp when step started
}

export interface AgentStepperProps {
  steps: AgentStep[];
  isExpanded: boolean;
  onToggleExpand: () => void;
  complexity?: 'SIMPLE' | 'MODERATE' | 'COMPLEX' | null;
}

// SSE Event Types for Stepper
export interface StepSSEEvent {
  type: 'step';
  step: AgentName;
  status: 'started' | 'completed' | 'error';
}

export interface StepDetailSSEEvent {
  type: 'step_detail';
  agent: AgentName;
  data: {
    complexity?: string;
    reasoning?: string;
    [key: string]: any;
  };
}

export interface SubQuerySSEEvent {
  type: 'sub_query';
  index: number;
  total: number;
  query: string;
  intent?: string;
}

export interface RetrievalProgressSSEEvent {
  type: 'retrieval_progress';
  status: string;
  source?: string;
  chunks_found?: number;
}

export interface AnalysisInsightSSEEvent {
  type: 'analysis_insight';
  metric?: string;
  value?: string;
  company?: string;
  [key: string]: any;
}

export type StepperSSEEvent =
  | StepSSEEvent
  | StepDetailSSEEvent
  | SubQuerySSEEvent
  | RetrievalProgressSSEEvent
  | AnalysisInsightSSEEvent;
