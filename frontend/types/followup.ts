import type { Citation } from "./chat";

export type FollowUpCategory = "temporal" | "deeper" | "comparative" | "related";

export interface FollowUpQuestion {
  id: string;
  text: string;
  category: FollowUpCategory;
  relevantChunkIds: string[];
  requiresNewRetrieval: boolean;
}

export interface QueryResponseWithFollowUps {
  answer: string;
  citations: Citation[];
  reasoningTrace: string[];
  followUpQuestions: FollowUpQuestion[];
  queryId: string;
}

export interface FollowUpResponse {
  question: string;
  answer: string;
  citations: Citation[];
  executionTimeMs: number;
  error?: boolean;
}
