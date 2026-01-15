export interface SourceDocument {
  id: string;
  ticker: string;
  document_type: "SEC_10K" | "SEC_10Q" | "SEC_8K" | "EARNINGS_CALL";
  filing_date: Date;
  fiscal_year?: number;
  fiscal_quarter?: number;
  title: string;
  url?: string;
}

export interface SourceChunk {
  chunk_id: string;
  document: SourceDocument;
  section: string;
  subsection?: string;
  page_number?: number;
  paragraph_number?: number;
  text: string;
  text_preview: string;
}

export interface EnhancedCitation {
  id: string;
  claim_text: string;
  source_chunk: SourceChunk;
  confidence_score: number;
  match_type: "exact" | "paraphrase" | "inference";
  highlight_ranges?: Array<{ start: number; end: number }>;
}

export interface CitedResponse {
  answer: string;
  citations: EnhancedCitation[];
  citation_coverage: number;
  uncited_claims?: string[];
}
