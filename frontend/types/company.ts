export type DocumentType = "SEC_10K" | "SEC_10Q" | "SEC_8K" | "EARNINGS_CALL";

export type DocumentStatus = "pending" | "processing" | "ready" | "failed";

export type IngestStatus =
  | "queued"
  | "processing"
  | "fetching"
  | "parsing"
  | "chunking"
  | "embedding"
  | "storing"
  | "indexing"
  | "completed"
  | "failed";

export interface IngestProgress {
  ticker: string;
  status: IngestStatus;
  progress_percent: number;
  current_step: string;
  documents_processed: number;
  documents_total: number;
  chunks_created: number;
  error_message?: string;
}

export interface IndexedDocument {
  id: string;
  document_type: DocumentType;
  fiscal_year?: number;
  fiscal_quarter?: number;
  chunk_count: number;
  status: DocumentStatus;
  indexed_at: string;
  metadata?: Record<string, unknown>;
}

export interface Company {
  ticker: string;
  name: string;
  sector?: string;
  industry?: string;
  indexed_documents: IndexedDocument[];
  last_updated: string;
  total_documents: number;
  total_chunks: number;
}
