"use client";

import { useState } from "react";
import { Citation } from "@/types/chat";
import { ResponseWithCitations } from "./response-with-citations";
import { CitationPanel } from "./citation-panel";
import { useCitationExplorer } from "@/hooks/useCitationExplorer";

/**
 * Demo component showing how to use the Citation Explorer
 * 
 * Usage in your component:
 * ```tsx
 * <ResponseWithCitations
 *   answer={message.content}
 *   citations={message.citations}
 *   onCitationClick={openCitation}
 *   activeCitationId={activeCitation?.citation_id || null}
 * />
 * 
 * <CitationPanel
 *   citation={activeCitation}
 *   isOpen={isPanelOpen}
 *   onClose={closeCitation}
 * />
 * ```
 */
export function CitationExplorerDemo() {
  // Example data matching backend structure
  const exampleAnswer = "Apple's gross margin was 43.5% in FY2023[1], while Microsoft reported strong cloud growth[2].";
  
  const exampleCitations: Citation[] = [
    {
      citation_id: "cit_001",
      citation_number: 1,
      claim: "Apple's gross margin was 43.5% in FY2023",
      source_text: "For fiscal 2023, our gross margin was 43.5 percent...",
      source_context: "...compared to 43.3 percent in fiscal 2022. For fiscal 2023, our gross margin was 43.5 percent, compared to 43.3 percent in fiscal 2022. The year-over-year...",
      highlight_start: 52,
      highlight_end: 98,
      confidence: 0.95,
      validation_method: "exact_match",
      preview_text: "For fiscal 2023, our gross margin was 43.5 per...",
      source_chunk_id: "chunk_001",
      source_metadata: {
        ticker: "AAPL",
        company_name: "Apple Inc.",
        document_type: "10-K",
        filing_date: "2023-11-03",
        section: "Item 7 - MD&A",
        page_number: 24,
        source_url: "https://sec.gov/...",
      },
      metadata: {
        ticker: "AAPL",
        company_name: "Apple Inc.",
        document_type: "10-K",
        filing_date: "2023-11-03",
        section: "Item 7 - MD&A",
        page_number: 24,
        source_url: "https://sec.gov/...",
      },
    },
    {
      citation_id: "cit_002",
      citation_number: 2,
      claim: "Microsoft reported strong cloud growth",
      source_text: "Intelligent Cloud revenue increased 19% driven by Azure...",
      source_context: "In fiscal 2023, Intelligent Cloud revenue increased 19% driven by Azure and other cloud services growth of 27%. Our commercial cloud business...",
      highlight_start: 17,
      highlight_end: 80,
      confidence: 0.88,
      validation_method: "semantic_similarity",
      preview_text: "Intelligent Cloud revenue increased 19% driven...",
      source_chunk_id: "chunk_002",
      source_metadata: {
        ticker: "MSFT",
        company_name: "Microsoft Corporation",
        document_type: "10-K",
        filing_date: "2023-07-27",
        section: "Item 7 - Management's Discussion",
        page_number: 32,
        source_url: "https://sec.gov/...",
      },
      metadata: {
        ticker: "MSFT",
        company_name: "Microsoft Corporation",
        document_type: "10-K",
        filing_date: "2023-07-27",
        section: "Item 7 - Management's Discussion",
        page_number: 32,
        source_url: "https://sec.gov/...",
      },
    },
  ];

  const { activeCitation, isPanelOpen, openCitation, closeCitation } =
    useCitationExplorer(exampleCitations);

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-4">
      <div className="border rounded-lg p-4 bg-card">
        <h3 className="font-semibold mb-4">Citation Explorer Demo</h3>
        <ResponseWithCitations
          answer={exampleAnswer}
          citations={exampleCitations}
          onCitationClick={openCitation}
          activeCitationId={activeCitation?.citation_id || null}
        />
      </div>

      <div className="text-sm text-muted-foreground space-y-1">
        <p>• Hover over [1] or [2] to see preview tooltip</p>
        <p>• Click on [1] or [2] to open the citation panel</p>
        <p>• Panel shows highlighted source text and metadata</p>
      </div>

      <CitationPanel
        citation={activeCitation}
        isOpen={isPanelOpen}
        onClose={closeCitation}
      />
    </div>
  );
}
