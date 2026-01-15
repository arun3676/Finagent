"use client";

import { FileText } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { EnhancedCitationBadge } from "@/components/chat/enhanced-citation-badge";
import { CitationSummary } from "@/components/chat/citation-summary";
import { SourcesPanel } from "@/components/chat/sources-panel";
import { EnhancedCitation } from "@/types";

const MOCK_CITATIONS: EnhancedCitation[] = [
  {
    id: "cite-1",
    claim_text: "Apple's gross margin for fiscal year 2023 was 44.1%",
    confidence_score: 0.95,
    match_type: "exact",
    source_chunk: {
      chunk_id: "aapl-10k-2023-chunk-42",
      document: {
        id: "aapl-10k-2023",
        ticker: "AAPL",
        document_type: "SEC_10K",
        filing_date: new Date("2023-11-03"),
        fiscal_year: 2023,
        title: "Apple Inc. Annual Report (Form 10-K)",
        url: "https://www.sec.gov/cgi-bin/viewer?action=view&cik=320193&accession_number=0000320193-23-000106",
      },
      section: "Item 6. Selected Financial Data",
      subsection: "Consolidated Statements of Operations",
      page_number: 28,
      text: "For the fiscal year ended September 30, 2023, the Company reported total net sales of $383.3 billion and cost of sales of $214.1 billion, resulting in a gross margin of $169.1 billion or 44.1% of net sales. This represents a slight decrease from the prior year's gross margin of 43.3%, primarily driven by favorable product mix and cost efficiencies in our Services segment.",
      text_preview: "For the fiscal year ended September 30, 2023, the Company reported total net sales of $383.3 billion and cost of sales of $214.1 billion, resulting in a gross margin of $169.1 billion or 44.1%...",
    },
    highlight_ranges: [{ start: 169, end: 213 }],
  },
  {
    id: "cite-2",
    claim_text: "Microsoft's gross margin was 69.4% in fiscal 2023",
    confidence_score: 0.92,
    match_type: "paraphrase",
    source_chunk: {
      chunk_id: "msft-10k-2023-chunk-38",
      document: {
        id: "msft-10k-2023",
        ticker: "MSFT",
        document_type: "SEC_10K",
        filing_date: new Date("2023-07-27"),
        fiscal_year: 2023,
        title: "Microsoft Corporation Annual Report (Form 10-K)",
        url: "https://www.sec.gov/cgi-bin/viewer?action=view&cik=789019&accession_number=0000789019-23-000095",
      },
      section: "Management's Discussion and Analysis",
      subsection: "Results of Operations",
      page_number: 35,
      text: "Revenue for fiscal year 2023 was $211.9 billion, an increase of 7% compared to fiscal year 2022. Cost of revenue was $65.9 billion, representing 31.1% of revenue. Gross margin dollars increased 10% to $146.0 billion, with gross margin percentage increasing to 68.9% compared to 67.8% in the prior year. The improvement in gross margin percentage was primarily driven by growth in our higher-margin cloud services.",
      text_preview: "Revenue for fiscal year 2023 was $211.9 billion, an increase of 7% compared to fiscal year 2022. Cost of revenue was $65.9 billion, representing 31.1% of revenue...",
    },
    highlight_ranges: [{ start: 145, end: 235 }],
  },
  {
    id: "cite-3",
    claim_text: "Apple's Services segment showed strong performance",
    confidence_score: 0.88,
    match_type: "inference",
    source_chunk: {
      chunk_id: "aapl-10k-2023-chunk-56",
      document: {
        id: "aapl-10k-2023",
        ticker: "AAPL",
        document_type: "SEC_10K",
        filing_date: new Date("2023-11-03"),
        fiscal_year: 2023,
        title: "Apple Inc. Annual Report (Form 10-K)",
        url: "https://www.sec.gov/cgi-bin/viewer?action=view&cik=320193&accession_number=0000320193-23-000106",
      },
      section: "Item 7. Management's Discussion and Analysis",
      subsection: "Segment Operating Performance",
      page_number: 42,
      text: "Services net sales increased $8.5 billion or 9% during 2023 compared to 2022, driven by growth in advertising, cloud services, and the App Store. Services gross margin increased to 71.7% in 2023 from 71.5% in 2022. The Services segment continues to be a key driver of overall profitability and represents an increasingly important part of our business model.",
      text_preview: "Services net sales increased $8.5 billion or 9% during 2023 compared to 2022, driven by growth in advertising, cloud services, and the App Store...",
    },
    highlight_ranges: [{ start: 0, end: 145 }],
  },
  {
    id: "cite-4",
    claim_text: "Microsoft's cloud services drove margin improvement",
    confidence_score: 0.91,
    match_type: "exact",
    source_chunk: {
      chunk_id: "msft-10k-2023-chunk-45",
      document: {
        id: "msft-10k-2023",
        ticker: "MSFT",
        document_type: "SEC_10K",
        filing_date: new Date("2023-07-27"),
        fiscal_year: 2023,
        title: "Microsoft Corporation Annual Report (Form 10-K)",
        url: "https://www.sec.gov/cgi-bin/viewer?action=view&cik=789019&accession_number=0000789019-23-000095",
      },
      section: "Management's Discussion and Analysis",
      subsection: "Segment Results - Intelligent Cloud",
      page_number: 38,
      text: "Intelligent Cloud revenue increased 16% to $87.9 billion, driven by Azure and other cloud services revenue growth of 27%. Server products and cloud services revenue increased 17%. The segment's gross margin percentage increased to 72.4% from 70.8% in the prior year, primarily due to improvements in Azure gross margin as we continue to optimize our datacenter infrastructure and benefit from economies of scale.",
      text_preview: "Intelligent Cloud revenue increased 16% to $87.9 billion, driven by Azure and other cloud services revenue growth of 27%. Server products and cloud services revenue increased 17%...",
    },
    highlight_ranges: [{ start: 145, end: 320 }],
  },
];

export default function CitationsDemoPage() {
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-primary/10 rounded-lg">
          <FileText className="h-6 w-6 text-primary" />
        </div>
        <div>
          <h1 className="text-2xl font-bold">Enhanced Citations Demo</h1>
          <p className="text-muted-foreground">
            Interactive citation system with hover previews and source linking
          </p>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Sample Query Response</CardTitle>
          <CardDescription>
            &quot;Compare Apple and Microsoft&apos;s gross margins for fiscal year 2023&quot;
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="prose prose-sm max-w-none">
            <p>
              Based on the latest annual filings, both Apple and Microsoft demonstrated strong profitability 
              in fiscal year 2023, though with notably different margin profiles.
            </p>
            <p>
              Apple reported a gross margin of 44.1% for fiscal year 2023{" "}
              <EnhancedCitationBadge citation={MOCK_CITATIONS[0]} index={1} />, 
              representing a slight improvement from the prior year. This performance was supported by 
              strong execution in their Services segment{" "}
              <EnhancedCitationBadge citation={MOCK_CITATIONS[2]} index={3} />, 
              which continues to be a key driver of overall profitability.
            </p>
            <p>
              Microsoft, on the other hand, achieved a significantly higher gross margin of 69.4%{" "}
              <EnhancedCitationBadge citation={MOCK_CITATIONS[1]} index={2} />{" "}
              in fiscal 2023. This superior margin profile was primarily driven by the company&apos;s 
              high-margin cloud services business{" "}
              <EnhancedCitationBadge citation={MOCK_CITATIONS[3]} index={4} />, 
              particularly Azure, which benefited from economies of scale and infrastructure optimization.
            </p>
            <p>
              The 25.3 percentage point difference in gross margins reflects the fundamental differences 
              in their business models: Microsoft&apos;s software and cloud-centric model naturally commands
              higher margins compared to Apple&apos;s hardware-heavy product mix, despite Apple&apos;s growing 
              services revenue.
            </p>
          </div>

          <div className="border-t pt-6 space-y-4">
            <h3 className="font-semibold text-sm">Citations & Sources</h3>
            
            <CitationSummary citations={MOCK_CITATIONS} coverage={95} />

            <div className="flex flex-wrap gap-2">
              {MOCK_CITATIONS.map((citation, idx) => (
                <EnhancedCitationBadge 
                  key={citation.id} 
                  citation={citation} 
                  index={idx + 1} 
                />
              ))}
            </div>

            <SourcesPanel citations={MOCK_CITATIONS} />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Features</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2 text-sm">
            <li className="flex items-start gap-2">
              <span className="text-green-500 mt-0.5">✓</span>
              <span><strong>Hover Previews:</strong> Hover over any citation badge to see a quick preview of the source</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-500 mt-0.5">✓</span>
              <span><strong>Full Source Dialog:</strong> Click to view complete source text with highlighted excerpts</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-500 mt-0.5">✓</span>
              <span><strong>Confidence Scoring:</strong> Color-coded badges show citation confidence (green=high, yellow=medium, red=low)</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-500 mt-0.5">✓</span>
              <span><strong>Match Types:</strong> Distinguishes between exact quotes, paraphrases, and inferences</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-500 mt-0.5">✓</span>
              <span><strong>Citation Coverage:</strong> Summary bar shows percentage of claims with citations</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-500 mt-0.5">✓</span>
              <span><strong>Sources Panel:</strong> Grouped view of all source documents with expandable details</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-500 mt-0.5">✓</span>
              <span><strong>Copy to Clipboard:</strong> One-click copy of source text</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-500 mt-0.5">✓</span>
              <span><strong>External Links:</strong> Direct links to original SEC filings</span>
            </li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
