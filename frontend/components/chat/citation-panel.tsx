"use client";

import { Citation } from "@/types/chat";
import { X, ExternalLink } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

interface CitationPanelProps {
  citation: Citation | null;
  isOpen: boolean;
  onClose: () => void;
}

export function CitationPanel({ citation, isOpen, onClose }: CitationPanelProps) {
  if (!citation) return null;

  const confidenceColor =
    citation.confidence >= 0.8
      ? "bg-green-500"
      : citation.confidence >= 0.6
      ? "bg-yellow-500"
      : "bg-red-500";

  const confidencePercent = Math.round(citation.confidence * 100);
  const confidenceWidth = `${confidencePercent}%`;

  // Get display text with highlighting
  const displayText = citation.source_context || citation.source_text;
  const hasHighlight =
    citation.highlight_start !== undefined && citation.highlight_end !== undefined;

  const renderHighlightedText = () => {
    if (!hasHighlight) {
      return <p className="text-sm leading-relaxed whitespace-pre-wrap">{displayText}</p>;
    }

    const before = displayText.slice(0, citation.highlight_start);
    const highlighted = displayText.slice(citation.highlight_start, citation.highlight_end);
    const after = displayText.slice(citation.highlight_end);

    return (
      <p className="text-sm leading-relaxed whitespace-pre-wrap">
        {before}
        <mark className="bg-yellow-200 dark:bg-yellow-800/50 px-0.5 rounded font-medium">
          {highlighted}
        </mark>
        {after}
      </p>
    );
  };

  const metadata = citation.metadata || {};
  const companyName = metadata.company_name || metadata.ticker || "Unknown";
  const documentType = metadata.document_type || metadata.filing_type || "Document";
  const filingDate = metadata.filing_date
    ? new Date(metadata.filing_date).toLocaleDateString("en-US", {
        year: "numeric",
        month: "short",
        day: "numeric",
      })
    : null;
  const section = metadata.section || "N/A";
  const pageNumber = metadata.page_number || metadata.page;

  return (
    <>
      {/* Backdrop */}
      <div
        className={cn(
          "fixed inset-0 bg-black/20 z-40 transition-opacity duration-300",
          isOpen ? "opacity-100" : "opacity-0 pointer-events-none"
        )}
        onClick={onClose}
      />

      {/* Panel */}
      <div
        className={cn(
          "fixed top-0 right-0 h-full w-full md:w-[450px] bg-background border-l shadow-2xl z-50",
          "transform transition-transform duration-300 ease-in-out",
          isOpen ? "translate-x-0" : "translate-x-full"
        )}
      >
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b">
            <div className="flex items-center gap-2">
              <h3 className="font-semibold text-lg">Source [{citation.citation_number}]</h3>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={onClose}
              className="h-8 w-8"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>

          {/* Content */}
          <ScrollArea className="flex-1">
            <div className="p-4 space-y-4">
              {/* Document Info */}
              <div className="space-y-1">
                <h4 className="font-semibold text-base">
                  {companyName} {documentType}
                  {filingDate && ` • ${filingDate}`}
                </h4>
                <p className="text-sm text-muted-foreground">{section}</p>
                {pageNumber && (
                  <p className="text-xs text-muted-foreground">Page {pageNumber}</p>
                )}
              </div>

              {/* Source Text with Highlight */}
              <div className="border rounded-lg p-4 bg-muted/30">
                <p className="text-xs font-medium text-muted-foreground mb-3">
                  SOURCE TEXT
                </p>
                {renderHighlightedText()}
              </div>

              {/* Divider */}
              <div className="border-t" />

              {/* Claim */}
              <div className="space-y-2">
                <p className="text-xs font-medium text-muted-foreground">CLAIM</p>
                <p className="text-sm italic border-l-4 border-primary pl-3 py-1">
                  &ldquo;{citation.claim}&rdquo;
                </p>
              </div>

              {/* Confidence */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <p className="text-xs font-medium text-muted-foreground">
                    CONFIDENCE
                  </p>
                  <span className="text-sm font-semibold">{confidencePercent}%</span>
                </div>
                <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
                  <div
                    className={cn("h-full transition-all duration-500", confidenceColor)}
                    style={{ width: confidenceWidth }}
                  />
                </div>
                <p className="text-xs text-muted-foreground">
                  {citation.confidence >= 0.8
                    ? "High confidence - Direct quote or exact match"
                    : citation.confidence >= 0.6
                    ? "Medium confidence - Paraphrased or inferred"
                    : "Low confidence - Verify this claim independently"}
                </p>
              </div>
            </div>
          </ScrollArea>

          {/* Footer */}
          {(citation.source_url || metadata.source_url) && (
            <div className="p-4 border-t">
              <Button
                variant="outline"
                className="w-full"
                asChild
              >
                <a
                  href={citation.source_url || metadata.source_url}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <ExternalLink className="h-4 w-4 mr-2" />
                  View Full Document
                </a>
              </Button>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
