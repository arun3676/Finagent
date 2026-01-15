"use client";

import { useState } from "react";
import { FileText, ExternalLink, Copy, Check, ChevronRight } from "lucide-react";
import { EnhancedCitation } from "@/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { format } from "date-fns";

interface EnhancedCitationBadgeProps {
  citation: EnhancedCitation;
  index: number;
}

const CONFIDENCE_COLORS = {
  high: "bg-green-500/10 text-green-600 border-green-500/30",
  medium: "bg-yellow-500/10 text-yellow-600 border-yellow-500/30",
  low: "bg-red-500/10 text-red-600 border-red-500/30",
};

const MATCH_TYPE_LABELS = {
  exact: "Exact Quote",
  paraphrase: "Paraphrased",
  inference: "Inferred",
};

export function EnhancedCitationBadge({ citation, index }: EnhancedCitationBadgeProps) {
  const [copied, setCopied] = useState(false);
  const [dialogOpen, setDialogOpen] = useState(false);

  const confidenceLevel =
    citation.confidence_score >= 0.9 ? "high" :
    citation.confidence_score >= 0.7 ? "medium" : "low";

  const handleCopy = async () => {
    await navigator.clipboard.writeText(citation.source_chunk.text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const { source_chunk: source } = citation;
  const { document: doc } = source;

  return (
    <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
      <HoverCard>
        <HoverCardTrigger asChild>
          <DialogTrigger asChild>
            <Badge
              variant="outline"
              className={cn(
                "cursor-pointer hover:bg-muted transition-all",
                "font-mono text-xs gap-1.5 pr-2",
                CONFIDENCE_COLORS[confidenceLevel]
              )}
            >
              <FileText className="h-3 w-3" />
              <span>[{index}]</span>
              <ChevronRight className="h-3 w-3 opacity-50" />
            </Badge>
          </DialogTrigger>
        </HoverCardTrigger>

        <HoverCardContent className="w-96" align="start">
          <div className="space-y-3">
            <div className="flex items-start justify-between">
              <div>
                <p className="font-semibold text-sm">{doc.ticker} - {doc.document_type.replace("SEC_", "")}</p>
                <p className="text-xs text-muted-foreground">
                  {source.section}
                  {source.subsection && ` > ${source.subsection}`}
                </p>
              </div>
              <Badge variant="outline" className={cn("text-xs", CONFIDENCE_COLORS[confidenceLevel])}>
                {Math.round(citation.confidence_score * 100)}%
              </Badge>
            </div>

            <div className="bg-primary/5 rounded p-2 border-l-2 border-primary">
              <p className="text-xs font-medium text-muted-foreground mb-1">Claim:</p>
              <p className="text-sm">&ldquo;{citation.claim_text}&rdquo;</p>
            </div>

            <div className="text-xs text-muted-foreground">
              <p className="line-clamp-3">{source.text_preview}...</p>
            </div>

            <p className="text-xs text-muted-foreground">
              Click to view full source
            </p>
          </div>
        </HoverCardContent>
      </HoverCard>

      <DialogContent className="max-w-2xl max-h-[80vh]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5" />
            Source Document
          </DialogTitle>
          <DialogDescription>
            {doc.ticker} {doc.document_type.replace("SEC_", "")} - 
            Filed {format(new Date(doc.filing_date), "MMM d, yyyy")}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-muted-foreground">Section:</span>
              <span className="ml-2 font-medium">{source.section}</span>
            </div>
            {source.page_number && (
              <div>
                <span className="text-muted-foreground">Page:</span>
                <span className="ml-2 font-medium">{source.page_number}</span>
              </div>
            )}
            <div>
              <span className="text-muted-foreground">Match Type:</span>
              <span className="ml-2 font-medium">
                {MATCH_TYPE_LABELS[citation.match_type]}
              </span>
            </div>
            <div>
              <span className="text-muted-foreground">Confidence:</span>
              <span className={cn("ml-2 font-medium", 
                confidenceLevel === "high" ? "text-green-600" :
                confidenceLevel === "medium" ? "text-yellow-600" : "text-red-600"
              )}>
                {Math.round(citation.confidence_score * 100)}%
              </span>
            </div>
          </div>

          <div className="bg-primary/5 rounded-lg p-4 border-l-4 border-primary">
            <p className="text-xs font-medium text-muted-foreground mb-2">
              CITED CLAIM
            </p>
            <p className="text-sm">&ldquo;{citation.claim_text}&rdquo;</p>
          </div>

          <div className="border rounded-lg">
            <div className="flex items-center justify-between p-3 border-b bg-muted/30">
              <p className="text-sm font-medium">Source Text</p>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleCopy}
                className="h-7"
              >
                {copied ? (
                  <Check className="h-3 w-3 mr-1" />
                ) : (
                  <Copy className="h-3 w-3 mr-1" />
                )}
                {copied ? "Copied" : "Copy"}
              </Button>
            </div>
            <ScrollArea className="h-64 p-4">
              <HighlightedText
                text={source.text}
                highlights={citation.highlight_ranges}
              />
            </ScrollArea>
          </div>

          {doc.url && (
            <div className="flex justify-end">
              <Button variant="outline" asChild>
                <a href={doc.url} target="_blank" rel="noopener noreferrer">
                  <ExternalLink className="h-4 w-4 mr-2" />
                  View Original Filing
                </a>
              </Button>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}

function HighlightedText({ 
  text, 
  highlights 
}: { 
  text: string; 
  highlights?: Array<{ start: number; end: number }> 
}) {
  if (!highlights || highlights.length === 0) {
    return <p className="text-sm whitespace-pre-wrap">{text}</p>;
  }

  const sortedHighlights = [...highlights].sort((a, b) => a.start - b.start);
  
  const parts: React.ReactNode[] = [];
  let lastEnd = 0;

  sortedHighlights.forEach((highlight, idx) => {
    if (highlight.start > lastEnd) {
      parts.push(
        <span key={`text-${idx}`}>
          {text.slice(lastEnd, highlight.start)}
        </span>
      );
    }
    
    parts.push(
      <mark
        key={`highlight-${idx}`}
        className="bg-yellow-200 dark:bg-yellow-800 px-0.5 rounded"
      >
        {text.slice(highlight.start, highlight.end)}
      </mark>
    );
    
    lastEnd = highlight.end;
  });

  if (lastEnd < text.length) {
    parts.push(<span key="text-end">{text.slice(lastEnd)}</span>);
  }

  return <p className="text-sm whitespace-pre-wrap">{parts}</p>;
}
