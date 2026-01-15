"use client";

import { FileText } from "lucide-react";
import { Citation } from "@/types/chat";
import { Badge } from "@/components/ui/badge";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { cn } from "@/lib/utils";

interface CitationBadgeProps {
  citation: Citation;
  index: number;
}

export function CitationBadge({ citation, index }: CitationBadgeProps) {
  const confidenceColor =
    citation.confidence >= 0.9 ? "bg-green-500/10 text-green-600 border-green-500/20" :
    citation.confidence >= 0.7 ? "bg-yellow-500/10 text-yellow-600 border-yellow-500/20" :
    "bg-red-500/10 text-red-600 border-red-500/20";
  const sourceUrl = citation.source_url?.trim();

  return (
    <HoverCard>
      <HoverCardTrigger asChild>
        <span className="inline-flex">
          <Badge
            variant="outline"
            className={cn(
              "cursor-pointer hover:bg-muted transition-colors",
              "font-mono text-xs"
            )}
          >
            <FileText className="h-3 w-3 mr-1" />
            [{index}]
          </Badge>
        </span>
      </HoverCardTrigger>
      <HoverCardContent className="w-96" align="start">
        <div className="space-y-3">
          {/* Header */}
          <div className="flex items-center justify-between">
            <span className="font-semibold text-sm">Source Citation</span>
            <Badge variant="outline" className={cn("text-xs", confidenceColor)}>
              {Math.round(citation.confidence * 100)}% confidence
            </Badge>
          </div>

          {/* Document Info */}
          <div className="text-xs space-y-1">
            <div className="flex items-center gap-2 text-muted-foreground">
              <span className="font-medium">Document:</span>
              <span>{citation.source_chunk_id}</span>
            </div>
            {sourceUrl && (
              <div className="flex items-center gap-2 text-muted-foreground">
                <span className="font-medium">Link:</span>
                <a
                  href={sourceUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="text-primary underline"
                >
                  Open source
                </a>
              </div>
            )}
            {citation.metadata?.section && (
              <div className="flex items-center gap-2 text-muted-foreground">
                <span className="font-medium">Section:</span>
                <span>{citation.metadata.section}</span>
              </div>
            )}
            {citation.page_reference && (
              <div className="flex items-center gap-2 text-muted-foreground">
                <span className="font-medium">Page:</span>
                <span>{citation.page_reference}</span>
              </div>
            )}
          </div>

          {/* Claim */}
          <div className="bg-muted/50 rounded p-2">
            <p className="text-xs font-medium text-muted-foreground mb-1">Claim:</p>
            <p className="text-sm">&ldquo;{citation.claim}&rdquo;</p>
          </div>

          {/* Source Text */}
          <div className="border rounded p-2">
            <p className="text-xs font-medium text-muted-foreground mb-1">Source Text:</p>
            <p className="text-xs text-muted-foreground line-clamp-4">
              {citation.source_text}
            </p>
          </div>
        </div>
      </HoverCardContent>
    </HoverCard>
  );
}
