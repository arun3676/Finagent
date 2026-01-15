"use client";

import { useState } from "react";
import { FileText, ChevronRight, ExternalLink, Building2 } from "lucide-react";
import { EnhancedCitation, SourceDocument } from "@/types";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { cn } from "@/lib/utils";

interface SourcesPanelProps {
  citations: EnhancedCitation[];
  trigger?: React.ReactNode;
}

export function SourcesPanel({ citations, trigger }: SourcesPanelProps) {
  const [selectedDoc, setSelectedDoc] = useState<string | null>(null);

  const groupedCitations = citations.reduce((acc, citation) => {
    const docId = citation.source_chunk.document.id;
    if (!acc[docId]) {
      acc[docId] = {
        document: citation.source_chunk.document,
        citations: [],
      };
    }
    acc[docId].citations.push(citation);
    return acc;
  }, {} as Record<string, { document: SourceDocument; citations: EnhancedCitation[] }>);

  const documents = Object.values(groupedCitations);

  return (
    <Sheet>
      <SheetTrigger asChild>
        {trigger || (
          <Button variant="outline" size="sm" className="gap-2">
            <FileText className="h-4 w-4" />
            View Sources ({documents.length})
          </Button>
        )}
      </SheetTrigger>
      <SheetContent className="w-[500px] sm:max-w-[500px]">
        <SheetHeader>
          <SheetTitle>Source Documents</SheetTitle>
          <SheetDescription>
            {citations.length} citations from {documents.length} documents
          </SheetDescription>
        </SheetHeader>

        <ScrollArea className="h-[calc(100vh-120px)] mt-4">
          <div className="space-y-4">
            {documents.map(({ document, citations }) => (
              <DocumentCard
                key={document.id}
                document={document}
                citations={citations}
                isExpanded={selectedDoc === document.id}
                onToggle={() => setSelectedDoc(
                  selectedDoc === document.id ? null : document.id
                )}
              />
            ))}
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}

interface DocumentCardProps {
  document: SourceDocument;
  citations: EnhancedCitation[];
  isExpanded: boolean;
  onToggle: () => void;
}

function DocumentCard({ document, citations, isExpanded, onToggle }: DocumentCardProps) {
  const avgConfidence = citations.reduce((sum, c) => sum + c.confidence_score, 0) / citations.length;

  return (
    <div className="border rounded-lg overflow-hidden">
      <button
        className="w-full flex items-center gap-3 p-4 hover:bg-muted/50 transition-colors text-left"
        onClick={onToggle}
      >
        <div className="p-2 bg-primary/10 rounded-lg">
          <Building2 className="h-5 w-5 text-primary" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-semibold">{document.ticker}</span>
            <Badge variant="outline" className="text-xs">
              {document.document_type.replace("SEC_", "")}
            </Badge>
          </div>
          <p className="text-sm text-muted-foreground truncate">
            {document.title || `FY${document.fiscal_year}${document.fiscal_quarter ? ` Q${document.fiscal_quarter}` : ""}`}
          </p>
        </div>
        <div className="text-right">
          <p className="text-sm font-medium">{citations.length} citations</p>
          <p className="text-xs text-muted-foreground">
            {Math.round(avgConfidence * 100)}% avg confidence
          </p>
        </div>
        <ChevronRight className={cn(
          "h-4 w-4 text-muted-foreground transition-transform",
          isExpanded && "rotate-90"
        )} />
      </button>

      {isExpanded && (
        <div className="border-t divide-y">
          {citations.map((citation, idx) => (
            <div key={citation.id} className="p-3 text-sm">
              <div className="flex items-start gap-2">
                <Badge variant="outline" className="text-xs shrink-0">
                  [{idx + 1}]
                </Badge>
                <p className="text-muted-foreground line-clamp-2">
                  {citation.claim_text}
                </p>
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                Section: {citation.source_chunk.section}
              </p>
            </div>
          ))}
          
          {document.url && (
            <div className="p-3">
              <Button variant="ghost" size="sm" asChild className="w-full">
                <a href={document.url} target="_blank" rel="noopener noreferrer">
                  <ExternalLink className="h-4 w-4 mr-2" />
                  View Original Filing
                </a>
              </Button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
