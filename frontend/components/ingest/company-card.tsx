"use client";

import { useState } from "react";
import { 
  RefreshCw, 
  Trash2, 
  ChevronDown, 
  ChevronRight,
  CheckCircle,
  Clock,
  AlertCircle
} from "lucide-react";
import { Company, IndexedDocument, DocumentType } from "@/types/company";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";
import { formatDistanceToNow } from "date-fns";

interface CompanyCardProps {
  company: Company;
  onRefresh?: (ticker: string) => void;
  onDelete?: (ticker: string) => void;
}

const DOC_TYPE_ICONS: Record<DocumentType, string> = {
  SEC_10K: "📄",
  SEC_10Q: "📋",
  SEC_8K: "📰",
  EARNINGS_CALL: "🎤",
};

const DOC_TYPE_LABELS: Record<DocumentType, string> = {
  SEC_10K: "10-K",
  SEC_10Q: "10-Q",
  SEC_8K: "8-K",
  EARNINGS_CALL: "Earnings",
};

export function CompanyCard({ company, onRefresh, onDelete }: CompanyCardProps) {
  const [isOpen, setIsOpen] = useState(true);

  const totalChunks = company.indexed_documents.reduce(
    (sum, doc) => sum + doc.chunk_count,
    0
  );

  const readyDocs = company.indexed_documents.filter(
    (doc) => doc.status === "ready"
  ).length;

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <div className="border rounded-lg overflow-hidden bg-card">
        {/* Header */}
        <CollapsibleTrigger asChild>
          <div className="flex items-center justify-between p-4 cursor-pointer hover:bg-muted/50 transition-colors">
            <div className="flex items-center gap-3">
              {isOpen ? (
                <ChevronDown className="h-4 w-4 text-muted-foreground" />
              ) : (
                <ChevronRight className="h-4 w-4 text-muted-foreground" />
              )}
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="font-semibold">{company.ticker}</h3>
                  <span className="text-muted-foreground">-</span>
                  <span className="text-sm text-muted-foreground truncate max-w-[200px]">
                    {company.name}
                  </span>
                </div>
                <div className="flex items-center gap-2 text-xs text-muted-foreground mt-1">
                  <span>{readyDocs}/{company.indexed_documents.length} docs ready</span>
                  <span>•</span>
                  <span>{totalChunks.toLocaleString()} chunks</span>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={(e) => {
                  e.stopPropagation();
                  onRefresh?.(company.ticker);
                }}
              >
                <RefreshCw className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 text-destructive hover:text-destructive"
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete?.(company.ticker);
                }}
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CollapsibleTrigger>

        {/* Documents List */}
        <CollapsibleContent>
          <div className="border-t divide-y">
            {company.indexed_documents.map((doc) => (
              <DocumentRow key={doc.id} document={doc} />
            ))}
            {company.indexed_documents.length === 0 && (
              <div className="p-4 text-center text-sm text-muted-foreground">
                No documents indexed yet
              </div>
            )}
          </div>
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
}

function DocumentRow({ document }: { document: IndexedDocument }) {
  const statusIcon = {
    ready: <CheckCircle className="h-4 w-4 text-green-500" />,
    processing: <Clock className="h-4 w-4 text-yellow-500 animate-pulse" />,
    pending: <Clock className="h-4 w-4 text-muted-foreground" />,
    failed: <AlertCircle className="h-4 w-4 text-red-500" />,
  }[document.status];

  const statusColor = {
    ready: "text-green-600 bg-green-500/10",
    processing: "text-yellow-600 bg-yellow-500/10",
    pending: "text-muted-foreground bg-muted",
    failed: "text-red-600 bg-red-500/10",
  }[document.status];

  return (
    <div className="flex items-center justify-between p-3 hover:bg-muted/30 transition-colors">
      <div className="flex items-center gap-3">
        <span className="text-lg">{DOC_TYPE_ICONS[document.document_type]}</span>
        <div>
          <div className="flex items-center gap-2">
            <span className="font-medium text-sm">
              {DOC_TYPE_LABELS[document.document_type]}
            </span>
            {document.fiscal_year && (
              <span className="text-sm text-muted-foreground">
                FY{document.fiscal_year}
                {document.fiscal_quarter && ` Q${document.fiscal_quarter}`}
              </span>
            )}
          </div>
          <p className="text-xs text-muted-foreground">
            {document.chunk_count.toLocaleString()} chunks • 
            Indexed {formatDistanceToNow(new Date(document.indexed_at), { addSuffix: true })}
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2">
        {statusIcon}
        <Badge variant="outline" className={cn("text-xs", statusColor)}>
          {document.status}
        </Badge>
      </div>
    </div>
  );
}
