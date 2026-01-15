"use client";

import { EnhancedCitation } from "@/types";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { FileText } from "lucide-react";
import { cn } from "@/lib/utils";

interface CitationSummaryProps {
  citations: EnhancedCitation[];
  coverage: number;
}

export function CitationSummary({ citations, coverage }: CitationSummaryProps) {
  const highConfidence = citations.filter((c: EnhancedCitation) => c.confidence_score >= 0.9).length;
  const mediumConfidence = citations.filter((c: EnhancedCitation) => c.confidence_score >= 0.7 && c.confidence_score < 0.9).length;
  const lowConfidence = citations.filter((c: EnhancedCitation) => c.confidence_score < 0.7).length;

  const coverageColor = 
    coverage >= 90 ? "text-green-500" :
    coverage >= 70 ? "text-yellow-500" : "text-red-500";

  return (
    <div className="flex items-center gap-4 p-3 bg-muted/30 rounded-lg text-sm">
      <div className="flex items-center gap-2">
        <FileText className="h-4 w-4 text-muted-foreground" />
        <span className="font-medium">{citations.length} Citations</span>
      </div>

      <div className="h-4 w-px bg-border" />

      <div className="flex items-center gap-2">
        <span className="text-muted-foreground">Coverage:</span>
        <span className={cn("font-medium", coverageColor)}>
          {Math.round(coverage)}%
        </span>
        <Progress value={coverage} className="w-16 h-2" />
      </div>

      <div className="h-4 w-px bg-border" />

      <div className="flex items-center gap-2">
        {highConfidence > 0 && (
          <Badge variant="outline" className="bg-green-500/10 text-green-600 border-green-500/30">
            {highConfidence} high
          </Badge>
        )}
        {mediumConfidence > 0 && (
          <Badge variant="outline" className="bg-yellow-500/10 text-yellow-600 border-yellow-500/30">
            {mediumConfidence} medium
          </Badge>
        )}
        {lowConfidence > 0 && (
          <Badge variant="outline" className="bg-red-500/10 text-red-600 border-red-500/30">
            {lowConfidence} low
          </Badge>
        )}
      </div>
    </div>
  );
}
