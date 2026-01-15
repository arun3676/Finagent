"use client";

import { useState, useEffect } from "react";
import { ValidationResult } from "@/types/validation";
import { Button } from "@/components/ui/button";
import { ChevronDown, ChevronUp, CheckCircle2, FileText, RefreshCw } from "lucide-react";
import { cn } from "@/lib/utils";
import { TrustBadge } from "./trust-badge";
import { ConfidenceBar } from "./confidence-bar";

interface TrustScoreFooterProps {
  validation: ValidationResult | null;
  isExpanded: boolean;
  onToggle: () => void;
}

export function TrustScoreFooter({
  validation,
  isExpanded,
  onToggle,
}: TrustScoreFooterProps) {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    if (validation) {
      const timer = setTimeout(() => setIsVisible(true), 300);
      return () => clearTimeout(timer);
    }
  }, [validation]);

  if (!validation) return null;

  const {
    trust_level,
    trust_label,
    trust_color,
    overall_confidence,
    confidence_breakdown,
    claims_checked,
    claims_verified,
    sources_used,
    source_diversity,
    validation_notes,
    validation_attempts,
  } = validation;

  const confidencePercent = Math.round(overall_confidence * 100);
  const allVerified = claims_checked === claims_verified;

  return (
    <div
      className={cn(
        "border-t mt-4 pt-3 transition-all duration-500",
        isVisible ? "opacity-100" : "opacity-0",
        trust_level === "low" && "border-red-200 dark:border-red-900/30"
      )}
    >
      {/* Collapsed View */}
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-3 flex-wrap">
          <TrustBadge
            trustLevel={trust_level}
            trustLabel={trust_label}
            trustColor={trust_color}
          />
          <span className="text-xs text-muted-foreground">
            {confidencePercent}% confidence
          </span>
          <span className="text-xs text-muted-foreground">•</span>
          <span className="text-xs text-muted-foreground">
            {sources_used} source{sources_used !== 1 ? "s" : ""}
          </span>
          {allVerified && (
            <>
              <span className="text-xs text-muted-foreground">•</span>
              <span className="text-xs text-green-600 dark:text-green-400 flex items-center gap-1">
                <CheckCircle2 className="h-3 w-3" />
                All claims verified
              </span>
            </>
          )}
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={onToggle}
          className="h-7 text-xs"
        >
          {isExpanded ? (
            <>
              <ChevronUp className="h-3 w-3 mr-1" />
              Collapse
            </>
          ) : (
            <>
              <ChevronDown className="h-3 w-3 mr-1" />
              Details
            </>
          )}
        </Button>
      </div>

      {/* Expanded View */}
      {isExpanded && (
        <div className="mt-4 space-y-4 animate-in slide-in-from-top-2 duration-300">
          {/* Overall Confidence */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Overall Confidence</span>
              <span className="text-sm font-semibold">{confidencePercent}%</span>
            </div>
            <div className="h-3 bg-muted rounded-full overflow-hidden">
              <div
                className={cn(
                  "h-full transition-all duration-1000 ease-out",
                  overall_confidence >= 0.85
                    ? "bg-green-500"
                    : overall_confidence >= 0.65
                    ? "bg-yellow-500"
                    : "bg-red-500"
                )}
                style={{ width: `${confidencePercent}%` }}
              />
            </div>
          </div>

          {/* Confidence Breakdown */}
          <div className="space-y-2 pt-2 border-t">
            <ConfidenceBar
              label="Factual Accuracy"
              value={confidence_breakdown.factual_accuracy}
            />
            <ConfidenceBar
              label="Citation Coverage"
              value={confidence_breakdown.citation_coverage}
            />
            <ConfidenceBar
              label="Numerical Accuracy"
              value={confidence_breakdown.numerical_accuracy}
            />
            <ConfidenceBar
              label="Source Quality"
              value={confidence_breakdown.source_quality}
            />
          </div>

          {/* Verification Stats */}
          <div className="pt-2 border-t space-y-2 text-xs">
            <div className="flex items-center gap-2 text-muted-foreground">
              <CheckCircle2 className="h-3.5 w-3.5 text-green-600 dark:text-green-400" />
              <span>
                {claims_verified}/{claims_checked} claims verified
              </span>
            </div>
            <div className="flex items-center gap-2 text-muted-foreground">
              <FileText className="h-3.5 w-3.5" />
              <span>Sources: {source_diversity}</span>
            </div>
            {validation_attempts > 1 && (
              <div className="flex items-center gap-2 text-muted-foreground">
                <RefreshCw className="h-3.5 w-3.5" />
                <span>Required {validation_attempts} validation passes</span>
              </div>
            )}
          </div>

          {/* Validation Notes */}
          {validation_notes.length > 0 && (
            <div className="pt-2 border-t">
              <p className="text-xs font-medium text-muted-foreground mb-2">
                Notes:
              </p>
              <ul className="space-y-1">
                {validation_notes.map((note, idx) => (
                  <li
                    key={idx}
                    className="text-xs text-muted-foreground flex items-start gap-2"
                  >
                    <span className="text-primary mt-0.5">•</span>
                    <span>{note}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
