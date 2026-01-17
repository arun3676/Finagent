"use client";

import { useState } from "react";
import { ValidationResult } from "@/types/validation";
import { TrustScoreFooter } from "./trust-score-footer";
import { Button } from "@/components/ui/button";

/**
 * Demo component showing how to use the Trust Score Footer
 * 
 * Integration example:
 * ```tsx
 * const [validation, setValidation] = useState<ValidationResult | null>(null);
 * const [isTrustExpanded, setIsTrustExpanded] = useState(false);
 * 
 * // In SSE handler:
 * if (event.type === 'validation') {
 *   setValidation(event.validation);
 * }
 * 
 * // Auto-expand for low trust
 * useEffect(() => {
 *   if (validation?.trust_level === 'low') {
 *     setIsTrustExpanded(true);
 *   }
 * }, [validation?.trust_level]);
 * 
 * // In message component:
 * <TrustScoreFooter
 *   validation={validation}
 *   isExpanded={isTrustExpanded}
 *   onToggle={() => setIsTrustExpanded(!isTrustExpanded)}
 * />
 * ```
 */
export function TrustScoreDemo() {
  const [isTrustExpanded, setIsTrustExpanded] = useState(false);
  const [currentExample, setCurrentExample] = useState<"high" | "medium" | "low">("high");

  const examples: Record<"high" | "medium" | "low", ValidationResult> = {
    high: {
      is_valid: true,
      trust_level: "high",
      trust_label: "Verified",
      trust_color: "green",
      overall_confidence: 0.91,
      confidence_breakdown: {
        factual_accuracy: 0.95,
        citation_coverage: 0.88,
        numerical_accuracy: 1.0,
        source_quality: 0.82,
      },
      claims_checked: 5,
      claims_verified: 5,
      claims_unverified: 0,
      sources_used: 4,
      avg_source_relevance: 0.92,
      source_diversity: "2 10-K filings (AAPL, MSFT)",
      validation_notes: ["All numbers verified against source"],
      validation_attempts: 1,
      required_revalidation: false,
    },
    medium: {
      is_valid: true,
      trust_level: "medium",
      trust_label: "Review Recommended",
      trust_color: "amber",
      overall_confidence: 0.72,
      confidence_breakdown: {
        factual_accuracy: 0.78,
        citation_coverage: 0.65,
        numerical_accuracy: 0.85,
        source_quality: 0.60,
      },
      claims_checked: 4,
      claims_verified: 3,
      claims_unverified: 1,
      sources_used: 2,
      avg_source_relevance: 0.75,
      source_diversity: "1 10-K filing (AAPL)",
      validation_notes: [
        "One claim could not be fully verified",
        "Limited source diversity",
      ],
      validation_attempts: 2,
      required_revalidation: false,
    },
    low: {
      is_valid: false,
      trust_level: "low",
      trust_label: "Low Confidence",
      trust_color: "red",
      overall_confidence: 0.45,
      confidence_breakdown: {
        factual_accuracy: 0.50,
        citation_coverage: 0.40,
        numerical_accuracy: 0.55,
        source_quality: 0.35,
      },
      claims_checked: 6,
      claims_verified: 2,
      claims_unverified: 4,
      sources_used: 1,
      avg_source_relevance: 0.45,
      source_diversity: "1 source (limited)",
      validation_notes: [
        "Multiple claims could not be verified",
        "Insufficient source coverage",
        "Recommend manual verification",
      ],
      validation_attempts: 3,
      required_revalidation: true,
    },
  };

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
      <div className="space-y-2">
        <h3 className="font-semibold text-lg">Trust Score Footer Demo</h3>
        <p className="text-sm text-muted-foreground">
          Select a trust level to see how the footer displays validation results
        </p>
      </div>

      <div className="flex gap-2">
        <Button
          variant={currentExample === "high" ? "default" : "outline"}
          onClick={() => setCurrentExample("high")}
        >
          High Trust
        </Button>
        <Button
          variant={currentExample === "medium" ? "default" : "outline"}
          onClick={() => setCurrentExample("medium")}
        >
          Medium Trust
        </Button>
        <Button
          variant={currentExample === "low" ? "default" : "outline"}
          onClick={() => setCurrentExample("low")}
        >
          Low Trust
        </Button>
      </div>

      <div className="border rounded-lg p-6 bg-card">
        <div className="prose prose-sm max-w-none mb-4">
          <p>
            This is a sample response with validation results. The trust score
            footer appears below showing the validation status and confidence
            metrics.
          </p>
        </div>

        <TrustScoreFooter
          validation={examples[currentExample]}
          isExpanded={isTrustExpanded}
          onToggle={() => setIsTrustExpanded(!isTrustExpanded)}
        />
      </div>

      <div className="text-sm text-muted-foreground space-y-1">
        <p className="font-medium">Features:</p>
        <ul className="list-disc list-inside space-y-1">
          <li>Color-coded trust badges (green/amber/red)</li>
          <li>Overall confidence percentage and progress bar</li>
          <li>Detailed confidence breakdown when expanded</li>
          <li>Claims verification status</li>
          <li>Source diversity information</li>
          <li>Validation notes and attempt count</li>
          <li>Auto-expands for low trust level</li>
        </ul>
      </div>
    </div>
  );
}
