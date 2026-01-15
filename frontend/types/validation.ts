export interface ConfidenceBreakdown {
  factual_accuracy: number;
  citation_coverage: number;
  numerical_accuracy: number;
  source_quality: number;
}

export interface ValidationResult {
  // Overall status
  is_valid: boolean;
  trust_level: "high" | "medium" | "low";
  trust_label: string;
  trust_color: 'green' | 'amber' | 'red' | string;

  // Confidence breakdown
  overall_confidence: number;
  confidence_breakdown: ConfidenceBreakdown;

  // Validation details
  claims_checked: number;
  claims_verified: number;
  claims_unverified: number;

  // Sources quality
  sources_used: number;
  avg_source_relevance: number;
  source_diversity: string;

  // Feedback for user
  validation_notes: string[];

  // Loop info
  validation_attempts: number;
  required_revalidation: boolean;
}
