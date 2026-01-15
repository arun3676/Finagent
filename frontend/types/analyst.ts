// Analyst Notebook Types

export type ChangeDirection = 'up' | 'down' | 'flat';
export type DataQuality = 'high' | 'medium' | 'low';
export type ComparisonType =
  | 'company_vs_company'
  | 'period_vs_period'
  | 'metric_vs_metric'
  | 'industry_vs_company';

export type MetricUnit = 'percent' | 'currency' | 'ratio' | 'count' | 'other';

export interface ExtractedMetric {
  metric_name: string;
  display_name: string;
  value: number;
  formatted_value: string;
  unit: MetricUnit;
  currency?: string;           // e.g., 'USD' if unit is currency
  company: string;
  fiscal_period: string;
  source_section: string;
  source_citation_id?: string;
  comparison_value?: number;
  comparison_label?: string;
  change_percent?: number;
  change_direction?: ChangeDirection;
}

export interface ComparisonItem {
  company: string;
  value: number;
  formatted_value: string;
  citation_id?: string;
}

export interface ExtractedComparison {
  comparison_type: ComparisonType;
  metric_name: string;
  display_name: string;
  items: ComparisonItem[];
  winner?: string;
  insight?: string;
}

export interface AnalystNotebook {
  metrics: ExtractedMetric[];
  comparisons: ExtractedComparison[];
  key_findings: string[];
  data_quality: DataQuality;
  companies_analyzed: string[];
  periods_covered: string[];
}

export interface AnalystNotebookProps {
  notebook: AnalystNotebook | null;
  onCitationClick?: (citationId: string) => void;
  isExpanded: boolean;
  onToggle: () => void;
}

export interface MetricCardProps {
  metric: ExtractedMetric;
  onCitationClick?: (citationId: string) => void;
}

export interface ComparisonChartProps {
  comparison: ExtractedComparison;
  onCitationClick?: (citationId: string) => void;
}

export interface KeyFindingsProps {
  findings: string[];
}
