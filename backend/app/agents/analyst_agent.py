"""
Analyst Agent

Extracts data and performs financial calculations from retrieved documents.
Specializes in SEC filing analysis and financial metrics.

Capabilities:
- Data extraction (revenue, margins, ratios)
- Financial calculations (growth rates, comparisons)
- Trend identification
- Cross-document analysis

Usage:
    agent = AnalystAgent()
    analysis = await agent.analyze(query, documents)
"""

from typing import List, Optional, Dict, Any
from decimal import Decimal
from openai import AsyncOpenAI

from app.config import settings
from app.models import (
    AgentState,
    RetrievedDocument,
    AgentRole,
    StepEvent,
    ExtractedMetric,
    ExtractedComparison,
    AnalystNotebook
)
from app.agents.prompts import ANALYST_SYSTEM_PROMPT, ANALYST_USER_TEMPLATE
from datetime import datetime
import re


class AnalystAgent:
    """
    Financial analysis agent.
    
    Extracts structured data from documents and performs
    calculations to answer financial queries.
    """
    
    def __init__(self, model: str = None):
        """
        Initialize analyst agent.
        
        Args:
            model: LLM model to use
        """
        self.model = model or settings.LLM_MODEL
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    
    async def analyze(
        self,
        query: str,
        documents: List[RetrievedDocument]
    ) -> Dict[str, Any]:
        """
        Analyze documents to answer a query.
        
        Args:
            query: Analysis query
            documents: Retrieved documents
            
        Returns:
            Analysis results with extracted data
        """
        if not documents:
            return {"summary": "No documents available for analysis.", "key_metrics": {}}

        try:
            # Call LLM to extract and analyze data
            analysis_text = await self._call_llm_for_extraction(query, documents)
            
            # Parse the response (assuming LLM returns JSON or structured text)
            # For now, we'll try to parse it as JSON if possible, otherwise wrap it
            import json
            try:
                # Find JSON block
                start = analysis_text.find('{')
                end = analysis_text.rfind('}') + 1
                if start != -1 and end > start:
                    json_str = analysis_text[start:end]
                    return json.loads(json_str)
                else:
                    return {"summary": analysis_text}
            except json.JSONDecodeError:
                return {"summary": analysis_text}
                
        except Exception as e:
            # Fallback in case of error
            return {
                "summary": f"Error during analysis: {str(e)}",
                "error": str(e)
            }

    async def _call_llm_for_extraction(
        self,
        query: str,
        documents: List[RetrievedDocument]
    ) -> str:
        """
        Use LLM to extract and analyze data.
        
        Args:
            query: Analysis query
            documents: Source documents
            
        Returns:
            LLM analysis response
        """
        context = self._format_documents_for_context(documents)
        
        messages = [
            {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
            {"role": "user", "content": ANALYST_USER_TEMPLATE.format(
                query=query,
                documents=context
            )}
        ]
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"} # Force JSON for easier parsing
        )
        
        return response.choices[0].message.content or "{}"
    
    def _format_documents_for_context(
        self,
        documents: List[RetrievedDocument],
        max_chars: int = 10000
    ) -> str:
        """
        Format documents for LLM context.
        
        Args:
            documents: Documents to format
            max_chars: Maximum characters to include
            
        Returns:
            Formatted document string
        """
        formatted = []
        total_chars = 0
        
        for i, doc in enumerate(documents):
            section = doc.chunk.section or "Unknown"
            source = f"{doc.chunk.metadata.ticker} {doc.chunk.metadata.document_type.value}"
            
            doc_text = f"[Document {i+1}] Source: {source}, Section: {section}\n{doc.chunk.content}\n"
            
            if total_chars + len(doc_text) > max_chars:
                break
            
            formatted.append(doc_text)
            total_chars += len(doc_text)
        
        return "\n---\n".join(formatted)
    
    def validate_extraction(
        self,
        extracted: Dict[str, Any],
        documents: List[RetrievedDocument]
    ) -> Dict[str, bool]:
        """
        Validate extracted data against source documents.
        
        Args:
            extracted: Extracted data points
            documents: Source documents
            
        Returns:
            Validation results for each data point
        """
        validation_results = {}
        
        # Combine all document content into one string for searching
        # Lowercase for case-insensitive matching
        full_text = " ".join([doc.chunk.content.lower() for doc in documents])
        
        def check_value(value: Any) -> bool:
            if isinstance(value, (int, float, Decimal)):
                # precise number check could be tricky due to formatting, 
                # but we can try basic string presence
                str_val = str(value)
                if str_val in full_text:
                    return True
                # Try with commas
                if f"{value:,}" in full_text:
                    return True
                return False
            elif isinstance(value, str):
                return value.lower() in full_text
            elif isinstance(value, dict):
                return all(check_value(v) for v in value.values())
            elif isinstance(value, list):
                return all(check_value(v) for v in value)
            return True # Skip validation for other types
            
        for key, value in extracted.items():
            if key == "summary":
                continue # Skip summary validation as it's generated text
            validation_results[key] = check_value(value)
            
        return validation_results

    def _build_analyst_notebook(
        self,
        raw_data: Dict[str, Any],
        documents: List[RetrievedDocument],
        query: str
    ) -> AnalystNotebook:
        """
        Build structured AnalystNotebook from raw extracted data.

        Args:
            raw_data: Raw extracted data from LLM
            documents: Source documents
            query: Original query

        Returns:
            Structured AnalystNotebook
        """
        metrics = []
        comparisons = []
        key_findings = []
        companies = set()
        periods = set()

        # Extract companies from documents
        for doc in documents:
            companies.add(doc.chunk.metadata.ticker)
            # Extract period info
            if doc.chunk.metadata.fiscal_year:
                year = doc.chunk.metadata.fiscal_year
                if doc.chunk.metadata.fiscal_quarter:
                    periods.add(f"Q{doc.chunk.metadata.fiscal_quarter} {year}")
                else:
                    periods.add(f"FY{year}")

        # Build metrics from raw data
        for key, value in raw_data.items():
            if key in ["summary", "error", "reasoning"]:
                continue  # Skip non-metric fields

            # Try to parse metrics
            metric = self._parse_metric(key, value, list(companies), list(periods), documents)
            if metric:
                metrics.append(metric)

        # Build comparisons if multiple companies
        if len(companies) > 1:
            comparisons = self._build_comparisons(metrics, list(companies))

        # Generate key findings
        key_findings = self._generate_key_findings(metrics, comparisons, raw_data)

        # Assess data quality
        data_quality = self._assess_data_quality(metrics, documents)

        return AnalystNotebook(
            metrics=metrics,
            comparisons=comparisons,
            key_findings=key_findings,
            data_quality=data_quality,
            companies_analyzed=sorted(list(companies)),
            periods_covered=sorted(list(periods), reverse=True)
        )

    def _parse_metric(
        self,
        key: str,
        value: Any,
        companies: List[str],
        periods: List[str],
        documents: List[RetrievedDocument]
    ) -> Optional[ExtractedMetric]:
        """
        Parse a raw data entry into an ExtractedMetric.

        Args:
            key: Metric key (e.g., "apple_revenue", "gross_margin")
            value: Metric value
            companies: List of companies
            periods: List of periods
            documents: Source documents

        Returns:
            ExtractedMetric or None if can't parse
        """
        # Extract numeric value
        numeric_value = self._extract_numeric_value(value)
        if numeric_value is None:
            return None

        # Determine company (check if key contains ticker)
        company = companies[0] if companies else "UNKNOWN"
        for ticker in companies:
            if ticker.lower() in key.lower():
                company = ticker
                break

        # Determine metric name and display name
        metric_name = key.lower().replace(company.lower() + "_", "").replace("_", " ")
        display_name = metric_name.replace("_", " ").title()

        # Determine unit and format value
        unit, formatted_value, currency = self._determine_unit_and_format(value, numeric_value)

        # Get fiscal period (use most recent from documents)
        fiscal_period = periods[0] if periods else "Unknown"

        # Try to find source section
        source_section = ""
        if documents:
            source_section = documents[0].chunk.section or ""

        return ExtractedMetric(
            metric_name=metric_name,
            display_name=display_name,
            value=numeric_value,
            formatted_value=formatted_value,
            unit=unit,
            currency=currency,
            company=company,
            fiscal_period=fiscal_period,
            source_section=source_section
        )

    def _extract_numeric_value(self, value: Any) -> Optional[float]:
        """Extract numeric value from various formats."""
        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            # Remove common formatting
            cleaned = value.replace("$", "").replace(",", "").replace("%", "").strip()

            # Handle B, M, K suffixes
            multiplier = 1
            if cleaned.endswith("B"):
                multiplier = 1e9
                cleaned = cleaned[:-1]
            elif cleaned.endswith("M"):
                multiplier = 1e6
                cleaned = cleaned[:-1]
            elif cleaned.endswith("K"):
                multiplier = 1e3
                cleaned = cleaned[:-1]

            try:
                return float(cleaned) * multiplier
            except ValueError:
                pass

        return None

    def _determine_unit_and_format(
        self,
        original_value: Any,
        numeric_value: float
    ) -> tuple:
        """
        Determine unit type and format the value appropriately.

        Returns:
            Tuple of (unit, formatted_value, currency)
        """
        str_value = str(original_value)

        # Check for percentage
        if "%" in str_value or "percent" in str_value.lower():
            return ("percent", f"{numeric_value:.1f}%", None)

        # Check for currency
        if "$" in str_value or "usd" in str_value.lower():
            # Determine scale
            if numeric_value >= 1e9:
                return ("currency", f"${numeric_value/1e9:.1f}B", "USD")
            elif numeric_value >= 1e6:
                return ("currency", f"${numeric_value/1e6:.1f}M", "USD")
            elif numeric_value >= 1e3:
                return ("currency", f"${numeric_value/1e3:.1f}K", "USD")
            else:
                return ("currency", f"${numeric_value:.2f}", "USD")

        # Check for ratio
        if "ratio" in str_value.lower() or "times" in str_value.lower():
            return ("ratio", f"{numeric_value:.2f}x", None)

        # Default to count or other
        if numeric_value == int(numeric_value):
            return ("count", f"{int(numeric_value):,}", None)
        else:
            return ("other", f"{numeric_value:.2f}", None)

    def _build_comparisons(
        self,
        metrics: List[ExtractedMetric],
        companies: List[str]
    ) -> List[ExtractedComparison]:
        """Build comparison objects from metrics."""
        comparisons = []

        # Group metrics by metric_name
        metrics_by_name = {}
        for metric in metrics:
            if metric.metric_name not in metrics_by_name:
                metrics_by_name[metric.metric_name] = []
            metrics_by_name[metric.metric_name].append(metric)

        # Create comparisons for metrics with multiple companies
        for metric_name, metric_list in metrics_by_name.items():
            if len(metric_list) >= 2:
                # Determine winner (higher is better for most metrics)
                winner_metric = max(metric_list, key=lambda m: m.value)

                # Calculate insight
                values = [m.value for m in metric_list]
                diff = max(values) - min(values)

                if winner_metric.unit == "percent":
                    insight = f"{winner_metric.company} leads by {diff:.1f} percentage points"
                elif winner_metric.unit == "currency":
                    if diff >= 1e9:
                        insight = f"{winner_metric.company} leads by ${diff/1e9:.1f}B"
                    else:
                        insight = f"{winner_metric.company} leads by ${diff/1e6:.1f}M"
                else:
                    insight = f"{winner_metric.company} has higher {metric_name}"

                comparison = ExtractedComparison(
                    comparison_type="company_vs_company",
                    metric_name=metric_name,
                    display_name=metric_list[0].display_name,
                    items=metric_list,
                    winner=winner_metric.company,
                    insight=insight
                )
                comparisons.append(comparison)

        return comparisons

    def _generate_key_findings(
        self,
        metrics: List[ExtractedMetric],
        comparisons: List[ExtractedComparison],
        raw_data: Dict[str, Any]
    ) -> List[str]:
        """Generate key findings bullet points."""
        findings = []

        # Add comparison insights
        for comp in comparisons:
            if comp.insight:
                findings.append(comp.insight)

        # Add summary if present
        if "summary" in raw_data and isinstance(raw_data["summary"], str):
            # Extract first sentence or two
            summary = raw_data["summary"]
            sentences = summary.split(".")
            for sent in sentences[:2]:
                if sent.strip():
                    findings.append(sent.strip())

        # Add metric highlights
        if len(metrics) >= 2:
            findings.append(f"Analyzed {len(metrics)} financial metrics across {len(set(m.company for m in metrics))} companies")

        return findings[:5]  # Limit to top 5 findings

    def _assess_data_quality(
        self,
        metrics: List[ExtractedMetric],
        documents: List[RetrievedDocument]
    ) -> str:
        """Assess the quality of extracted data."""
        if not metrics:
            return "low"

        # Check if we have good source documents
        avg_score = sum(d.score for d in documents) / len(documents) if documents else 0

        # High quality: multiple metrics with high-confidence sources
        if len(metrics) >= 3 and avg_score >= 0.8:
            return "high"

        # Medium quality: some metrics with decent sources
        if len(metrics) >= 1 and avg_score >= 0.5:
            return "medium"

        # Low quality: few metrics or low confidence
        return "low"

    async def analyze_for_state(self, state: AgentState) -> Dict[str, Any]:
        """
        Analyze extracted data and update agent state.

        LangGraph-compatible interface - returns dict with updated fields.

        Args:
            state: Current agent state

        Returns:
            Dict with extracted_data field for state update
        """
        new_events = list(state.step_events) if state.step_events else []

        # Emit analysis start event
        analysis_start_event = StepEvent(
            event_type="step_detail",
            agent=AgentRole.ANALYST,
            timestamp=datetime.now(),
            data={
                "action": "extracting_data",
                "documents_to_analyze": len(state.retrieved_docs),
                "message": f"Analyzing {len(state.retrieved_docs)} document chunks"
            }
        )
        new_events.append(analysis_start_event)

        analysis = await self.analyze(
            state.original_query,
            state.retrieved_docs
        )

        # Build structured analyst notebook
        analyst_notebook = None
        if isinstance(analysis, dict):
            try:
                analyst_notebook = self._build_analyst_notebook(
                    raw_data=analysis,
                    documents=state.retrieved_docs,
                    query=state.original_query
                )

                # Emit insight events for metrics in notebook
                for metric in analyst_notebook.metrics:
                    insight_event = StepEvent(
                        event_type="analysis_insight",
                        agent=AgentRole.ANALYST,
                        timestamp=datetime.now(),
                        data={
                            "metric": metric.metric_name,
                            "value": metric.formatted_value,
                            "company": metric.company,
                            "source": "extracted_from_documents"
                        }
                    )
                    new_events.append(insight_event)
            except Exception as e:
                # If notebook building fails, fall back to raw data events
                for key, value in analysis.items():
                    insight_event = StepEvent(
                        event_type="analysis_insight",
                        agent=AgentRole.ANALYST,
                        timestamp=datetime.now(),
                        data={
                            "metric": key,
                            "value": str(value),
                            "source": "extracted_from_documents"
                        }
                    )
                    new_events.append(insight_event)

        return {
            "extracted_data": analysis,
            "analyst_notebook": analyst_notebook,
            "step_events": new_events
        }
