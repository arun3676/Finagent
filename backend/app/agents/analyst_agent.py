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
from app.models import AgentState, RetrievedDocument
from app.agents.prompts import ANALYST_SYSTEM_PROMPT, ANALYST_USER_TEMPLATE


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
        # TODO: Implement analysis pipeline
        # 1. Extract relevant data points
        # 2. Perform calculations if needed
        # 3. Identify trends
        # 4. Return structured analysis
        raise NotImplementedError("Analysis pipeline not yet implemented")
    
    async def analyze_for_state(self, state: AgentState) -> AgentState:
        """
        Analyze documents and update agent state.
        
        LangGraph-compatible interface.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with analysis results
        """
        analysis = await self.analyze(
            state.original_query,
            state.retrieved_docs
        )
        state.extracted_data = analysis
        return state
    
    async def extract_data_points(
        self,
        documents: List[RetrievedDocument],
        data_types: List[str]
    ) -> Dict[str, Any]:
        """
        Extract specific data points from documents.
        
        Args:
            documents: Source documents
            data_types: Types of data to extract (e.g., "revenue", "eps")
            
        Returns:
            Extracted data with sources
        """
        # TODO: Implement data extraction
        # 1. For each data type, search documents
        # 2. Extract values with context
        # 3. Normalize units
        # 4. Return with source citations
        raise NotImplementedError("Data extraction not yet implemented")
    
    async def calculate_metrics(
        self,
        data: Dict[str, Any],
        metrics: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate financial metrics from extracted data.
        
        Supported metrics:
        - growth_rate: YoY or QoQ growth
        - margin: Gross, operating, net margins
        - ratio: Financial ratios
        - comparison: Compare across companies/periods
        
        Args:
            data: Extracted data points
            metrics: Metrics to calculate
            
        Returns:
            Calculated metrics with methodology
        """
        # TODO: Implement metric calculations
        # 1. Validate required data exists
        # 2. Perform calculations
        # 3. Include calculation methodology
        raise NotImplementedError("Metric calculation not yet implemented")
    
    def _calculate_growth_rate(
        self,
        current: Decimal,
        previous: Decimal
    ) -> Optional[float]:
        """
        Calculate growth rate between two values.
        
        Args:
            current: Current period value
            previous: Previous period value
            
        Returns:
            Growth rate as percentage, or None if invalid
        """
        if previous == 0:
            return None
        return float((current - previous) / previous * 100)
    
    def _calculate_margin(
        self,
        numerator: Decimal,
        denominator: Decimal
    ) -> Optional[float]:
        """
        Calculate margin/ratio.
        
        Args:
            numerator: Top value (e.g., gross profit)
            denominator: Bottom value (e.g., revenue)
            
        Returns:
            Margin as percentage, or None if invalid
        """
        if denominator == 0:
            return None
        return float(numerator / denominator * 100)
    
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
        # TODO: Implement LLM call
        # 1. Format documents for context
        # 2. Build prompt
        # 3. Call LLM
        # 4. Return response
        raise NotImplementedError("LLM extraction not yet implemented")
    
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
        # TODO: Implement validation
        # 1. For each extracted value
        # 2. Check if it appears in cited source
        # 3. Return validation status
        raise NotImplementedError("Extraction validation not yet implemented")
