"""
Synthesizer Agent

Generates final responses with proper citations from analyzed data.
Produces professional financial research output.

Output format:
- Direct answer to the question
- Supporting evidence with inline citations
- Relevant context and caveats
- Professional financial language

Usage:
    synthesizer = Synthesizer()
    response = await synthesizer.synthesize(query, analysis, sources)
"""

from typing import List, Optional, Dict, Any
from openai import AsyncOpenAI

from app.config import settings
from app.models import AgentState, RetrievedDocument, Citation, CitedResponse
from app.agents.prompts import SYNTHESIZER_SYSTEM_PROMPT, SYNTHESIZER_USER_TEMPLATE


class Synthesizer:
    """
    Response synthesis agent.
    
    Generates well-structured, cited responses from
    analyzed financial data.
    """
    
    def __init__(self, model: str = None):
        """
        Initialize synthesizer.
        
        Args:
            model: LLM model to use
        """
        self.model = model or settings.LLM_MODEL
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    
    async def synthesize(
        self,
        query: str,
        analysis: Dict[str, Any],
        documents: List[RetrievedDocument]
    ) -> CitedResponse:
        """
        Synthesize a cited response.
        
        Args:
            query: Original user query
            analysis: Analyzed data from analyst agent
            documents: Source documents
            
        Returns:
            CitedResponse with answer and citations
        """
        # TODO: Implement synthesis pipeline
        # 1. Format analysis and sources
        # 2. Call LLM for response generation
        # 3. Extract and link citations
        # 4. Return CitedResponse
        raise NotImplementedError("Synthesis pipeline not yet implemented")
    
    async def synthesize_for_state(self, state: AgentState) -> AgentState:
        """
        Synthesize response and update agent state.
        
        LangGraph-compatible interface.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with draft response and citations
        """
        response = await self.synthesize(
            state.original_query,
            state.extracted_data,
            state.retrieved_docs
        )
        
        state.draft_response = response.answer
        state.citations = response.citations
        return state
    
    async def _generate_response(
        self,
        query: str,
        analysis: Dict[str, Any],
        sources: List[Dict[str, Any]]
    ) -> str:
        """
        Generate response text using LLM.
        
        Args:
            query: User query
            analysis: Analyzed data
            sources: Formatted source information
            
        Returns:
            Generated response with citation markers
        """
        # TODO: Implement LLM response generation
        # 1. Format prompt
        # 2. Call LLM
        # 3. Return response text
        raise NotImplementedError("Response generation not yet implemented")
    
    def _extract_citations(
        self,
        response: str,
        documents: List[RetrievedDocument]
    ) -> List[Citation]:
        """
        Extract citations from response text.
        
        Finds [1], [2], etc. markers and links to sources.
        
        Args:
            response: Response text with citation markers
            documents: Source documents
            
        Returns:
            List of Citation objects
        """
        import re
        
        citations = []
        
        # Find all citation markers
        pattern = r'\[(\d+)\]'
        matches = re.finditer(pattern, response)
        
        for match in matches:
            citation_num = int(match.group(1))
            
            # Map to document (1-indexed)
            if 1 <= citation_num <= len(documents):
                doc = documents[citation_num - 1]
                
                # Find the claim being cited (text before the marker)
                start = max(0, match.start() - 200)
                context = response[start:match.start()]
                
                # Extract the sentence containing the citation
                sentences = context.split('.')
                claim = sentences[-1].strip() if sentences else ""
                
                citation = Citation(
                    citation_id=f"cite_{citation_num}",
                    claim=claim,
                    source_chunk_id=doc.chunk.chunk_id,
                    source_text=doc.chunk.content[:500],
                    confidence=doc.score,
                    page_reference=f"{doc.chunk.metadata.document_type.value}, {doc.chunk.section}"
                )
                citations.append(citation)
        
        return citations
    
    def _format_sources(
        self,
        documents: List[RetrievedDocument]
    ) -> str:
        """
        Format source documents for prompt.
        
        Args:
            documents: Source documents
            
        Returns:
            Formatted source string
        """
        formatted = []
        
        for i, doc in enumerate(documents, 1):
            meta = doc.chunk.metadata
            source_info = (
                f"[{i}] {meta.company_name} ({meta.ticker}) - "
                f"{meta.document_type.value}, {meta.filing_date.strftime('%Y-%m-%d')}\n"
                f"Section: {doc.chunk.section or 'N/A'}\n"
                f"Content: {doc.chunk.content[:500]}..."
            )
            formatted.append(source_info)
        
        return "\n\n".join(formatted)
    
    def _format_analysis(self, analysis: Dict[str, Any]) -> str:
        """
        Format analysis data for prompt.
        
        Args:
            analysis: Analyzed data
            
        Returns:
            Formatted analysis string
        """
        if not analysis:
            return "No specific analysis data available."
        
        lines = []
        for key, value in analysis.items():
            if isinstance(value, dict):
                lines.append(f"{key}:")
                for k, v in value.items():
                    lines.append(f"  - {k}: {v}")
            else:
                lines.append(f"{key}: {value}")
        
        return "\n".join(lines)
    
    def estimate_response_quality(
        self,
        response: CitedResponse
    ) -> Dict[str, Any]:
        """
        Estimate quality metrics for a response.
        
        Args:
            response: Generated response
            
        Returns:
            Quality metrics
        """
        return {
            "answer_length": len(response.answer),
            "num_citations": len(response.citations),
            "num_sources": len(response.sources),
            "avg_citation_confidence": (
                sum(c.confidence for c in response.citations) / len(response.citations)
                if response.citations else 0
            ),
            "has_direct_answer": not response.answer.startswith("I"),
        }
