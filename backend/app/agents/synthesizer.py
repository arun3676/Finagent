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

import logging
from typing import List, Optional, Dict, Any
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import settings
from app.models import AgentState, RetrievedDocument, Citation, CitedResponse
from app.agents.prompts import (
    SYNTHESIZER_SYSTEM_PROMPT, 
    SYNTHESIZER_USER_TEMPLATE,
    get_synthesizer_prompt_for_length,
    get_max_tokens_for_length
)
from app.citations.utils import (
    extract_context,
    generate_preview_text,
    format_source_metadata,
    determine_validation_method
)


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
        if settings.OPENAI_API_KEY:
            self.llm = ChatOpenAI(
                model=self.model,
                api_key=settings.OPENAI_API_KEY,
                temperature=settings.LLM_TEMPERATURE,
                streaming=True
            )
        else:
            self.llm = None
    
    async def synthesize(
        self,
        query: str,
        analysis: Dict[str, Any],
        documents: List[RetrievedDocument],
        response_length: str = "normal"
    ) -> CitedResponse:
        """
        Synthesize a cited response.
        """
        try:
            if not self.llm:
                # Fallback if no LLM configured
                return CitedResponse(
                    answer=f"I couldn't generate a response for '{query}' because the LLM is not configured.",
                    citations=[],
                    sources=[]
                )

            # If no documents, return a helpful message
            if not documents:
                return CitedResponse(
                    answer=f"I couldn't find any SEC filing data to answer your question about '{query}'. Please try a different company or check if the company has public SEC filings.",
                    citations=[],
                    sources=[]
                )

            # 1. Format sources
            formatted_sources = self._format_sources(documents)
            
            # 2. Generate response
            response_text = await self._generate_response(
                query, 
                analysis, 
                formatted_sources,
                response_length
            )
            
            # 3. Extract citations
            citations = self._extract_citations(response_text, documents)
            
            from app.models import DocumentMetadata
            sources = [doc.chunk.metadata for doc in documents]
            
            return CitedResponse(
                answer=response_text,
                citations=citations,
                sources=sources
            )
            
        except Exception as e:
            logger.error(f"Synthesizer LLM failed: {e}")
            return CitedResponse(
                answer=f"I encountered an error while processing your question about '{query}'. Please try again or contact support if the issue persists.",
                citations=[],
                sources=[]
            )

    async def synthesize_for_state(self, state: AgentState) -> Dict[str, Any]:
        """
        Synthesize response and update agent state.

        LangGraph-compatible interface - returns dict with updated fields.

        Args:
            state: Current agent state

        Returns:
            Dict with draft_response and citations fields for state update
        """
        response = await self.synthesize(
            state.original_query,
            state.extracted_data,
            state.retrieved_docs,
            state.response_length.value if hasattr(state.response_length, 'value') else str(state.response_length)
        )

        return {
            "draft_response": response.answer,
            "citations": response.citations
        }
    
    async def _generate_response(
        self,
        query: str,
        analysis: Dict[str, Any],
        sources: str,
        response_length: str = "normal"
    ) -> str:
        """
        Generate response text using LLM with length-specific prompts.
        
        Args:
            query: User query
            analysis: Analyzed data
            sources: Formatted source information string
            response_length: Desired response length ("short", "normal", "detailed")
            
        Returns:
            Generated response with citation markers
        """
        # Get length-specific system prompt
        system_prompt = get_synthesizer_prompt_for_length(response_length)
        max_tokens = get_max_tokens_for_length(response_length)
        
        # Build prompt with length-specific system message
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", SYNTHESIZER_USER_TEMPLATE)
        ])
        
        # Update LLM with appropriate max_tokens
        llm_with_tokens = self.llm.bind(max_tokens=max_tokens)
        
        # Create chain
        chain = prompt | llm_with_tokens | StrOutputParser()
        
        # Generate response
        # Note: We use ainvoke here. The streaming events will be captured 
        # by the parent graph's streamEvents
        response_text = await chain.ainvoke({
            "query": query,
            "analysis": self._format_analysis(analysis),
            "sources": sources
        })
        
        return response_text
    
    def _extract_citations(
        self,
        response: str,
        documents: List[RetrievedDocument]
    ) -> List[Citation]:
        """
        Extract citations from response text with enhanced metadata.

        Finds [1], [2], etc. markers and links to sources with full context.

        Args:
            response: Response text with citation markers
            documents: Source documents

        Returns:
            List of Citation objects with enhanced fields
        """
        import re

        citations = []

        # Find all citation markers
        pattern = r'\[(\d+)\]'
        matches = re.finditer(pattern, response)

        # Track unique citations (same doc might be cited multiple times)
        seen_citations = {}

        for idx, match in enumerate(matches):
            citation_num = int(match.group(1))

            # Map to document (1-indexed)
            if 1 <= citation_num <= len(documents):
                doc = documents[citation_num - 1]

                # Check if we've already created a citation for this doc
                if citation_num in seen_citations:
                    continue

                seen_citations[citation_num] = True

                # Find the claim being cited (text before the marker)
                start = max(0, match.start() - 200)
                context_text = response[start:match.start()]

                # Extract the sentence containing the citation
                sentences = context_text.split('.')
                claim = sentences[-1].strip() if sentences else ""

                # Get source text (first 500 chars for backward compat)
                source_text = doc.chunk.content[:500]

                # Extract context with highlighting
                source_context, highlight_start, highlight_end = extract_context(
                    full_text=doc.chunk.content,
                    target_text=source_text[:200],  # Use beginning of source as target
                    context_sentences=2
                )

                # Generate preview text
                preview = generate_preview_text(source_text, max_length=50)

                # Format full source metadata
                source_meta = format_source_metadata(doc.chunk)

                # Determine validation method
                validation_method = determine_validation_method(
                    claim=claim,
                    source_text=doc.chunk.content,
                    confidence=doc.score
                )

                meta = doc.chunk.metadata
                citation = Citation(
                    citation_id=f"cite_{citation_num}",
                    citation_number=citation_num,
                    claim=claim,
                    source_chunk_id=doc.chunk.chunk_id,
                    source_document_id=doc.chunk.document_id,
                    source_text=source_text,
                    source_context=source_context,
                    highlight_start=highlight_start,
                    highlight_end=highlight_end,
                    source_metadata=source_meta,
                    confidence=doc.score,
                    validation_method=validation_method,
                    preview_text=preview,
                    # Legacy fields for backward compatibility
                    page_reference=f"{meta.document_type.value}, {doc.chunk.section}",
                    source_url=meta.source_url,
                    metadata={
                        "ticker": meta.ticker,
                        "filing_type": meta.document_type.value,
                        "section": doc.chunk.section,
                        "source_url": meta.source_url,
                    }
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
