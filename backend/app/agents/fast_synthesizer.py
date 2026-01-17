"""
Fast Synthesizer Agent

Optimized synthesizer for SIMPLE queries that bypasses complex extraction
and multi-step reasoning. Uses a minimal prompt for fast response generation.

Target: <2 seconds total latency for simple factual queries
- Single LLM call with fast model (Gemini Flash Lite)
- Direct response generation
- Basic citations only

Usage:
    synthesizer = FastSynthesizer()
    response = await synthesizer.synthesize(query, chunks)
"""

import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from openai import AsyncOpenAI

from app.config import settings
from app.models import (
    AgentState, RetrievedDocument, Citation, AgentRole, StepEvent
)
from app.agents.prompts import (
    FAST_SYNTHESIS_PROMPT,
    get_fast_synthesis_prompt_for_length,
    get_max_tokens_for_length
)
from app.llm import get_client_for_agent, get_model_for_agent

logger = logging.getLogger(__name__)


class FastSynthesizer:
    """
    Speed-optimized synthesizer for simple factual queries.
    
    Uses fast model (Gemini Flash Lite) and minimal prompt for
    responses with basic citations.
    """
    
    def __init__(self, model: str = None):
        """
        Initialize fast synthesizer.
        
        Args:
            model: LLM model to use (defaults to fast model)
        """
        # Use fast model for speed
        self.model = model or get_model_for_agent("fast_synthesizer")
        self.llm_client = get_client_for_agent("fast_synthesizer")
        # Fallback OpenAI client
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        logger.info(f"FastSynthesizer initialized with model: {self.model}")
    
    async def synthesize(
        self,
        query: str,
        documents: List[RetrievedDocument],
        response_length: str = "normal",
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate a fast response for a simple query.
        
        Args:
            query: The user's query
            documents: Retrieved document chunks
            response_length: Desired response length ("short", "normal", "detailed")
            max_tokens: Maximum response tokens (optional, will use length-based default)
            
        Returns:
            Dict with response, citations, and metadata
        """
        logger.info(f"Fast synthesizing response for: '{query[:50]}...' (length: {response_length})")
        
        if not documents:
            return {
                "draft_response": "I couldn't find relevant information to answer your question.",
                "citations": []
            }
        
        # Get length-specific prompt and token limit
        prompt_template = get_fast_synthesis_prompt_for_length(response_length)
        if max_tokens is None:
            max_tokens = get_max_tokens_for_length(response_length)
        
        # Build context from top chunks (limit to 5 for speed)
        context_parts = []
        source_map = {}  # citation_number -> document
        
        for i, doc in enumerate(documents[:5], 1):
            chunk = doc.chunk
            context_parts.append(
                f"[{i}] {chunk.metadata.company_name} {chunk.metadata.document_type.value} "
                f"({chunk.metadata.filing_date.strftime('%Y-%m-%d')}):\n{chunk.content[:1500]}"
            )
            source_map[i] = doc
        
        context = "\n\n".join(context_parts)
        
        # Format prompt with length-specific template
        prompt = prompt_template.format(
            query=query,
            context=context
        )
        
        try:
            # Use tiered model client for fast synthesis
            try:
                answer = await self.llm_client.generate(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=max_tokens
                )
            except Exception as e:
                # Fallback to OpenAI
                logger.warning(f"Fast model failed, falling back to OpenAI: {e}")
                response = await self.client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=max_tokens
                )
                answer = response.choices[0].message.content.strip()
            
            # Extract citations from response
            citations = self._extract_citations(answer, source_map)
            
            logger.info(f"Fast synthesis complete: {len(answer)} chars, {len(citations)} citations")
            
            return {
                "draft_response": answer,
                "citations": citations
            }
            
        except Exception as e:
            logger.error(f"Fast synthesis failed: {e}")
            return {
                "draft_response": f"Error generating response: {str(e)}",
                "citations": []
            }
    
    def _extract_citations(
        self,
        response: str,
        source_map: Dict[int, RetrievedDocument]
    ) -> List[Citation]:
        """
        Extract citation objects from response text.
        
        Args:
            response: Generated response with [N] markers
            source_map: Mapping of citation numbers to documents
            
        Returns:
            List of Citation objects
        """
        import re
        
        citations = []
        seen_numbers = set()
        
        # Find all citation markers [1], [2], etc.
        citation_pattern = r'\[(\d+)\]'
        matches = re.findall(citation_pattern, response)
        
        for match in matches:
            num = int(match)
            if num in seen_numbers or num not in source_map:
                continue
            
            seen_numbers.add(num)
            doc = source_map[num]
            chunk = doc.chunk
            
            # Create citation with minimal required fields
            citation = Citation(
                citation_id=f"cite_{uuid.uuid4().hex[:8]}",
                citation_number=num,
                claim=f"Reference {num}",  # Simplified for speed
                source_chunk_id=chunk.chunk_id,
                source_document_id=chunk.document_id,
                source_text=chunk.content[:500],
                source_context=chunk.content[:800],
                highlight_start=0,
                highlight_end=min(500, len(chunk.content)),
                source_metadata={
                    "ticker": chunk.metadata.ticker,
                    "company_name": chunk.metadata.company_name,
                    "document_type": chunk.metadata.document_type.value,
                    "filing_date": chunk.metadata.filing_date.isoformat(),
                    "section": chunk.section,
                    "source_url": chunk.metadata.source_url
                },
                confidence=doc.score,
                validation_method="semantic_similarity",
                preview_text=chunk.content[:50]
            )
            
            citations.append(citation)
        
        return citations
    
    async def synthesize_for_state(self, state: AgentState) -> Dict[str, Any]:
        """
        LangGraph-compatible interface for fast synthesis.
        
        Args:
            state: Current agent state with retrieved_docs
            
        Returns:
            Dict with draft_response, citations, and step_events
        """
        new_events = list(state.step_events) if state.step_events else []
        
        # Emit synthesis start event
        start_event = StepEvent(
            event_type="step_detail",
            agent=AgentRole.SYNTHESIZER,
            timestamp=datetime.now(),
            data={
                "status": "started",
                "mode": "fast",
                "model": self.model,
                "docs_count": len(state.retrieved_docs)
            }
        )
        new_events.append(start_event)
        
        # Run fast synthesis
        result = await self.synthesize(
            query=state.original_query,
            documents=state.retrieved_docs,
            response_length=state.response_length.value if hasattr(state.response_length, 'value') else str(state.response_length)
        )
        
        # Emit completion event
        complete_event = StepEvent(
            event_type="step_detail",
            agent=AgentRole.SYNTHESIZER,
            timestamp=datetime.now(),
            data={
                "status": "completed",
                "mode": "fast",
                "citations_count": len(result.get("citations", []))
            }
        )
        new_events.append(complete_event)
        
        return {
            "draft_response": result["draft_response"],
            "citations": result["citations"],
            "step_events": new_events,
            "is_valid": True,  # Skip validation for fast path
            "iteration_count": 1
        }
