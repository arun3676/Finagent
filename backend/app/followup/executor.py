"""
Follow-Up Question Executor

Executes follow-up questions using cached context for fast response times.
Target: <1.5 seconds for cached path.
"""

import time
import logging
from typing import List, Optional
from pydantic import BaseModel, Field

from app.models import Citation, DocumentChunk
from app.followup.generator import FollowUpQuestion
from app.followup.cache import ChunkCache, get_chunk_cache
from app.llm.model_selector import get_llm_client, ModelTier
from app.agents.prompts import FOLLOW_UP_SYNTHESIZER_PROMPT

logger = logging.getLogger(__name__)


class FollowUpResponse(BaseModel):
    """Response from executing a follow-up question."""
    question: str = Field(..., description="The follow-up question that was answered")
    answer: str = Field(..., description="Short answer (3-5 sentences)")
    citations: List[Citation] = Field(default_factory=list, description="Supporting citations")
    execution_time_ms: int = Field(..., description="Total execution time in milliseconds")
    used_cache: bool = Field(..., description="Whether cached chunks were used")
    parent_query_id: str = Field(default="", description="ID of the parent query")


class FollowUpExecutor:
    """
    Execute follow-up questions using cached context.

    Fast execution path:
    1. Get cached chunks (no retrieval!)
    2. Filter to relevant chunks
    3. Generate SHORT response (3-5 sentences)

    Target: <1.5 seconds total
    """

    def __init__(
        self,
        cache: Optional[ChunkCache] = None
    ):
        """
        Initialize the executor.

        Args:
            cache: ChunkCache instance for retrieving cached chunks
        """
        self.cache = cache or get_chunk_cache()

    async def execute(
        self,
        follow_up_question: FollowUpQuestion,
        parent_query_id: str
    ) -> FollowUpResponse:
        """
        Fast execution path for follow-up questions.

        Args:
            follow_up_question: The follow-up question to answer
            parent_query_id: ID of the parent query (for cache lookup)

        Returns:
            FollowUpResponse with answer and citations
        """
        start_time = time.time()

        logger.info(f"Executing follow-up: {follow_up_question.text[:50]}...")

        # Get cached chunks
        cache_entry = await self.cache.get(parent_query_id)

        if not cache_entry:
            # Cache expired or not found - need minimal retrieval
            logger.warning(f"Cache miss for query {parent_query_id}, using minimal retrieval")
            return await self._execute_with_retrieval(follow_up_question, start_time)

        # Get relevant chunks (use all if no specific IDs)
        if follow_up_question.relevant_chunk_ids:
            relevant_chunks = await self.cache.get_chunks_by_ids(
                parent_query_id,
                follow_up_question.relevant_chunk_ids
            )
        else:
            relevant_chunks = cache_entry.chunks

        if not relevant_chunks:
            logger.warning("No relevant chunks found in cache")
            return await self._execute_with_retrieval(follow_up_question, start_time)

        # Generate response using cached context
        answer, citations = await self._generate_response(
            follow_up_question.text,
            relevant_chunks
        )

        execution_time_ms = int((time.time() - start_time) * 1000)

        logger.info(f"Follow-up executed in {execution_time_ms}ms (cached path)")

        return FollowUpResponse(
            question=follow_up_question.text,
            answer=answer,
            citations=citations,
            execution_time_ms=execution_time_ms,
            used_cache=True,
            parent_query_id=parent_query_id
        )

    async def _generate_response(
        self,
        question: str,
        chunks: List[DocumentChunk]
    ) -> tuple:
        """
        Generate a short response using the follow-up synthesizer prompt.

        Args:
            question: The follow-up question
            chunks: Relevant document chunks

        Returns:
            Tuple of (answer_text, citations_list)
        """
        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks[:5], 1):  # Top 5 chunks max
            ticker = chunk.metadata.ticker if chunk.metadata else "Unknown"
            section = chunk.section or "General"
            context_parts.append(f"[{i}] ({ticker} - {section}):\n{chunk.content[:500]}")

        context = "\n\n".join(context_parts)

        # Format prompt
        prompt = FOLLOW_UP_SYNTHESIZER_PROMPT.format(
            context=context,
            question=question
        )

        try:
            # Use FAST model for speed (target <1s for LLM call)
            llm = get_llm_client(ModelTier.FAST)

            # Format messages for LLMClient
            messages = [
                {"role": "system", "content": "You are a financial research assistant providing concise answers."},
                {"role": "user", "content": prompt}
            ]

            answer = await llm.generate(messages, temperature=0.1, max_tokens=300)

            # Extract citations from response
            citations = self._extract_citations(answer, chunks)

            return answer, citations

        except Exception as e:
            logger.error(f"Failed to generate follow-up response: {e}")
            return f"Unable to generate response: {str(e)}", []

    def _extract_citations(
        self,
        answer: str,
        chunks: List[DocumentChunk]
    ) -> List[Citation]:
        """
        Extract citations from the answer text.

        Args:
            answer: The generated answer with [1], [2] markers
            chunks: Source chunks used for context

        Returns:
            List of Citation objects
        """
        import re
        import uuid

        citations = []
        citation_pattern = r'\[(\d+)\]'

        # Find all citation markers in the answer
        found_numbers = set(int(m) for m in re.findall(citation_pattern, answer))

        for num in sorted(found_numbers):
            chunk_idx = num - 1
            if 0 <= chunk_idx < len(chunks):
                chunk = chunks[chunk_idx]

                citation = Citation(
                    citation_id=f"followup_cite_{uuid.uuid4().hex[:8]}",
                    citation_number=num,
                    claim=f"Reference {num} from follow-up response",
                    source_chunk_id=chunk.chunk_id,
                    source_document_id=chunk.document_id,
                    source_text=chunk.content[:200],
                    source_context=chunk.content[:500],
                    source_metadata={
                        "ticker": chunk.metadata.ticker if chunk.metadata else "",
                        "section": chunk.section or "",
                        "document_type": chunk.metadata.document_type.value if chunk.metadata else ""
                    },
                    confidence=0.85,  # Slightly lower confidence for follow-ups
                    preview_text=chunk.content[:50]
                )
                citations.append(citation)

        return citations

    async def _execute_with_retrieval(
        self,
        follow_up_question: FollowUpQuestion,
        start_time: float
    ) -> FollowUpResponse:
        """
        Execute follow-up with minimal retrieval when cache misses.

        This is the fallback path when cached chunks are not available.

        Args:
            follow_up_question: The follow-up question
            start_time: Execution start time

        Returns:
            FollowUpResponse with answer
        """
        logger.info("Executing follow-up with minimal retrieval (cache miss)")

        try:
            # Import here to avoid circular imports
            from app.agents.retriever_agent import RetrieverAgent

            retriever = RetrieverAgent()

            # Do minimal retrieval (fewer docs for speed)
            docs = await retriever.fast_retrieve(
                query=follow_up_question.text,
                top_k=3  # Minimal retrieval
            )

            # Convert RetrievedDocument to DocumentChunk
            chunks = [doc.chunk for doc in docs]

            # Generate response
            answer, citations = await self._generate_response(
                follow_up_question.text,
                chunks
            )

            execution_time_ms = int((time.time() - start_time) * 1000)

            return FollowUpResponse(
                question=follow_up_question.text,
                answer=answer,
                citations=citations,
                execution_time_ms=execution_time_ms,
                used_cache=False,
                parent_query_id=""
            )

        except Exception as e:
            logger.error(f"Fallback retrieval failed: {e}")

            execution_time_ms = int((time.time() - start_time) * 1000)

            return FollowUpResponse(
                question=follow_up_question.text,
                answer=f"Unable to answer follow-up question: {str(e)}",
                citations=[],
                execution_time_ms=execution_time_ms,
                used_cache=False,
                parent_query_id=""
            )

    async def execute_batch(
        self,
        questions: List[FollowUpQuestion],
        parent_query_id: str
    ) -> List[FollowUpResponse]:
        """
        Execute multiple follow-up questions (useful for pre-computing).

        Args:
            questions: List of follow-up questions
            parent_query_id: ID of the parent query

        Returns:
            List of FollowUpResponse objects
        """
        import asyncio

        tasks = [
            self.execute(q, parent_query_id)
            for q in questions
        ]

        return await asyncio.gather(*tasks)


# Singleton instance
_executor_instance: Optional[FollowUpExecutor] = None


def get_follow_up_executor() -> FollowUpExecutor:
    """Get or create the singleton FollowUpExecutor instance."""
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = FollowUpExecutor()
    return _executor_instance
