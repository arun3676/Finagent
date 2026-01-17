"""
Follow-Up Question Generator

Generates contextual follow-up questions based on the original query,
response, and retrieved documents.
"""

import uuid
import json
import logging
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

from app.models import DocumentChunk
from app.llm.model_selector import get_llm_client, ModelTier

logger = logging.getLogger(__name__)


class FollowUpQuestion(BaseModel):
    """Single follow-up question."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique question identifier")
    text: str = Field(..., description="The follow-up question text")
    category: Literal["temporal", "deeper", "comparative", "related"] = Field(
        ..., description="Category of follow-up question"
    )
    relevant_chunk_ids: List[str] = Field(
        default_factory=list, description="Chunk IDs that can answer this question"
    )
    requires_new_retrieval: bool = Field(
        default=False, description="True for comparisons to other companies"
    )


FOLLOW_UP_GENERATION_PROMPT = """You are a financial research assistant generating follow-up questions.

Given:
- Original query: {query}
- Response summary: {response_summary}
- Companies mentioned: {companies}
- Metrics discussed: {metrics}
- Available context topics: {chunk_summaries}

Generate exactly 3 follow-up questions that:
1. Are self-contained (work without seeing original query)
2. Can mostly be answered from the same SEC filings already retrieved
3. Would genuinely help a financial analyst explore deeper
4. Are specific, not generic

Categories (generate one of each if possible):
- TEMPORAL: How has X changed over time? (trend analysis)
- DEEPER: What factors/reasons/details about X? (drill down)
- COMPARATIVE or RELATED: Compare to peer OR explore adjacent metric

Format response as JSON:
{{
  "questions": [
    {{
      "text": "How has Apple's gross margin trended from 2021 to 2023?",
      "category": "temporal",
      "can_answer_from_cache": true
    }},
    {{
      "text": "What factors does Apple cite for the margin improvement?",
      "category": "deeper",
      "can_answer_from_cache": true
    }},
    {{
      "text": "How does Apple's gross margin compare to Microsoft's?",
      "category": "comparative",
      "can_answer_from_cache": false
    }}
  ]
}}

Keep questions concise (<15 words each)."""


class FollowUpGenerator:
    """Generates contextual follow-up questions."""

    def __init__(self):
        """Initialize the generator."""
        pass

    async def generate(
        self,
        original_query: str,
        response_summary: str,
        retrieved_chunks: List[DocumentChunk],
        companies: List[str],
        metrics_mentioned: List[str]
    ) -> List[FollowUpQuestion]:
        """
        Generate 3 follow-up questions based on query context.

        CRITICAL: Questions must be answerable from cached chunks
        (except comparative questions to other companies).

        Args:
            original_query: The original user query
            response_summary: Summary of the response (first ~500 chars)
            retrieved_chunks: Chunks retrieved for the original query
            companies: List of company tickers mentioned
            metrics_mentioned: List of financial metrics discussed

        Returns:
            List of 3 FollowUpQuestion objects
        """
        logger.info(f"Generating follow-up questions for query: {original_query[:50]}...")

        # Extract chunk summaries for context
        chunk_summaries = self._extract_chunk_summaries(retrieved_chunks)

        # Format the prompt
        prompt = FOLLOW_UP_GENERATION_PROMPT.format(
            query=original_query,
            response_summary=response_summary[:500] if response_summary else "No response yet",
            companies=", ".join(companies) if companies else "Unknown",
            metrics=", ".join(metrics_mentioned) if metrics_mentioned else "General financial data",
            chunk_summaries=chunk_summaries
        )

        try:
            # Use FAST model for quick generation
            llm = get_llm_client(ModelTier.FAST)

            # Format messages for LLMClient
            messages = [
                {"role": "system", "content": "You are a financial research assistant generating follow-up questions."},
                {"role": "user", "content": prompt}
            ]

            response = await llm.generate(messages, temperature=0.3, max_tokens=500)

            # Extract JSON from response
            questions = self._parse_response(response, retrieved_chunks)

            logger.info(f"Generated {len(questions)} follow-up questions")
            return questions

        except Exception as e:
            logger.error(f"Failed to generate follow-up questions: {e}")
            # Return fallback questions
            return self._generate_fallback_questions(original_query, companies, metrics_mentioned)

    def _extract_chunk_summaries(self, chunks: List[DocumentChunk]) -> str:
        """Extract brief summaries of chunk topics."""
        if not chunks:
            return "No context available"

        summaries = []
        for i, chunk in enumerate(chunks[:5]):  # Top 5 chunks
            # Get section and first 100 chars
            section = chunk.section or "General"
            preview = chunk.content[:100].replace("\n", " ") + "..."
            summaries.append(f"- {section}: {preview}")

        return "\n".join(summaries)

    def _parse_response(
        self,
        content: str,
        retrieved_chunks: List[DocumentChunk]
    ) -> List[FollowUpQuestion]:
        """Parse LLM response into FollowUpQuestion objects."""
        try:
            # Find JSON in response
            start = content.find("{")
            end = content.rfind("}") + 1

            if start >= 0 and end > start:
                json_str = content[start:end]
                data = json.loads(json_str)

                questions = []
                chunk_ids = [c.chunk_id for c in retrieved_chunks]

                for q in data.get("questions", [])[:3]:  # Max 3 questions
                    # Map category
                    category = q.get("category", "related").lower()
                    if category not in ["temporal", "deeper", "comparative", "related"]:
                        category = "related"

                    # Determine if new retrieval needed
                    requires_new = not q.get("can_answer_from_cache", True)

                    question = FollowUpQuestion(
                        text=q.get("text", ""),
                        category=category,
                        relevant_chunk_ids=chunk_ids if not requires_new else [],
                        requires_new_retrieval=requires_new
                    )
                    questions.append(question)

                return questions

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse follow-up JSON: {e}")

        return []

    def _generate_fallback_questions(
        self,
        query: str,
        companies: List[str],
        metrics: List[str]
    ) -> List[FollowUpQuestion]:
        """Generate generic fallback questions when LLM fails."""
        company = companies[0] if companies else "the company"
        metric = metrics[0] if metrics else "these metrics"

        return [
            FollowUpQuestion(
                text=f"How has {company}'s performance in this area changed over the past 3 years?",
                category="temporal",
                relevant_chunk_ids=[],
                requires_new_retrieval=False
            ),
            FollowUpQuestion(
                text=f"What factors does {company} cite as drivers for {metric}?",
                category="deeper",
                relevant_chunk_ids=[],
                requires_new_retrieval=False
            ),
            FollowUpQuestion(
                text=f"How does this compare to industry benchmarks?",
                category="comparative",
                relevant_chunk_ids=[],
                requires_new_retrieval=True
            )
        ]


# Singleton instance for reuse
_generator_instance: Optional[FollowUpGenerator] = None


def get_follow_up_generator() -> FollowUpGenerator:
    """Get or create the singleton FollowUpGenerator instance."""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = FollowUpGenerator()
    return _generator_instance
