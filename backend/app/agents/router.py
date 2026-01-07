"""
Query Router Agent

Classifies incoming queries by complexity to determine the appropriate
processing pipeline.

Complexity levels:
- SIMPLE: Direct retrieval + answer
- MODERATE: Multi-step retrieval + analysis
- COMPLEX: Full multi-agent pipeline with planning

Usage:
    router = QueryRouter()
    complexity = await router.classify(query)
"""

from typing import Optional, Dict, Any
import logging
import json
from openai import AsyncOpenAI

from app.config import settings
from app.models import QueryComplexity, AgentState
from app.agents.prompts import ROUTER_SYSTEM_PROMPT, ROUTER_USER_TEMPLATE

logger = logging.getLogger(__name__)


class QueryRouter:
    """
    Query complexity classifier.
    
    Uses LLM to classify queries, with fallback heuristics
    for common patterns.
    """
    
    # Heuristic patterns for quick classification
    SIMPLE_PATTERNS = [
        "what is", "what was", "who is", "when did",
        "how much", "how many"
    ]
    
    COMPLEX_PATTERNS = [
        "analyze", "compare", "trend", "impact",
        "risk", "strategy", "outlook", "forecast"
    ]
    
    def __init__(
        self,
        use_llm: bool = True,
        model: str = None
    ):
        """
        Initialize query router.
        
        Args:
            use_llm: Use LLM for classification (vs heuristics only)
            model: LLM model to use
        """
        self.use_llm = use_llm
        self.model = model or settings.LLM_MODEL
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    
    async def classify(self, query: str) -> QueryComplexity:
        """
        Classify query complexity.
        
        Args:
            query: User's financial research query
            
        Returns:
            QueryComplexity enum value
        """
        # Try heuristics first for speed
        heuristic_result = self._heuristic_classify(query)
        if heuristic_result and not self.use_llm:
            return heuristic_result
        
        # Use LLM for more nuanced classification
        if self.use_llm:
            return await self._llm_classify(query)
        
        return heuristic_result or QueryComplexity.MODERATE
    
    async def route(self, state: AgentState) -> AgentState:
        """
        Route query and update agent state.
        
        This is the LangGraph-compatible interface.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with complexity classification
        """
        complexity = await self.classify(state.original_query)
        state.complexity = complexity
        return state
    
    async def _llm_classify(self, query: str) -> QueryComplexity:
        """
        Use LLM to classify query complexity.
        
        Args:
            query: Query to classify
            
        Returns:
            QueryComplexity enum value
        """
        try:
            messages = [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": ROUTER_USER_TEMPLATE.format(query=query)}
            ]
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=50,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            complexity_str = result.get("complexity", "MODERATE").upper()
            
            complexity_map = {
                "SIMPLE": QueryComplexity.SIMPLE,
                "MODERATE": QueryComplexity.MODERATE,
                "COMPLEX": QueryComplexity.COMPLEX
            }
            
            complexity = complexity_map.get(complexity_str, QueryComplexity.MODERATE)
            logger.info(f"Classified query as {complexity.value}: '{query[:50]}...'")
            
            return complexity
            
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return self._heuristic_classify(query) or QueryComplexity.MODERATE
    
    def _heuristic_classify(self, query: str) -> Optional[QueryComplexity]:
        """
        Use heuristics for quick classification.
        
        Args:
            query: Query to classify
            
        Returns:
            QueryComplexity or None if uncertain
        """
        query_lower = query.lower()
        
        # Check for complex patterns
        complex_matches = sum(1 for p in self.COMPLEX_PATTERNS if p in query_lower)
        if complex_matches >= 2:
            return QueryComplexity.COMPLEX
        
        # Check for simple patterns
        simple_matches = sum(1 for p in self.SIMPLE_PATTERNS if query_lower.startswith(p))
        if simple_matches > 0 and complex_matches == 0:
            # Additional check: short queries are usually simple
            if len(query.split()) <= 10:
                return QueryComplexity.SIMPLE
        
        # Check query length and structure
        word_count = len(query.split())
        if word_count <= 8:
            return QueryComplexity.SIMPLE
        elif word_count >= 25:
            return QueryComplexity.COMPLEX
        
        return None  # Uncertain, use LLM
    
    def get_pipeline_for_complexity(
        self,
        complexity: QueryComplexity
    ) -> list:
        """
        Get the agent pipeline for a complexity level.
        
        Args:
            complexity: Query complexity
            
        Returns:
            List of agent names in execution order
        """
        pipelines = {
            QueryComplexity.SIMPLE: [
                "retriever",
                "synthesizer"
            ],
            QueryComplexity.MODERATE: [
                "retriever",
                "analyst",
                "synthesizer",
                "validator"
            ],
            QueryComplexity.COMPLEX: [
                "planner",
                "retriever",
                "analyst",
                "synthesizer",
                "validator"
            ]
        }
        return pipelines.get(complexity, pipelines[QueryComplexity.MODERATE])
