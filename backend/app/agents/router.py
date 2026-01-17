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
from app.models import QueryComplexity, AgentState, AgentRole, StepEvent
from app.agents.prompts import ROUTER_SYSTEM_PROMPT, ROUTER_USER_TEMPLATE
from app.llm import get_client_for_agent, get_model_for_agent
from datetime import datetime

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
            model: LLM model to use (defaults to fast model for router)
        """
        self.use_llm = use_llm
        # Router uses fast model by default (classification is simple)
        self.model = model or get_model_for_agent("router")
        self.llm_client = get_client_for_agent("router")
        # Keep OpenAI client as fallback
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        logger.info(f"Router initialized with model: {self.model}")
    
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
        if self.use_llm and settings.OPENAI_API_KEY:
            return await self._llm_classify(query)
        
        return heuristic_result or QueryComplexity.SIMPLE
    
    def _build_complexity_info(self, complexity: QueryComplexity, query: str) -> "ComplexityInfo":
        """
        Build user-facing complexity info for frontend display.

        Args:
            complexity: Classified complexity level
            query: Original query for reasoning

        Returns:
            ComplexityInfo with display metadata
        """
        from app.models import ComplexityInfo

        # Configuration for each complexity level
        COMPLEXITY_CONFIG = {
            QueryComplexity.SIMPLE: {
                "display_label": "Quick Look",
                "display_color": "green",
                "estimated_time_seconds": 3,
                "features_enabled": ["retriever", "synthesizer"]
            },
            QueryComplexity.MODERATE: {
                "display_label": "Analysis",
                "display_color": "blue",
                "estimated_time_seconds": 10,
                "features_enabled": ["retriever", "analyst", "synthesizer", "validator"]
            },
            QueryComplexity.COMPLEX: {
                "display_label": "Deep Research",
                "display_color": "purple",
                "estimated_time_seconds": 25,
                "features_enabled": ["planner", "retriever", "analyst", "synthesizer", "validator"]
            }
        }

        config = COMPLEXITY_CONFIG[complexity]

        # Build reasoning text based on query patterns
        reasoning = self._generate_reasoning(complexity, query)

        return ComplexityInfo(
            level=complexity,
            display_label=config["display_label"],
            display_color=config["display_color"],
            estimated_time_seconds=config["estimated_time_seconds"],
            reasoning=reasoning,
            features_enabled=config["features_enabled"]
        )

    def _generate_reasoning(self, complexity: QueryComplexity, query: str) -> str:
        """Generate user-friendly reasoning for complexity classification."""
        query_lower = query.lower()

        if complexity == QueryComplexity.SIMPLE:
            if any(word in query_lower for word in ["what is", "how much", "who is"]):
                return "Single fact lookup"
            return "Direct information retrieval"

        elif complexity == QueryComplexity.MODERATE:
            if "compare" in query_lower or "vs" in query_lower:
                return "Comparison analysis required"
            if any(word in query_lower for word in ["trend", "growth", "change"]):
                return "Temporal analysis needed"
            return "Multi-step reasoning required"

        else:  # COMPLEX
            if "compare" in query_lower and "analyze" in query_lower:
                return "Multi-company comparison with deep analysis"
            if "impact" in query_lower or "risk" in query_lower:
                return "Complex risk and impact assessment"
            return "Multi-document analysis with calculations"

    async def route(self, state: AgentState) -> Dict[str, Any]:
        """
        Route query and update agent state.

        This is the LangGraph-compatible interface.

        Args:
            state: Current agent state

        Returns:
            Dict with complexity and complexity_info fields for state update
        """
        complexity = await self.classify(state.original_query)
        complexity_info = self._build_complexity_info(complexity, state.original_query)

        # Create step event for streaming
        step_event = StepEvent(
            event_type="step_detail",
            agent=AgentRole.ROUTER,
            timestamp=datetime.now(),
            data={
                "complexity": complexity.value,
                "reasoning": complexity_info.reasoning,
                "estimated_time": complexity_info.estimated_time_seconds,
                "agents_enabled": complexity_info.features_enabled
            }
        )

        # Append to existing events
        new_events = list(state.step_events) if state.step_events else []
        new_events.append(step_event)

        return {
            "complexity": complexity,
            "complexity_info": complexity_info,
            "step_events": new_events
        }
    
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

            # Use tiered model client for faster classification
            try:
                content = await self.llm_client.generate(
                    messages=messages,
                    temperature=0.0,
                    max_tokens=256
                )
            except Exception as e:
                # Fallback to OpenAI client if unified client fails
                logger.warning(f"Tiered model failed, falling back to OpenAI: {e}")
                response = await self.client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=256,
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content

            # Validate response content
            if not content:
                raise ValueError("Empty response from LLM")

            # Strip whitespace and parse JSON
            content = content.strip()
            logger.debug(f"Router LLM response: {content}")

            try:
                result = json.loads(content)
            except json.JSONDecodeError as jde:
                logger.warning(f"Initial JSON parse failed: {jde}. Trying regex fallback.")
                # Try to find JSON object in the text if it's wrapped in something
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(0))
                    except json.JSONDecodeError as jde2:
                        logger.error(f"Regex fallback JSON parse failed: {jde2}")
                        raise
                else:
                    logger.error("No JSON object found in content")
                    raise

            if not isinstance(result, dict):
                logger.error(f"LLM returned non-dict JSON: {type(result)} - {result}")
                return self._heuristic_classify(query) or QueryComplexity.MODERATE

            complexity_str = result.get("complexity", "MODERATE")
            if isinstance(complexity_str, str):
                complexity_str = complexity_str.upper()
            else:
                complexity_str = "MODERATE"

            complexity_map = {
                "SIMPLE": QueryComplexity.SIMPLE,
                "MODERATE": QueryComplexity.MODERATE,
                "COMPLEX": QueryComplexity.COMPLEX
            }

            complexity = complexity_map.get(complexity_str, QueryComplexity.MODERATE)
            logger.info(f"Classified query as {complexity.value}: '{query[:50]}...'")

            return complexity

        except json.JSONDecodeError as e:
            logger.error(f"LLM classification JSON parse error: {e}. Content: {content if 'content' in locals() else 'N/A'}")
            return self._heuristic_classify(query) or QueryComplexity.MODERATE
        except Exception as e:
            logger.error(f"LLM classification failed: {type(e).__name__}: {e}. Content: {content if 'content' in locals() else 'N/A'}")
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

        # Check for complex patterns first
        complex_matches = sum(1 for p in self.COMPLEX_PATTERNS if p in query_lower)
        if complex_matches >= 2:
            return QueryComplexity.COMPLEX

        # Check for comparison patterns (moderate complexity)
        comparison_indicators = ["compare", "vs", "versus", "difference between", "better than"]
        has_comparison = any(indicator in query_lower for indicator in comparison_indicators)

        # Check for trend/temporal analysis (moderate complexity)
        temporal_indicators = ["trend", "growth", "over time", "change", "history", "past", "years"]
        has_temporal = any(indicator in query_lower for indicator in temporal_indicators)

        # If has comparison or temporal analysis, classify as MODERATE
        if has_comparison or has_temporal:
            # Only escalate to COMPLEX if it has multiple complex patterns
            # or is very long (>20 words)
            word_count = len(query.split())
            if complex_matches >= 2 or word_count > 20:
                return QueryComplexity.COMPLEX
            return QueryComplexity.MODERATE

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
