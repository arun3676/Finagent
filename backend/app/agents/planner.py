"""
Query Planner Agent

Decomposes complex financial queries into simpler sub-queries
that can be answered independently and combined.

Planning strategy:
1. Identify information needs
2. Determine document requirements
3. Order sub-queries by dependency
4. Assign priorities

Usage:
    planner = QueryPlanner()
    sub_queries = await planner.decompose(query, context)
"""

import json
from typing import List, Optional, Dict, Any
from openai import AsyncOpenAI

from app.config import settings
from app.models import AgentState, SubQuery, DocumentType, AgentRole, StepEvent
from app.agents.prompts import (
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_TEMPLATE,
    PLANNER_FEW_SHOT
)
from app.utils.temporal import extract_temporal_constraints, merge_temporal_filters
from datetime import datetime


class QueryPlanner:
    """
    Query decomposition agent.
    
    Breaks complex queries into manageable sub-queries
    with clear information needs and document requirements.
    """
    
    def __init__(self, model: str = None):
        """
        Initialize query planner.
        
        Args:
            model: LLM model to use
        """
        self.model = model or settings.LLM_MODEL
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    
    async def decompose(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[SubQuery]:
        """
        Decompose a complex query into sub-queries.

        Args:
            query: Complex financial query
            context: Additional context (ticker, date range, etc.)

        Returns:
            List of SubQuery objects
        """
        context_str = json.dumps(context or {}, indent=2)
        
        try:
            response = await self._call_llm(query, context_str)
            sub_queries = self._parse_sub_queries(response)
            
            if not sub_queries:
                raise ValueError("No sub-queries parsed")
                
            return sub_queries
            
        except Exception as e:
            # Fallback to single query if decomposition fails
            return [
                SubQuery(
                    sub_query=query,
                    intent="Answer the query",
                    required_docs=[DocumentType.SEC_10K],
                    priority=1
                )
            ]

    async def plan(self, state: AgentState) -> Dict[str, Any]:
        """
        Plan query execution and update state.

        LangGraph-compatible interface - returns dict with updated fields.

        Args:
            state: Current agent state

        Returns:
            Dict with sub_queries field for state update
        """
        temporal_constraints = extract_temporal_constraints(state.original_query)
        merged_filters = merge_temporal_filters(state.filters, temporal_constraints)

        context = {
            "filters": merged_filters,
            "complexity": state.complexity.value if state.complexity else None,
            "temporal_constraints": temporal_constraints.fiscal_period
        }

        sub_queries = await self.decompose(state.original_query, context)

        # Create step events for each sub-query
        new_events = list(state.step_events) if state.step_events else []

        # Overall planning event
        plan_event = StepEvent(
            event_type="step_detail",
            agent=AgentRole.PLANNER,
            timestamp=datetime.now(),
            data={
                "action": "query_decomposition",
                "total_sub_queries": len(sub_queries),
                "message": f"Decomposed into {len(sub_queries)} sub-queries"
            }
        )
        new_events.append(plan_event)

        # Event for each sub-query
        for idx, sq in enumerate(sub_queries, 1):
            sub_query_event = StepEvent(
                event_type="sub_query",
                agent=AgentRole.PLANNER,
                timestamp=datetime.now(),
                data={
                    "index": idx,
                    "total": len(sub_queries),
                    "query": sq.sub_query,
                    "intent": sq.intent,
                    "priority": sq.priority,
                    "required_docs": [dt.value for dt in sq.required_docs]
                }
            )
            new_events.append(sub_query_event)

        return {
            "sub_queries": sub_queries,
            "filters": merged_filters if merged_filters else state.filters,
            "step_events": new_events
        }
    
    async def _call_llm(
        self,
        query: str,
        context: str
    ) -> str:
        """
        Call LLM for query decomposition.
        
        Args:
            query: Query to decompose
            context: Additional context string
            
        Returns:
            LLM response text
        """
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": PLANNER_FEW_SHOT},
            {"role": "user", "content": PLANNER_USER_TEMPLATE.format(
                query=query,
                context=context
            )}
        ]
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        return response.choices[0].message.content
    
    def _parse_sub_queries(self, response: str) -> List[SubQuery]:
        """
        Parse LLM response into SubQuery objects.
        
        Args:
            response: LLM response (expected JSON)
            
        Returns:
            List of SubQuery objects
        """
        try:
            # Extract JSON from response
            json_str = self._extract_json(response)
            data = json.loads(json_str)
            
            sub_queries = []
            for item in data:
                # Map document type strings to enums
                doc_types = [
                    self._map_doc_type(dt) 
                    for dt in item.get("required_docs", [])
                ]
                
                sub_query = SubQuery(
                    sub_query=item["sub_query"],
                    intent=item["intent"],
                    required_docs=doc_types,
                    priority=item.get("priority", 1)
                )
                sub_queries.append(sub_query)
            
            # Sort by priority
            sub_queries.sort(key=lambda x: x.priority)
            return sub_queries
            
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback: return original query as single sub-query
            return [SubQuery(
                sub_query=response,
                intent="Answer the query",
                required_docs=[DocumentType.SEC_10K],
                priority=1
            )]
    
    def _extract_json(self, text: str) -> str:
        """
        Extract JSON array from text response.
        
        Args:
            text: Text potentially containing JSON
            
        Returns:
            JSON string
        """
        # Find JSON array in response
        start = text.find('[')
        end = text.rfind(']') + 1
        
        if start != -1 and end > start:
            return text[start:end]
        
        return text
    
    def _map_doc_type(self, doc_type_str: str) -> DocumentType:
        """
        Map document type string to enum.
        
        Args:
            doc_type_str: Document type string
            
        Returns:
            DocumentType enum value
        """
        mapping = {
            "10-K": DocumentType.SEC_10K,
            "10-Q": DocumentType.SEC_10Q,
            "8-K": DocumentType.SEC_8K,
            "earnings_call": DocumentType.EARNINGS_CALL,
            "press_release": DocumentType.PRESS_RELEASE
        }
        return mapping.get(doc_type_str, DocumentType.SEC_10K)
    
    def estimate_complexity(self, sub_queries: List[SubQuery]) -> Dict[str, Any]:
        """
        Estimate execution complexity from sub-queries.
        
        Args:
            sub_queries: List of planned sub-queries
            
        Returns:
            Complexity estimation
        """
        doc_types = set()
        for sq in sub_queries:
            doc_types.update(sq.required_docs)
        
        return {
            "num_sub_queries": len(sub_queries),
            "document_types_needed": len(doc_types),
            "estimated_retrievals": len(sub_queries) * 2,  # Assume 2 retrievals per sub-query
            "priority_levels": len(set(sq.priority for sq in sub_queries))
        }
