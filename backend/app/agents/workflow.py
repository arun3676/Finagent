"""
LangGraph Workflow

Orchestrates the multi-agent pipeline using LangGraph state machine.
Defines the flow between agents based on query complexity.

Workflow structure:
1. Router -> Classify complexity
2. Planner -> Decompose (if complex)
3. Retriever -> Fetch documents
4. Analyst -> Extract and analyze
5. Synthesizer -> Generate response
6. Validator -> Quality check
7. Loop back if validation fails

Usage:
    workflow = FinAgentWorkflow()
    result = await workflow.run(query)
"""

from typing import Dict, Any, Optional, Literal
import logging
import time
from langgraph.graph import StateGraph, END

from app.config import settings
from app.models import AgentState, QueryComplexity, QueryResponse
from app.agents.router import QueryRouter
from app.agents.planner import QueryPlanner
from app.agents.retriever_agent import RetrieverAgent
from app.agents.analyst_agent import AnalystAgent
from app.agents.synthesizer import Synthesizer
from app.agents.validator import Validator

logger = logging.getLogger(__name__)


class FinAgentWorkflow:
    """
    LangGraph-based multi-agent workflow.
    
    Orchestrates the flow between agents based on
    query complexity and validation results.
    """
    
    MAX_ITERATIONS = 3
    
    def __init__(
        self,
        router: QueryRouter = None,
        planner: QueryPlanner = None,
        retriever: RetrieverAgent = None,
        analyst: AnalystAgent = None,
        synthesizer: Synthesizer = None,
        validator: Validator = None
    ):
        """
        Initialize workflow with agents.
        
        Args:
            router: Query router agent
            planner: Query planner agent
            retriever: Document retriever agent
            analyst: Analysis agent
            synthesizer: Response synthesizer agent
            validator: Response validator agent
        """
        self.router = router or QueryRouter()
        self.planner = planner or QueryPlanner()
        self.retriever = retriever or RetrieverAgent()
        self.analyst = analyst or AnalystAgent()
        self.synthesizer = synthesizer or Synthesizer()
        self.validator = validator or Validator()
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state machine.
        
        Returns:
            Compiled StateGraph
        """
        logger.info("Building LangGraph state machine")
        
        # Create graph with AgentState
        workflow = StateGraph(AgentState)
        
        # Add agent nodes
        workflow.add_node("router", self._router_node)
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("retriever", self._retriever_node)
        workflow.add_node("analyst", self._analyst_node)
        workflow.add_node("synthesizer", self._synthesizer_node)
        workflow.add_node("validator", self._validator_node)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Add conditional routing after router
        workflow.add_conditional_edges(
            "router",
            self._route_after_classification,
            {
                "planner": "planner",
                "retriever": "retriever"
            }
        )
        
        # Planner always goes to retriever
        workflow.add_edge("planner", "retriever")
        
        # Retriever goes to analyst (or synthesizer for SIMPLE queries)
        workflow.add_conditional_edges(
            "retriever",
            self._route_after_retrieval,
            {
                "analyst": "analyst",
                "synthesizer": "synthesizer"
            }
        )
        
        # Analyst always goes to synthesizer
        workflow.add_edge("analyst", "synthesizer")
        
        # Synthesizer always goes to validator
        workflow.add_edge("synthesizer", "validator")
        
        # Validator can loop back or end
        workflow.add_conditional_edges(
            "validator",
            self._route_after_validation,
            {
                "retriever": "retriever",  # Loop back to get better sources
                "end": END
            }
        )
        
        # Compile the graph
        compiled = workflow.compile()
        logger.info("LangGraph state machine built successfully")
        
        return compiled
    
    async def run(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> QueryResponse:
        """
        Run the full workflow for a query.
        
        Args:
            query: User's financial research query
            filters: Optional metadata filters
            
        Returns:
            QueryResponse with answer and citations
        """
        start_time = time.time()
        
        logger.info(f"Starting workflow for query: '{query[:50]}...'")
        
        # Initialize state
        initial_state = AgentState(
            original_query=query,
            filters=filters
        )
        
        try:
            # Run the graph
            final_state = await self.graph.ainvoke(initial_state)
            
            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Build response
            response = self._build_response(final_state, processing_time_ms)
            
            logger.info(f"Workflow completed in {processing_time_ms}ms")
            return response
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            return QueryResponse(
                query=query,
                answer=f"Error processing query: {str(e)}",
                citations=[],
                sources=[],
                confidence=0.0,
                reasoning_trace=[{"agent": "error", "error": str(e)}],
                processing_time_ms=processing_time_ms
            )
    
    def _route_after_classification(
        self,
        state: AgentState
    ) -> Literal["planner", "retriever"]:
        """
        Route based on query complexity.
        
        Args:
            state: Current state with complexity
            
        Returns:
            Next node name
        """
        if state.complexity == QueryComplexity.COMPLEX:
            return "planner"
        return "retriever"
    
    def _route_after_retrieval(
        self,
        state: AgentState
    ) -> Literal["analyst", "synthesizer"]:
        """
        Route based on query complexity after retrieval.
        
        Args:
            state: Current state
            
        Returns:
            Next node name
        """
        # SIMPLE queries skip analyst
        if state.complexity == QueryComplexity.SIMPLE:
            return "synthesizer"
        return "analyst"
    
    def _route_after_validation(
        self,
        state: AgentState
    ) -> Literal["retriever", "end"]:
        """
        Route based on validation result.
        
        CRITICAL: This implements the validation loop for hallucination prevention.
        
        Args:
            state: Current state with validation
            
        Returns:
            Next node name
        """
        # If valid, we're done
        if state.is_valid:
            logger.info("Validation passed, ending workflow")
            return "end"
        
        # If max iterations reached, give up
        if state.iteration_count >= self.MAX_ITERATIONS:
            logger.warning(f"Max iterations ({self.MAX_ITERATIONS}) reached, ending workflow")
            return "end"
        
        # Loop back to retriever for better sources
        logger.info(f"Validation failed (attempt {state.iteration_count}), looping back to retriever")
        return "retriever"
    
    def _build_response(
        self,
        state: AgentState,
        processing_time_ms: int
    ) -> QueryResponse:
        """
        Build final QueryResponse from state.
        
        Args:
            state: Final agent state
            processing_time_ms: Total processing time
            
        Returns:
            QueryResponse object
        """
        # Calculate confidence from validation and retrieval scores
        confidence = 0.0
        if state.retrieved_docs:
            avg_retrieval_score = sum(
                d.score for d in state.retrieved_docs
            ) / len(state.retrieved_docs)
            confidence = avg_retrieval_score
        
        # Get source metadata
        sources = [
            doc.chunk.metadata 
            for doc in state.retrieved_docs[:5]  # Top 5 sources
        ]
        
        return QueryResponse(
            query=state.original_query,
            answer=state.draft_response or "Unable to generate response.",
            citations=state.citations,
            sources=sources,
            confidence=confidence,
            reasoning_trace=self._build_reasoning_trace(state),
            processing_time_ms=processing_time_ms
        )
    
    async def _router_node(self, state: AgentState) -> AgentState:
        """Router agent node."""
        return await self.router.route(state)
    
    async def _planner_node(self, state: AgentState) -> AgentState:
        """Planner agent node."""
        return await self.planner.plan(state)
    
    async def _retriever_node(self, state: AgentState) -> AgentState:
        """Retriever agent node."""
        return await self.retriever.retrieve(state)
    
    async def _analyst_node(self, state: AgentState) -> AgentState:
        """Analyst agent node."""
        return await self.analyst.analyze(state)
    
    async def _synthesizer_node(self, state: AgentState) -> AgentState:
        """Synthesizer agent node."""
        return await self.synthesizer.synthesize(state)
    
    async def _validator_node(self, state: AgentState) -> AgentState:
        """Validator agent node."""
        state.iteration_count += 1
        return await self.validator.validate_for_state(state)
    
    def _build_reasoning_trace(
        self,
        state: AgentState
    ) -> list:
        """
        Build reasoning trace from state history.
        
        Args:
            state: Final agent state
            
        Returns:
            List of reasoning steps
        """
        trace = []
        
        # Add complexity classification
        if state.complexity:
            trace.append({
                "agent": "router",
                "action": "classify_complexity",
                "result": state.complexity.value,
                "reasoning": f"Query classified as {state.complexity.value}"
            })
        
        # Add planning if present
        if state.sub_queries:
            trace.append({
                "agent": "planner",
                "action": "decompose_query",
                "result": [sq.sub_query for sq in state.sub_queries],
                "reasoning": f"Decomposed into {len(state.sub_queries)} sub-queries"
            })
        
        # Add retrieval stats
        if state.retrieved_docs:
            avg_score = sum(d.score for d in state.retrieved_docs) / len(state.retrieved_docs)
            trace.append({
                "agent": "retriever",
                "action": "retrieve_documents",
                "result": {
                    "num_docs": len(state.retrieved_docs),
                    "avg_score": avg_score,
                    "methods": list(set(d.retrieval_method for d in state.retrieved_docs))
                },
                "reasoning": f"Retrieved {len(state.retrieved_docs)} documents (avg score: {avg_score:.3f})"
            })
        
        # Add analyst results if present
        if state.extracted_data:
            trace.append({
                "agent": "analyst",
                "action": "extract_data",
                "result": {"num_facts": len(state.extracted_data)},
                "reasoning": f"Extracted {len(state.extracted_data)} data points"
            })
        
        # Add synthesis info
        if state.citations:
            trace.append({
                "agent": "synthesizer",
                "action": "generate_response",
                "result": {
                    "num_citations": len(state.citations),
                    "response_length": len(state.draft_response) if state.draft_response else 0
                },
                "reasoning": f"Generated response with {len(state.citations)} citations"
            })
        
        # Add validation result
        trace.append({
            "agent": "validator",
            "action": "validate_response",
            "result": {
                "is_valid": state.is_valid,
                "iterations": state.iteration_count,
                "feedback": state.validation_feedback
            },
            "reasoning": "Validation passed" if state.is_valid else f"Validation failed: {state.validation_feedback}"
        })
        
        return trace
    
    def get_workflow_diagram(self) -> str:
        """
        Get ASCII diagram of the workflow.
        
        Returns:
            ASCII art workflow diagram
        """
        return """
        ┌─────────┐
        │  Query  │
        └────┬────┘
             │
        ┌────▼────┐
        │ Router  │
        └────┬────┘
             │
        ┌────▼────┐    ┌─────────┐
        │Complex? │───►│ Planner │
        └────┬────┘    └────┬────┘
             │              │
             └──────┬───────┘
                    │
        ┌───────────▼───────────┐
        │      Retriever        │
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │       Analyst         │
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │     Synthesizer       │
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │      Validator        │◄──┐
        └───────────┬───────────┘   │
                    │               │
              ┌─────▼─────┐         │
              │  Valid?   │─────No──┘
              └─────┬─────┘
                    │Yes
              ┌─────▼─────┐
              │ Response  │
              └───────────┘
        """
