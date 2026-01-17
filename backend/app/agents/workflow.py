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

from typing import Dict, Any, Optional, Literal, List
import logging
import time
import uuid
import asyncio
from langgraph.graph import StateGraph, END

from app.config import settings
from app.models import AgentState, QueryComplexity, QueryResponse, ResponseLength
from app.agents.router import QueryRouter
from app.agents.planner import QueryPlanner
from app.agents.retriever_agent import RetrieverAgent
from app.agents.analyst_agent import AnalystAgent
from app.agents.synthesizer import Synthesizer
from app.agents.validator import Validator
from app.agents.error_recovery import ErrorRecoveryAgent, error_recovery_agent
from app.agents.fast_synthesizer import FastSynthesizer
from app.cache import SemanticQueryCache
from app.followup.cache import ChunkCache, get_chunk_cache
from app.followup.generator import FollowUpGenerator, get_follow_up_generator

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
        validator: Validator = None,
        error_recovery: ErrorRecoveryAgent = None,
        query_cache: SemanticQueryCache = None,
        chunk_cache: ChunkCache = None,
        follow_up_generator: FollowUpGenerator = None
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
            error_recovery: Error recovery agent
            query_cache: Semantic query cache for avoiding redundant LLM calls
            chunk_cache: Cache for retrieved chunks (for follow-up questions)
            follow_up_generator: Generator for follow-up questions
        """
        self.router = router or QueryRouter()
        self.planner = planner or QueryPlanner()
        self.retriever = retriever or RetrieverAgent()
        self.analyst = analyst or AnalystAgent()
        self.synthesizer = synthesizer or Synthesizer()
        self.validator = validator or Validator()
        self.error_recovery = error_recovery or error_recovery_agent
        self.fast_synthesizer = FastSynthesizer()  # For SIMPLE query fast path
        self.query_cache = query_cache  # Can be None, will be set by main.py
        self.chunk_cache = chunk_cache or get_chunk_cache()  # For follow-up questions
        self.follow_up_generator = follow_up_generator or get_follow_up_generator()

        self.graph = self._build_graph()
    
    async def initialize(self):
        """Initialize all agents."""
        logger.info("Initializing workflow agents...")
        if hasattr(self.retriever, "initialize"):
            await self.retriever.initialize()
        logger.info("Workflow agents initialized")
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state machine.
        
        Includes fast path for SIMPLE queries:
        - SIMPLE: router → fast_retrieve → fast_synthesize → END
        - MODERATE/COMPLEX: router → [planner] → retriever → analyst → synthesizer → validator → END
        
        Returns:
            Compiled StateGraph
        """
        logger.info("Building LangGraph state machine with fast path")
        
        # Create graph with AgentState
        workflow = StateGraph(AgentState)
        
        # Add agent nodes (standard path)
        workflow.add_node("router", self._router_node)
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("retriever", self._retriever_node)
        workflow.add_node("analyst", self._analyst_node)
        workflow.add_node("synthesizer", self._synthesizer_node)
        workflow.add_node("validator", self._validator_node)
        
        # Add fast path nodes
        workflow.add_node("fast_retrieve", self._fast_retrieve_node)
        workflow.add_node("fast_synthesize", self._fast_synthesize_node)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Add conditional routing after router - includes fast path
        workflow.add_conditional_edges(
            "router",
            self._route_by_complexity,
            {
                "fast_retrieve": "fast_retrieve",  # SIMPLE queries
                "planner": "planner",              # COMPLEX queries
                "retriever": "retriever"           # MODERATE queries
            }
        )
        
        # Fast path: fast_retrieve → fast_synthesize → END
        workflow.add_edge("fast_retrieve", "fast_synthesize")
        workflow.add_edge("fast_synthesize", END)
        
        # Planner always goes to retriever
        workflow.add_edge("planner", "retriever")
        
        # Retriever goes to analyst (MODERATE/COMPLEX always go through analyst now)
        workflow.add_edge("retriever", "analyst")
        
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
        logger.info("LangGraph state machine with fast path built successfully")
        
        return compiled
    
    async def run(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> QueryResponse:
        """
        Run the full workflow for a query with error recovery.

        Args:
            query: User's financial research query
            filters: Optional metadata filters

        Returns:
            QueryResponse with answer and citations
        """
        start_time = time.time()

        # Generate query_id for follow-up tracking
        query_id = str(uuid.uuid4())

        logger.info(f"Starting workflow for query: '{query[:50]}...' (query_id={query_id})")

        # CHECK CACHE FIRST - before any agent execution
        if self.query_cache:
            try:
                cached = await self.query_cache.get(query)
                if cached:
                    processing_time_ms = int((time.time() - start_time) * 1000)
                    logger.info(f"Cache hit! Returning cached response in {processing_time_ms}ms")

                    return QueryResponse(
                        query=query,
                        answer=cached.response,
                        citations=cached.citations,
                        sources=[],  # Cached responses don't include full sources
                        confidence=0.95,  # High confidence for cached responses
                        reasoning_trace=[{
                            "agent": "cache",
                            "action": "cache_hit",
                            "result": "Returned cached response",
                            "cached_at": cached.cached_at
                        }],
                        processing_time_ms=processing_time_ms
                    )
            except Exception as e:
                logger.warning(f"Cache lookup failed, proceeding with full workflow: {e}")

        # Initialize state
        initial_state = AgentState(
            original_query=query,
            filters=filters
        )

        try:
            # Run the graph with error recovery monitoring
            final_state = await self._run_with_recovery(initial_state)

            # Ensure final_state is AgentState
            if isinstance(final_state, dict):
                # LangGraph might return a dict, convert back to AgentState
                # We need to handle potential missing fields or extra fields
                # Filter out fields that are not in AgentState
                valid_fields = AgentState.model_fields.keys()
                filtered_state = {k: v for k, v in final_state.items() if k in valid_fields}
                final_state = AgentState(**filtered_state)

            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)

            # Build response with query_id
            response = self._build_response(final_state, processing_time_ms, query_id)

            # Cache chunks for follow-up questions (async, non-blocking)
            asyncio.create_task(
                self._cache_chunks_for_followup(
                    query_id=query_id,
                    query_text=query,
                    final_state=final_state,
                    response_summary=response.answer[:500] if response.answer else ""
                )
            )

            # STORE IN CACHE after successful response generation
            if self.query_cache and response.answer and response.confidence > 0.5:
                try:
                    await self.query_cache.set(
                        query=query,
                        response=response.answer,
                        citations=response.citations
                    )
                except Exception as e:
                    logger.warning(f"Failed to cache response: {e}")

            logger.info(f"Workflow completed in {processing_time_ms}ms")
            return response

        except Exception as e:
            logger.error(f"Workflow failed: {e}")

            # Attempt error recovery
            recovery_response = await self._attempt_error_recovery(query, str(e), start_time)
            if recovery_response:
                return recovery_response

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

    async def _cache_chunks_for_followup(
        self,
        query_id: str,
        query_text: str,
        final_state: AgentState,
        response_summary: str
    ) -> None:
        """
        Cache retrieved chunks and generate follow-up questions.

        This runs asynchronously to not block the response.

        Args:
            query_id: Unique query identifier
            query_text: Original query text
            final_state: Final workflow state
            response_summary: Summary of the response
        """
        try:
            # Extract companies and metrics from state
            companies = self._extract_companies(final_state)
            metrics = self._extract_metrics(final_state)

            # Convert retrieved docs to chunks (handle dict or AgentState)
            retrieved_docs = (
                final_state.get("retrieved_docs", [])
                if isinstance(final_state, dict)
                else final_state.retrieved_docs
            ) or []
            chunks = []
            for doc in retrieved_docs:
                if hasattr(doc, "chunk"):
                    chunks.append(doc.chunk)
                elif isinstance(doc, dict) and doc.get("chunk") is not None:
                    chunks.append(doc["chunk"])

            complexity_value = "unknown"
            if isinstance(final_state, dict):
                complexity = final_state.get("complexity")
                if hasattr(complexity, "value"):
                    complexity_value = complexity.value
                elif isinstance(complexity, str):
                    complexity_value = complexity
            elif final_state.complexity:
                complexity_value = final_state.complexity.value

            # Store in chunk cache
            await self.chunk_cache.store(
                query_id=query_id,
                query_text=query_text,
                chunks=chunks,
                metadata={
                    "complexity": complexity_value,
                    "num_citations": len(
                        (final_state.get("citations") or [])
                        if isinstance(final_state, dict)
                        else final_state.citations
                    ),
                },
                companies=companies,
                metrics=metrics,
                response_summary=response_summary
            )

            logger.info(f"Cached {len(chunks)} chunks for query {query_id}")

            # Generate follow-up questions
            follow_ups = await self.follow_up_generator.generate(
                original_query=query_text,
                response_summary=response_summary,
                retrieved_chunks=chunks,
                companies=companies,
                metrics_mentioned=metrics
            )

            # Update cache with follow-ups
            await self.chunk_cache.update_followups(query_id, follow_ups)

            logger.info(f"Generated {len(follow_ups)} follow-up questions for query {query_id}")

        except Exception as e:
            logger.warning(f"Failed to cache chunks/generate follow-ups: {e}")

    def _extract_companies(self, state: AgentState) -> List[str]:
        """Extract company tickers from state."""
        companies = set()

        # From retrieved docs metadata
        retrieved_docs = (
            state.get("retrieved_docs", [])
            if isinstance(state, dict)
            else state.retrieved_docs
        ) or []
        for doc in retrieved_docs:
            if hasattr(doc, "chunk") and doc.chunk.metadata and doc.chunk.metadata.ticker:
                companies.add(doc.chunk.metadata.ticker)
            elif isinstance(doc, dict):
                chunk = doc.get("chunk")
                metadata = getattr(chunk, "metadata", None) if chunk else None
                ticker = getattr(metadata, "ticker", None) if metadata else None
                if ticker:
                    companies.add(ticker)

        # From analyst notebook if present
        analyst_notebook = (
            state.get("analyst_notebook")
            if isinstance(state, dict)
            else state.analyst_notebook
        )
        if analyst_notebook:
            if hasattr(analyst_notebook, "companies_analyzed") and analyst_notebook.companies_analyzed:
                companies.update(analyst_notebook.companies_analyzed)
            elif isinstance(analyst_notebook, dict):
                companies.update(analyst_notebook.get("companies_analyzed", []))

        return list(companies)

    def _extract_metrics(self, state: AgentState) -> List[str]:
        """Extract financial metrics mentioned in state."""
        metrics = set()

        # From analyst notebook if present
        analyst_notebook = (
            state.get("analyst_notebook")
            if isinstance(state, dict)
            else state.analyst_notebook
        )
        if analyst_notebook:
            if hasattr(analyst_notebook, "metrics"):
                for metric in analyst_notebook.metrics:
                    if hasattr(metric, "metric_name"):
                        metrics.add(metric.metric_name)
            elif isinstance(analyst_notebook, dict):
                for metric in analyst_notebook.get("metrics", []):
                    if isinstance(metric, dict) and metric.get("metric_name"):
                        metrics.add(metric["metric_name"])

        # From extracted data
        extracted_data = (
            state.get("extracted_data")
            if isinstance(state, dict)
            else state.extracted_data
        )
        if extracted_data and isinstance(extracted_data, dict):
            if "key_metrics" in extracted_data:
                metrics.update(extracted_data["key_metrics"].keys())

        return list(metrics)
    
    def _route_by_complexity(
        self,
        state: AgentState
    ) -> Literal["fast_retrieve", "planner", "retriever"]:
        """
        Route based on response_length AND query complexity.

        Response length takes priority:
        - SHORT: Always fast path (fast_retrieve → fast_synthesize → END) - fastest response
        - NORMAL: Use complexity-based routing (current behavior)
        - DETAILED: Always full path with planner - most thorough

        Within NORMAL mode, complexity determines:
        - SIMPLE: fast path
        - MODERATE: standard path without planner
        - COMPLEX: full path with planner

        Args:
            state: Current state with complexity and response_length

        Returns:
            Next node name
        """
        response_length = state.response_length or ResponseLength.NORMAL

        # SHORT response: Always use fast path for speed
        if response_length == ResponseLength.SHORT:
            logger.info("Routing to FAST PATH for SHORT response length (speed priority)")
            return "fast_retrieve"

        # DETAILED response: Always use full path for thoroughness
        if response_length == ResponseLength.DETAILED:
            logger.info("Routing to FULL PATH with planner for DETAILED response length (quality priority)")
            return "planner"

        # NORMAL response: Use complexity-based routing
        if state.complexity == QueryComplexity.SIMPLE:
            logger.info("Routing to FAST PATH for SIMPLE query (normal length)")
            return "fast_retrieve"
        elif state.complexity == QueryComplexity.COMPLEX:
            logger.info("Routing to FULL PATH with planner for COMPLEX query (normal length)")
            return "planner"
        else:
            logger.info("Routing to STANDARD PATH for MODERATE query (normal length)")
            return "retriever"
    
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
        processing_time_ms: int,
        query_id: str = ""
    ) -> QueryResponse:
        """
        Build final QueryResponse from state.

        Args:
            state: Final agent state
            processing_time_ms: Total processing time
            query_id: Unique query identifier for follow-up tracking

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

        # Build reasoning trace with query_id
        reasoning_trace = self._build_reasoning_trace(state)
        if query_id:
            reasoning_trace.append({
                "agent": "workflow",
                "action": "assign_query_id",
                "result": query_id,
                "reasoning": "Query ID assigned for follow-up tracking"
            })

        return QueryResponse(
            query=state.original_query,
            answer=state.draft_response or "Unable to generate response.",
            citations=state.citations,
            sources=sources,
            confidence=confidence,
            reasoning_trace=reasoning_trace,
            processing_time_ms=processing_time_ms
        )
    
    async def _router_node(self, state: AgentState) -> Dict[str, Any]:
        """Router agent node."""
        return await self.router.route(state)

    async def _planner_node(self, state: AgentState) -> Dict[str, Any]:
        """Planner agent node."""
        return await self.planner.plan(state)

    async def _retriever_node(self, state: AgentState) -> Dict[str, Any]:
        """Retriever agent node."""
        return await self.retriever.retrieve_for_state(state)
    
    async def _analyst_node(self, state: AgentState) -> Dict[str, Any]:
        """Analyst agent node."""
        return await self.analyst.analyze_for_state(state)
    
    async def _synthesizer_node(self, state: AgentState) -> Dict[str, Any]:
        """Synthesizer agent node."""
        return await self.synthesizer.synthesize_for_state(state)
    
    async def _validator_node(self, state: AgentState) -> Dict[str, Any]:
        """Validator agent node."""
        return await self.validator.validate_for_state(state)
    
    async def _fast_retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Fast retrieval node for SIMPLE queries.
        
        Bypasses planning and uses direct retrieval optimized for speed.
        """
        logger.info("Executing FAST RETRIEVE node")
        docs = await self.retriever.fast_retrieve(
            query=state.original_query,
            top_k=5  # Fewer docs for speed
        )
        return {"retrieved_docs": docs}
    
    async def _fast_synthesize_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Fast synthesis node for SIMPLE queries.
        
        Uses minimal prompt and single LLM call for speed.
        Skips validation for fast path (citations still included).
        """
        logger.info("Executing FAST SYNTHESIZE node")
        return await self.fast_synthesizer.synthesize_for_state(state)
    
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
    
    async def _run_with_recovery(self, initial_state: AgentState) -> AgentState:
        """
        Run the graph with error recovery monitoring.
        
        Args:
            initial_state: Initial agent state
            
        Returns:
            Final agent state
        """
        try:
            # Run the graph normally
            final_state = await self.graph.ainvoke(initial_state)
            return final_state
            
        except Exception as e:
            logger.error(f"Graph execution failed: {e}")
            
            # Check if error recovery can handle this
            error_msg = str(e)
            error_type = self.error_recovery._classify_error(error_msg)
            
            if error_type and not self.error_recovery._is_circuit_open(error_type):
                logger.info(f"Attempting recovery for {error_type.value}")
                
                # Attempt recovery
                recovery_successful = await self.error_recovery._attempt_recovery(error_msg, None)
                
                if recovery_successful:
                    # Retry the graph execution
                    logger.info("Recovery successful, retrying graph execution")
                    return await self.graph.ainvoke(initial_state)
            
            # If recovery failed or not applicable, re-raise the exception
            raise e
    
    async def _attempt_error_recovery(
        self,
        query: str,
        error_msg: str,
        start_time: float
    ) -> Optional[QueryResponse]:
        """
        Attempt to recover from a workflow error and provide a fallback response.
        
        Args:
            query: Original query
            error_msg: Error message
            start_time: Start time for processing time calculation
            
        Returns:
            Fallback QueryResponse or None if recovery not possible
        """
        logger.info(f"Attempting error recovery for: {error_msg}")
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Check if this is a known recoverable error
        error_type = self.error_recovery._classify_error(error_msg)
        
        if error_type:
            # Provide appropriate fallback response based on error type
            if "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                return QueryResponse(
                    query=query,
                    answer="I'm experiencing connection issues. Please try your query again in a moment. The system is working to resolve this automatically.",
                    citations=[],
                    sources=[],
                    confidence=0.0,
                    reasoning_trace=[{
                        "agent": "error_recovery",
                        "action": "connection_fallback",
                        "result": "Provided fallback response for connection issue"
                    }],
                    processing_time_ms=processing_time_ms
                )
            
            elif "rate limit" in error_msg.lower():
                return QueryResponse(
                    query=query,
                    answer="The system is currently experiencing high demand. Please wait a moment and try again. Your query will be processed shortly.",
                    citations=[],
                    sources=[],
                    confidence=0.0,
                    reasoning_trace=[{
                        "agent": "error_recovery",
                        "action": "rate_limit_fallback",
                        "result": "Provided fallback response for rate limiting"
                    }],
                    processing_time_ms=processing_time_ms
                )
            
            elif "validation" in error_msg.lower():
                return QueryResponse(
                    query=query,
                    answer="I'm having trouble validating the response for your query. Please rephrase your question or try a more specific query.",
                    citations=[],
                    sources=[],
                    confidence=0.0,
                    reasoning_trace=[{
                        "agent": "error_recovery",
                        "action": "validation_fallback",
                        "result": "Provided fallback response for validation failure"
                    }],
                    processing_time_ms=processing_time_ms
                )
        
        # No recovery possible
        return None
    
    def get_error_recovery_status(self) -> Dict[str, Any]:
        """
        Get current error recovery status.
        
        Returns:
            Error recovery health status
        """
        return self.error_recovery.get_health_status()
    
    def reset_error_recovery(self):
        """Reset error recovery state (for testing or manual intervention)."""
        self.error_recovery.reset_error_counts()
        logger.info("Error recovery state reset")

    async def get_follow_up_questions(self, query_id: str) -> List[Any]:
        """
        Get generated follow-up questions for a query.

        Args:
            query_id: The query identifier

        Returns:
            List of FollowUpQuestion objects or empty list
        """
        cache_entry = await self.chunk_cache.get(query_id)
        if cache_entry:
            return cache_entry.follow_up_questions
        return []

    def get_chunk_cache(self) -> ChunkCache:
        """Get the chunk cache instance."""
        return self.chunk_cache

    def get_chunk_cache_stats(self) -> Dict[str, Any]:
        """Get chunk cache statistics."""
        return self.chunk_cache.get_stats()
    
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
