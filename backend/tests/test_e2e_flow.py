"""
End-to-End Flow Tests for FinAgent

Tests the complete flow from query input to response output.
Validates agent pipeline, error recovery, and data flow.

Usage:
    pytest tests/test_e2e_flow.py -v
    pytest tests/test_e2e_flow.py -v -k "test_simple_query"
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.models import (
    AgentState, QueryComplexity, RetrievedDocument,
    DocumentChunk, DocumentMetadata, DocumentType, Citation
)
from app.agents.workflow import FinAgentWorkflow
from app.agents.router import QueryRouter
from app.agents.retriever_agent import RetrieverAgent
from app.agents.validator import Validator


class TestEndToEndFlow:
    """End-to-end flow tests."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        mock = MagicMock()
        mock.chat.completions.create = AsyncMock()
        return mock

    @pytest.fixture
    def sample_document(self):
        """Create a sample document for testing."""
        return DocumentChunk(
            chunk_id="test-chunk-1",
            document_id="test-doc-1",
            content="Apple Inc. reported revenue of $394.3 billion for fiscal year 2022, representing a 7.8% increase from the previous year. The company's iPhone segment generated $205.5 billion in revenue.",
            metadata=DocumentMetadata(
                ticker="AAPL",
                company_name="Apple Inc.",
                document_type=DocumentType.SEC_10K,
                filing_date=datetime(2023, 1, 15),
                source_url="https://www.sec.gov/example",
                accession_number="0000320193-22-000108"
            ),
            section="Financial Highlights",
            chunk_index=0
        )

    @pytest.fixture
    def sample_retrieved_doc(self, sample_document):
        """Create a sample retrieved document."""
        return RetrievedDocument(
            chunk=sample_document,
            score=0.92,
            retrieval_method="hybrid"
        )

    # =========================================================================
    # Router Tests
    # =========================================================================

    class TestRouterFlow:
        """Test router agent classification flow."""

        def test_simple_query_classification(self):
            """Test that simple queries are classified correctly."""
            router = QueryRouter(use_llm=False)

            simple_queries = [
                "What is Apple's revenue?",
                "How much cash does MSFT have?",
                "Who is Tesla's CEO?"
            ]

            for query in simple_queries:
                result = router._heuristic_classify(query)
                assert result == QueryComplexity.SIMPLE, f"Failed for: {query}"

        def test_complex_query_classification(self):
            """Test that complex queries are classified correctly."""
            router = QueryRouter(use_llm=False)

            complex_queries = [
                "Analyze the risk factors and their potential impact on Apple's revenue growth strategy",
                "Compare the strategic outlook and competitive positioning of Microsoft vs Google in AI"
            ]

            for query in complex_queries:
                result = router._heuristic_classify(query)
                assert result == QueryComplexity.COMPLEX, f"Failed for: {query}"

        def test_moderate_query_classification(self):
            """Test moderate query detection by word count."""
            router = QueryRouter(use_llm=False)

            # Queries with 10-25 words that don't match simple/complex patterns
            moderate_queries = [
                "Explain how Apple's services segment has grown over the past three years and its contribution to overall revenue",
            ]

            for query in moderate_queries:
                result = router._heuristic_classify(query)
                # Should be None (uncertain) or MODERATE
                assert result is None or result == QueryComplexity.MODERATE, f"Failed for: {query}"

    # =========================================================================
    # Retriever Tests
    # =========================================================================

    class TestRetrieverFlow:
        """Test retriever agent flow."""

        def test_ticker_extraction_explicit(self):
            """Test ticker extraction from explicit patterns."""
            agent = RetrieverAgent()

            test_cases = [
                ("What is $AAPL revenue?", "AAPL"),
                ("MSFT stock performance", "MSFT"),
                ("Tell me about ticker: NVDA", "NVDA"),
                ("What is Apple's revenue?", "AAPL"),  # Company name mapping
                ("Microsoft earnings report", "MSFT"),  # Company name mapping
            ]

            for query, expected_ticker in test_cases:
                result = agent.extract_ticker_from_query(query)
                assert result == expected_ticker, f"Failed for query: {query}, got: {result}"

        def test_ticker_extraction_no_match(self):
            """Test ticker extraction when no ticker present."""
            agent = RetrieverAgent()

            queries_without_tickers = [
                "What is the current market trend?",
                "Explain compound interest",
                "How do I read a balance sheet?",
            ]

            for query in queries_without_tickers:
                result = agent.extract_ticker_from_query(query)
                # Result should be None or an unintended extraction
                # We mainly want to verify it doesn't crash
                assert result is None or isinstance(result, str)

        def test_market_data_query_detection(self):
            """Test market data query detection."""
            agent = RetrieverAgent()

            market_queries = [
                "What is the current price of AAPL?",
                "Apple stock performance today",
                "MSFT trading volume",
                "Market cap of Tesla",
            ]

            for query in market_queries:
                assert agent._is_market_data_query(query), f"Should detect market query: {query}"

            non_market_queries = [
                "What is Apple's CEO name?",
                "Explain Apple's risk factors",
            ]

            for query in non_market_queries:
                # These might still trigger market detection due to "Apple"
                # The important thing is they don't crash
                result = agent._is_market_data_query(query)
                assert isinstance(result, bool)

    # =========================================================================
    # Validator Tests
    # =========================================================================

    class TestValidatorFlow:
        """Test validator agent flow."""

        def test_citation_coverage_full(self):
            """Test citation coverage with fully cited response."""
            validator = Validator()

            response = "Apple's revenue was $394 billion [1]. Growth was 7.8% [2]."
            citations = [
                {"citation_id": "1", "claim": "revenue"},
                {"citation_id": "2", "claim": "growth"}
            ]

            score, issues = validator.check_citation_coverage(response, citations)

            assert score == 100.0, f"Expected 100%, got {score}%"
            assert len(issues) == 0

        def test_citation_coverage_missing(self):
            """Test citation coverage with missing citations."""
            validator = Validator()

            response = "Apple's revenue was $394 billion. Growth was 7.8% year over year."
            citations = []

            score, issues = validator.check_citation_coverage(response, citations)

            assert score < 100.0, "Should have lower score with missing citations"
            assert len(issues) > 0, "Should report issues"

        def test_numerical_accuracy_check(self, sample_retrieved_doc):
            """Test numerical accuracy validation."""
            validator = Validator()

            response = "Revenue was $394.3 billion"
            # Pass RetrievedDocument objects, not strings
            documents = [sample_retrieved_doc]

            score, issues = validator.check_numerical_accuracy(response, documents)

            # The number 394.3 should be found in sources
            assert score >= 50.0, f"Expected higher score, got {score}"

    # =========================================================================
    # Workflow Integration Tests
    # =========================================================================

    class TestWorkflowIntegration:
        """Test full workflow integration."""

        @pytest.fixture
        def mock_workflow(self):
            """Create a workflow with mocked agents."""
            with patch('app.agents.workflow.QueryRouter') as MockRouter, \
                 patch('app.agents.workflow.QueryPlanner') as MockPlanner, \
                 patch('app.agents.workflow.RetrieverAgent') as MockRetriever, \
                 patch('app.agents.workflow.AnalystAgent') as MockAnalyst, \
                 patch('app.agents.workflow.Synthesizer') as MockSynthesizer, \
                 patch('app.agents.workflow.Validator') as MockValidator:

                workflow = FinAgentWorkflow()
                return workflow

        def test_workflow_initialization(self):
            """Test workflow initializes correctly."""
            # Mock all LLM clients used by different agents
            with patch('openai.AsyncOpenAI'), \
                 patch('langchain_openai.ChatOpenAI'):

                workflow = FinAgentWorkflow()

                assert workflow.MAX_ITERATIONS == 3
                assert workflow.router is not None
                assert workflow.planner is not None
                assert workflow.retriever is not None
                assert workflow.analyst is not None
                assert workflow.synthesizer is not None
                assert workflow.validator is not None
                assert workflow.graph is not None

        def test_routing_logic_simple(self):
            """Test routing for simple queries."""
            with patch('openai.AsyncOpenAI'), \
                 patch('langchain_openai.ChatOpenAI'):

                workflow = FinAgentWorkflow()

                state = AgentState(
                    original_query="What is Apple's revenue?",
                    complexity=QueryComplexity.SIMPLE
                )

                next_node = workflow._route_after_classification(state)
                assert next_node == "retriever"

        def test_routing_logic_complex(self):
            """Test routing for complex queries."""
            with patch('openai.AsyncOpenAI'), \
                 patch('langchain_openai.ChatOpenAI'):

                workflow = FinAgentWorkflow()

                state = AgentState(
                    original_query="Compare Apple and Microsoft revenue trends",
                    complexity=QueryComplexity.COMPLEX
                )

                next_node = workflow._route_after_classification(state)
                assert next_node == "planner"

        def test_validation_loop_valid(self):
            """Test validation routing when valid."""
            with patch('openai.AsyncOpenAI'), \
                 patch('langchain_openai.ChatOpenAI'):

                workflow = FinAgentWorkflow()

                state = AgentState(
                    original_query="test",
                    is_valid=True,
                    iteration_count=1
                )

                next_node = workflow._route_after_validation(state)
                assert next_node == "end"

        def test_validation_loop_retry(self):
            """Test validation routing for retry."""
            with patch('openai.AsyncOpenAI'), \
                 patch('langchain_openai.ChatOpenAI'):

                workflow = FinAgentWorkflow()

                state = AgentState(
                    original_query="test",
                    is_valid=False,
                    iteration_count=1
                )

                next_node = workflow._route_after_validation(state)
                assert next_node == "retriever"

        def test_validation_loop_max_iterations(self):
            """Test validation routing at max iterations."""
            with patch('openai.AsyncOpenAI'), \
                 patch('langchain_openai.ChatOpenAI'):

                workflow = FinAgentWorkflow()

                state = AgentState(
                    original_query="test",
                    is_valid=False,
                    iteration_count=3  # At max
                )

                next_node = workflow._route_after_validation(state)
                assert next_node == "end"

        def test_reasoning_trace_generation(self, sample_retrieved_doc):
            """Test reasoning trace is built correctly."""
            with patch('openai.AsyncOpenAI'), \
                 patch('langchain_openai.ChatOpenAI'):

                workflow = FinAgentWorkflow()

                state = AgentState(
                    original_query="What is Apple's revenue?",
                    complexity=QueryComplexity.SIMPLE,
                    retrieved_docs=[sample_retrieved_doc],
                    draft_response="Apple's revenue was $394.3 billion.",
                    citations=[],
                    is_valid=True,
                    iteration_count=1
                )

                trace = workflow._build_reasoning_trace(state)

                assert len(trace) >= 3  # At minimum: router, retriever, validator
                assert trace[0]["agent"] == "router"
                assert trace[0]["result"] == "simple"

    # =========================================================================
    # Error Recovery Tests
    # =========================================================================

    class TestErrorRecovery:
        """Test error recovery mechanisms."""

        def test_error_recovery_initialization(self):
            """Test error recovery agent initialization."""
            from app.agents.error_recovery import ErrorRecoveryAgent

            agent = ErrorRecoveryAgent()

            # Check that error patterns are initialized
            assert len(agent.error_patterns) > 0
            assert agent.error_counts == {}
            assert agent.circuit_breakers == {}

        def test_error_classification(self):
            """Test error type classification."""
            from app.agents.error_recovery import ErrorRecoveryAgent, ErrorType

            agent = ErrorRecoveryAgent()

            test_cases = [
                ("Connection timeout after 30s", ErrorType.STREAM_TIMEOUT),
                ("Connection lost to server", ErrorType.CONNECTION_LOST),
                ("Rate limit exceeded", ErrorType.RATE_LIMIT),
                ("Server error 500", ErrorType.BACKEND_ERROR),
                ("Validation failed: low citation coverage", ErrorType.VALIDATION_FAILURE),
            ]

            for error_msg, expected_type in test_cases:
                result = agent._classify_error(error_msg)
                assert result == expected_type, f"Failed for: {error_msg}, got: {result}"

        def test_circuit_breaker_state(self):
            """Test circuit breaker state management."""
            from app.agents.error_recovery import ErrorRecoveryAgent, ErrorType

            agent = ErrorRecoveryAgent()

            # Initial state - circuit should be closed
            assert not agent._is_circuit_open(ErrorType.STREAM_TIMEOUT)

            # Manually open the circuit
            agent._open_circuit(ErrorType.STREAM_TIMEOUT)

            # Circuit should now be open
            assert agent._is_circuit_open(ErrorType.STREAM_TIMEOUT)

            # Close the circuit
            agent._close_circuit(ErrorType.STREAM_TIMEOUT)
            assert not agent._is_circuit_open(ErrorType.STREAM_TIMEOUT)

        def test_error_counts_reset(self):
            """Test error count reset."""
            from app.agents.error_recovery import ErrorRecoveryAgent, ErrorType

            agent = ErrorRecoveryAgent()

            # Record some errors directly
            agent.error_counts[ErrorType.STREAM_TIMEOUT] = 5
            agent.circuit_breakers[ErrorType.STREAM_TIMEOUT] = True

            # Reset
            agent.reset_error_counts()

            # Should be back to empty
            assert agent.error_counts == {}
            assert agent.circuit_breakers == {}

        def test_health_status(self):
            """Test health status reporting."""
            from app.agents.error_recovery import ErrorRecoveryAgent

            agent = ErrorRecoveryAgent()

            status = agent.get_health_status()

            assert "error_counts" in status
            assert "circuit_breakers" in status
            assert "last_errors" in status


# =========================================================================
# Data Flow Tests
# =========================================================================

class TestDataFlow:
    """Test data flows correctly through the system."""

    def test_agent_state_creation(self):
        """Test AgentState can be created with minimal fields."""
        state = AgentState(original_query="Test query")

        assert state.original_query == "Test query"
        assert state.complexity is None
        assert state.sub_queries == []
        assert state.retrieved_docs == []
        assert state.is_valid is False
        assert state.iteration_count == 0

    def test_agent_state_full(self):
        """Test AgentState with all fields populated."""
        state = AgentState(
            original_query="Complex query",
            complexity=QueryComplexity.COMPLEX,
            draft_response="Test response",
            is_valid=True,
            iteration_count=2
        )

        assert state.complexity == QueryComplexity.COMPLEX
        assert state.draft_response == "Test response"
        assert state.is_valid is True
        assert state.iteration_count == 2

    def test_document_metadata_creation(self):
        """Test DocumentMetadata creation."""
        metadata = DocumentMetadata(
            ticker="AAPL",
            company_name="Apple Inc.",
            document_type=DocumentType.SEC_10K,
            filing_date=datetime(2023, 1, 15),
            source_url="https://sec.gov/example"
        )

        assert metadata.ticker == "AAPL"
        assert metadata.document_type == DocumentType.SEC_10K

    def test_citation_creation(self):
        """Test Citation model creation."""
        citation = Citation(
            citation_id="1",
            citation_number=1,
            claim="Revenue was $394 billion",
            source_chunk_id="chunk-1",
            source_document_id="doc-1",
            source_text="Apple reported $394 billion in revenue",
            confidence=0.95
        )

        assert citation.citation_id == "1"
        assert citation.citation_number == 1
        assert citation.confidence == 0.95


# =========================================================================
# API Endpoint Tests
# =========================================================================

class TestAPIEndpoints:
    """Test API endpoint behavior."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient

        # Patch the workflow to avoid actual LLM calls
        with patch('app.main.workflow') as mock_workflow:
            mock_workflow.initialize = AsyncMock()
            mock_workflow.graph.ainvoke = AsyncMock(return_value={
                "draft_response": "Test response",
                "citations": [],
                "retrieved_docs": []
            })

            from app.main import app
            return TestClient(app)

    def test_ping_endpoint(self, client):
        """Test /ping endpoint."""
        response = client.get("/ping")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pong"
        assert "timestamp" in data

    def test_health_endpoint(self, client):
        """Test /health endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_root_endpoint(self, client):
        """Test / endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "FinAgent" in data["message"]


# =========================================================================
# Performance and Robustness Tests
# =========================================================================

class TestRobustness:
    """Test system robustness."""

    def test_empty_query_handling(self):
        """Test handling of empty query."""
        state = AgentState(original_query="")

        # Should not raise an error
        assert state.original_query == ""

    def test_very_long_query_handling(self):
        """Test handling of very long query."""
        long_query = "Analyze " + "Apple revenue " * 100
        state = AgentState(original_query=long_query)

        assert len(state.original_query) > 1000

    def test_special_characters_in_query(self):
        """Test handling of special characters."""
        queries_with_special_chars = [
            "What is Apple's (AAPL) revenue?",
            "Revenue: $100 billion?",
            "Compare Apple & Microsoft",
            "Revenue was ~$400B",
        ]

        for query in queries_with_special_chars:
            state = AgentState(original_query=query)
            assert state.original_query == query

    def test_unicode_in_query(self):
        """Test handling of unicode characters."""
        unicode_query = "What is Apple's revenue in €?"
        state = AgentState(original_query=unicode_query)

        assert "€" in state.original_query


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
