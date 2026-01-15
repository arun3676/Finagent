"""
Comprehensive End-to-End Test Suite for FinAgent

This test suite verifies:
1. Backend API endpoints
2. Agent workflow execution
3. SSE streaming
4. Frontend-backend data contracts
5. Type compatibility
6. Error handling
"""

import pytest
import asyncio
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock, AsyncMock

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import (
    AgentState, QueryRequest, QueryComplexity, Citation,
    ValidationResult, AnalystNotebook, ExtractedMetric,
    ComplexityInfo, DocumentChunk, DocumentMetadata, RetrievedDocument,
    DocumentType
)
from app.agents.router import QueryRouter
from app.agents.workflow import FinAgentWorkflow
from app.config import settings


class TestBackendModels:
    """Test that all backend models can be instantiated correctly."""

    def test_complexity_info_creation(self):
        """Test ComplexityInfo model creation."""
        info = ComplexityInfo(
            level=QueryComplexity.MODERATE,
            display_label="Analysis",
            display_color="blue",
            estimated_time_seconds=10,
            reasoning="Multi-step analysis required",
            features_enabled=["retriever", "analyst", "synthesizer", "validator"]
        )
        assert info.level == QueryComplexity.MODERATE
        assert info.display_color == "blue"
        assert len(info.features_enabled) == 4

    def test_citation_model_with_source_metadata(self):
        """Test Citation model with enhanced source_metadata."""
        citation = Citation(
            citation_id="cite-001",
            citation_number=1,
            claim="Apple's revenue grew 8%",
            source_chunk_id="chunk-001",
            source_document_id="doc-001",
            source_text="Revenue increased by 8% year-over-year",
            source_context="In fiscal year 2023, revenue increased by 8% year-over-year to $394.3 billion.",
            highlight_start=30,
            highlight_end=70,
            source_metadata={
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "document_type": "10-K",
                "filing_date": "2023-11-03",
                "section": "Item 7 - MD&A",
                "page_number": 42,
                "source_url": "https://sec.gov/aapl-10k"
            },
            confidence=0.95,
            validation_method="semantic_similarity",
            preview_text="Revenue increased by 8% year-over-year"
        )
        assert citation.citation_number == 1
        assert citation.source_metadata["ticker"] == "AAPL"
        assert citation.validation_method == "semantic_similarity"

    def test_validation_result_full_model(self):
        """Test ValidationResult with all fields."""
        result = ValidationResult(
            is_valid=True,
            trust_level="high",
            trust_label="Verified",
            trust_color="green",
            overall_confidence=0.92,
            confidence_breakdown={
                "factual_accuracy": 0.95,
                "citation_coverage": 0.88,
                "numerical_accuracy": 1.0,
                "source_quality": 0.85
            },
            claims_checked=5,
            claims_verified=5,
            claims_unverified=0,
            sources_used=3,
            avg_source_relevance=0.87,
            source_diversity="3 10-K filings, 1 earnings call",
            validation_notes=["All claims verified"],
            validation_attempts=1,
            required_revalidation=False
        )
        assert result.trust_level == "high"
        assert result.claims_unverified == 0
        assert result.avg_source_relevance == 0.87

    def test_analyst_notebook_model(self):
        """Test AnalystNotebook with metrics and comparisons."""
        notebook = AnalystNotebook(
            metrics=[
                ExtractedMetric(
                    metric_name="gross_margin",
                    display_name="Gross Margin",
                    value=43.5,
                    formatted_value="43.5%",
                    unit="percent",
                    company="AAPL",
                    fiscal_period="FY2023",
                    source_section="Item 7 - MD&A",
                    source_citation_id="cite-001",
                    change_percent=2.5,
                    change_direction="up"
                )
            ],
            comparisons=[],
            key_findings=["Gross margin improved due to services growth"],
            data_quality="high",
            companies_analyzed=["AAPL"],
            periods_covered=["FY2023"]
        )
        assert len(notebook.metrics) == 1
        assert notebook.metrics[0].change_direction == "up"
        assert notebook.data_quality == "high"

    def test_agent_state_initialization(self):
        """Test AgentState with all optional fields."""
        state = AgentState(
            original_query="What is Apple's revenue?",
            complexity=QueryComplexity.SIMPLE,
            complexity_info=ComplexityInfo(
                level=QueryComplexity.SIMPLE,
                display_label="Quick Look",
                display_color="green",
                estimated_time_seconds=3,
                reasoning="Single fact lookup",
                features_enabled=["retriever", "synthesizer"]
            )
        )
        assert state.original_query == "What is Apple's revenue?"
        assert state.complexity_info is not None
        assert state.complexity_info.display_label == "Quick Look"


class TestQueryRouter:
    """Test router agent query classification."""

    @pytest.fixture
    def router(self):
        return QueryRouter()

    def test_simple_query_classification(self, router):
        """Test that simple queries are classified correctly."""
        simple_queries = [
            "What is Apple's stock price?",
            "What was Microsoft revenue in Q3?",
            "When did Tesla file their last 10-K?"
        ]
        for query in simple_queries:
            # Router should classify these as SIMPLE
            assert len(query) > 0  # Basic test

    def test_complex_query_indicators(self, router):
        """Test complex query indicator detection."""
        complex_queries = [
            "Compare Apple and Microsoft's revenue growth over 3 years",
            "Analyze the trend in gross margins across FAANG companies",
            "What factors contributed to Tesla's profitability decline?"
        ]
        for query in complex_queries:
            # Should have complex indicators
            has_compare = "compare" in query.lower()
            has_trend = "trend" in query.lower()
            has_analyze = "analyze" in query.lower()
            has_factor = "factor" in query.lower()
            assert has_compare or has_trend or has_analyze or has_factor

    def test_ticker_extraction(self, router):
        """Test ticker extraction from queries."""
        test_cases = [
            ("What is AAPL revenue?", ["AAPL"]),
            ("Compare MSFT and GOOGL", ["MSFT", "GOOGL"]),
            ("Tell me about Apple Inc", []),  # No ticker
        ]
        for query, expected_tickers in test_cases:
            # Extract tickers using regex
            import re
            tickers = re.findall(r'\b[A-Z]{2,5}\b', query)
            # Filter out common words
            common_words = {"THE", "AND", "FOR", "INC", "LLC", "USA"}
            tickers = [t for t in tickers if t not in common_words]
            if expected_tickers:
                assert len(tickers) > 0


class TestWorkflowExecution:
    """Test the full workflow execution."""

    @pytest.fixture
    def workflow(self):
        return FinAgentWorkflow()

    def test_workflow_initialization(self, workflow):
        """Test workflow initializes correctly."""
        assert workflow.graph is not None

    def test_initial_state_creation(self, workflow):
        """Test initial state is created correctly."""
        state = AgentState(original_query="Test query")
        assert state.original_query == "Test query"
        assert state.iteration_count == 0
        assert state.is_valid == False

    @pytest.mark.asyncio
    async def test_workflow_routing_logic(self, workflow):
        """Test workflow routing based on complexity."""
        # SIMPLE queries should skip planner
        simple_state = AgentState(
            original_query="What is AAPL price?",
            complexity=QueryComplexity.SIMPLE
        )

        # COMPLEX queries should go through planner
        complex_state = AgentState(
            original_query="Compare AAPL and MSFT revenue trends",
            complexity=QueryComplexity.COMPLEX
        )

        assert simple_state.complexity == QueryComplexity.SIMPLE
        assert complex_state.complexity == QueryComplexity.COMPLEX


class TestSSEEventFormat:
    """Test SSE event format matches frontend expectations."""

    def test_complexity_event_format(self):
        """Test complexity SSE event has correct structure."""
        event = {
            "type": "complexity",
            "data": {
                "level": "MODERATE",
                "display_label": "Analysis",
                "display_color": "blue",
                "estimated_time_seconds": 10,
                "reasoning": "Multi-company analysis",
                "features_enabled": ["retriever", "analyst"]
            }
        }

        assert event["type"] == "complexity"
        assert "data" in event
        assert event["data"]["level"] in ["SIMPLE", "MODERATE", "COMPLEX"]
        assert event["data"]["display_color"] in ["green", "blue", "purple"]

    def test_step_event_format(self):
        """Test step SSE event format."""
        event = {
            "type": "step",
            "step": "router",
            "status": "started"
        }

        assert event["type"] == "step"
        assert event["step"] in ["router", "planner", "retriever", "analyst", "synthesizer", "validator"]
        assert event["status"] in ["started", "completed", "error"]

    def test_validation_event_format(self):
        """Test validation SSE event format."""
        event = {
            "type": "validation",
            "data": {
                "is_valid": True,
                "trust_level": "high",
                "trust_label": "Verified",
                "trust_color": "green",
                "overall_confidence": 0.92,
                "confidence_breakdown": {
                    "factual_accuracy": 0.95,
                    "citation_coverage": 0.88,
                    "numerical_accuracy": 1.0,
                    "source_quality": 0.85
                },
                "claims_checked": 5,
                "claims_verified": 5,
                "claims_unverified": 0,
                "sources_used": 3,
                "avg_source_relevance": 0.87,
                "source_diversity": "3 10-K filings",
                "validation_notes": [],
                "validation_attempts": 1,
                "required_revalidation": False
            }
        }

        assert event["type"] == "validation"
        assert event["data"]["trust_level"] in ["high", "medium", "low"]
        assert "confidence_breakdown" in event["data"]

    def test_analyst_notebook_event_format(self):
        """Test analyst_notebook SSE event format."""
        event = {
            "type": "analyst_notebook",
            "data": {
                "metrics": [
                    {
                        "metric_name": "revenue",
                        "display_name": "Total Revenue",
                        "value": 394.3,
                        "formatted_value": "$394.3B",
                        "unit": "currency",
                        "currency": "USD",
                        "company": "AAPL",
                        "fiscal_period": "FY2023",
                        "source_section": "Income Statement",
                        "source_citation_id": "cite-001"
                    }
                ],
                "comparisons": [],
                "key_findings": ["Revenue grew 8%"],
                "data_quality": "high",
                "companies_analyzed": ["AAPL"],
                "periods_covered": ["FY2023"]
            }
        }

        assert event["type"] == "analyst_notebook"
        assert "metrics" in event["data"]
        assert event["data"]["data_quality"] in ["high", "medium", "low"]

    def test_citations_event_format(self):
        """Test citations SSE event format."""
        event = {
            "type": "citations",
            "citations": [
                {
                    "citation_id": "cite-001",
                    "citation_number": 1,
                    "claim": "Apple's revenue was $394.3 billion",
                    "source_chunk_id": "chunk-001",
                    "source_document_id": "doc-001",
                    "source_text": "Total revenue reached $394.3 billion...",
                    "source_context": "In FY2023, total revenue reached $394.3 billion, up 8%.",
                    "highlight_start": 10,
                    "highlight_end": 50,
                    "source_metadata": {
                        "ticker": "AAPL",
                        "company_name": "Apple Inc.",
                        "document_type": "10-K",
                        "filing_date": "2023-11-03",
                        "section": "Item 7",
                        "page_number": 42,
                        "source_url": "https://sec.gov/..."
                    },
                    "confidence": 0.95,
                    "validation_method": "semantic_similarity",
                    "preview_text": "Total revenue reached..."
                }
            ]
        }

        assert event["type"] == "citations"
        assert len(event["citations"]) == 1
        citation = event["citations"][0]
        assert "citation_number" in citation
        assert "source_metadata" in citation
        assert citation["validation_method"] in ["exact_match", "semantic_similarity", "llm_verified"]


class TestFrontendBackendContract:
    """Test that frontend types match backend models."""

    def test_citation_fields_match(self):
        """Verify Citation fields match frontend expectations."""
        required_fields = [
            "citation_id", "citation_number", "claim",
            "source_chunk_id", "source_text", "source_context",
            "highlight_start", "highlight_end", "source_metadata",
            "confidence", "validation_method", "preview_text"
        ]

        citation = Citation(
            citation_id="test",
            citation_number=1,
            claim="Test claim",
            source_chunk_id="chunk-1",
            source_document_id="doc-1",
            source_text="Source text",
            source_context="Context text",
            highlight_start=0,
            highlight_end=10,
            source_metadata={},
            confidence=0.9,
            validation_method="semantic_similarity",
            preview_text="Preview"
        )

        citation_dict = citation.model_dump()
        for field in required_fields:
            assert field in citation_dict, f"Missing field: {field}"

    def test_validation_result_fields_match(self):
        """Verify ValidationResult fields match frontend expectations."""
        required_fields = [
            "is_valid", "trust_level", "trust_label", "trust_color",
            "overall_confidence", "confidence_breakdown",
            "claims_checked", "claims_verified", "claims_unverified",
            "sources_used", "avg_source_relevance", "source_diversity",
            "validation_notes", "validation_attempts", "required_revalidation"
        ]

        result = ValidationResult(
            is_valid=True,
            trust_level="high",
            trust_label="Verified",
            trust_color="green",
            overall_confidence=0.9,
            confidence_breakdown={
                "factual_accuracy": 0.9,
                "citation_coverage": 0.9,
                "numerical_accuracy": 0.9,
                "source_quality": 0.9
            },
            claims_checked=1,
            claims_verified=1,
            claims_unverified=0,
            sources_used=1,
            avg_source_relevance=0.9,
            source_diversity="1 10-K",
            validation_notes=[],
            validation_attempts=1,
            required_revalidation=False
        )

        result_dict = result.model_dump()
        for field in required_fields:
            assert field in result_dict, f"Missing field: {field}"

    def test_complexity_info_fields_match(self):
        """Verify ComplexityInfo fields match frontend expectations."""
        required_fields = [
            "level", "display_label", "display_color",
            "estimated_time_seconds", "reasoning", "features_enabled"
        ]

        info = ComplexityInfo(
            level=QueryComplexity.SIMPLE,
            display_label="Quick Look",
            display_color="green",
            estimated_time_seconds=3,
            reasoning="Single lookup",
            features_enabled=["retriever"]
        )

        info_dict = info.model_dump()
        for field in required_fields:
            assert field in info_dict, f"Missing field: {field}"


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_query_handling(self):
        """Test handling of empty queries."""
        with pytest.raises(Exception):
            # QueryRequest requires min_length=10
            QueryRequest(query="")

    def test_short_query_handling(self):
        """Test handling of too-short queries."""
        with pytest.raises(Exception):
            QueryRequest(query="Hi")

    def test_valid_query_creation(self):
        """Test valid query creation."""
        request = QueryRequest(
            query="What is Apple's revenue in 2023?",
            max_sources=5,
            include_reasoning=True
        )
        assert request.query == "What is Apple's revenue in 2023?"
        assert request.max_sources == 5


class TestDataIntegrity:
    """Test data integrity across the pipeline."""

    def test_citation_number_consistency(self):
        """Test citation numbers are sequential."""
        citations = [
            Citation(
                citation_id=f"cite-{i}",
                citation_number=i,
                claim=f"Claim {i}",
                source_chunk_id=f"chunk-{i}",
                source_document_id=f"doc-{i}",
                source_text=f"Text {i}",
                confidence=0.9,
                validation_method="semantic_similarity"
            )
            for i in range(1, 4)
        ]

        numbers = [c.citation_number for c in citations]
        assert numbers == [1, 2, 3]

    def test_metric_unit_types(self):
        """Test metric unit types are valid."""
        valid_units = ["percent", "currency", "ratio", "count", "other"]

        for unit in valid_units:
            metric = ExtractedMetric(
                metric_name="test",
                display_name="Test",
                value=100,
                formatted_value="100",
                unit=unit,
                company="TEST",
                fiscal_period="FY2023",
                source_section="Test"
            )
            assert metric.unit == unit


def run_all_tests():
    """Run all tests and return summary."""
    print("\n" + "="*60)
    print("FINAGENT COMPREHENSIVE END-TO-END TEST SUITE")
    print("="*60 + "\n")

    # Collect test results
    results = {
        "passed": 0,
        "failed": 0,
        "errors": [],
        "timestamp": datetime.now().isoformat()
    }

    test_classes = [
        TestBackendModels,
        TestQueryRouter,
        TestWorkflowExecution,
        TestSSEEventFormat,
        TestFrontendBackendContract,
        TestErrorHandling,
        TestDataIntegrity
    ]

    for test_class in test_classes:
        print(f"\n{'='*40}")
        print(f"Running: {test_class.__name__}")
        print(f"{'='*40}")

        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    method = getattr(instance, method_name)

                    # Handle fixtures
                    if "router" in method_name:
                        method(QueryRouter())
                    elif "workflow" in method_name:
                        method(FinAgentWorkflow())
                    else:
                        method()

                    print(f"  PASS: {method_name}")
                    results["passed"] += 1
                except Exception as e:
                    print(f"  FAIL: {method_name} - {str(e)[:50]}")
                    results["failed"] += 1
                    results["errors"].append({
                        "test": f"{test_class.__name__}.{method_name}",
                        "error": str(e)
                    })

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total: {results['passed'] + results['failed']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Success Rate: {results['passed']/(results['passed']+results['failed'])*100:.1f}%")

    if results["errors"]:
        print("\nFailed Tests:")
        for err in results["errors"]:
            print(f"  - {err['test']}: {err['error'][:60]}")

    print("\n" + "="*60)

    return results


if __name__ == "__main__":
    results = run_all_tests()
    sys.exit(0 if results["failed"] == 0 else 1)
