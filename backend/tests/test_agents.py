"""
Tests for Agents Module

Tests router, planner, retriever, analyst, synthesizer, and validator agents.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agents.router import QueryRouter
from app.agents.planner import QueryPlanner
from app.agents.retriever_agent import RetrieverAgent
from app.agents.analyst_agent import AnalystAgent
from app.agents.synthesizer import Synthesizer
from app.agents.validator import Validator
from app.models import QueryComplexity, AgentState, DocumentType


class TestQueryRouter:
    """Tests for query router agent."""
    
    @pytest.fixture
    def router(self):
        """Create router instance."""
        return QueryRouter(use_llm=False)
    
    def test_initialization(self, router):
        """Test router initializes correctly."""
        assert router.use_llm is False
        assert len(router.SIMPLE_PATTERNS) > 0
        assert len(router.COMPLEX_PATTERNS) > 0
    
    def test_heuristic_classify_simple(self, router):
        """Test simple query classification."""
        simple_queries = [
            "What is Apple's revenue?",
            "Who is the CEO of Microsoft?",
            "How much cash does Tesla have?"
        ]
        
        for query in simple_queries:
            result = router._heuristic_classify(query)
            assert result == QueryComplexity.SIMPLE, f"Failed for: {query}"
    
    def test_heuristic_classify_complex(self, router):
        """Test complex query classification."""
        complex_queries = [
            "Analyze the risk factors and their potential impact on revenue growth",
            "Compare the strategic outlook and competitive positioning of Apple vs Microsoft"
        ]
        
        for query in complex_queries:
            result = router._heuristic_classify(query)
            assert result == QueryComplexity.COMPLEX, f"Failed for: {query}"
    
    def test_get_pipeline_for_complexity(self, router):
        """Test pipeline selection."""
        simple_pipeline = router.get_pipeline_for_complexity(QueryComplexity.SIMPLE)
        assert "retriever" in simple_pipeline
        assert "planner" not in simple_pipeline
        
        complex_pipeline = router.get_pipeline_for_complexity(QueryComplexity.COMPLEX)
        assert "planner" in complex_pipeline
        assert "validator" in complex_pipeline


class TestQueryPlanner:
    """Tests for query planner agent."""
    
    @pytest.fixture
    def planner(self):
        """Create planner instance."""
        with patch('app.agents.planner.AsyncOpenAI'):
            return QueryPlanner()
    
    def test_initialization(self, planner):
        """Test planner initializes correctly."""
        assert planner.model is not None
    
    def test_map_doc_type(self, planner):
        """Test document type mapping."""
        assert planner._map_doc_type("10-K") == DocumentType.SEC_10K
        assert planner._map_doc_type("10-Q") == DocumentType.SEC_10Q
        assert planner._map_doc_type("earnings_call") == DocumentType.EARNINGS_CALL
    
    def test_extract_json(self, planner):
        """Test JSON extraction from text."""
        text = 'Here is the result: [{"key": "value"}] and some more text'
        json_str = planner._extract_json(text)
        
        assert json_str == '[{"key": "value"}]'
    
    def test_extract_json_no_array(self, planner):
        """Test JSON extraction when no array present."""
        text = "No JSON here"
        json_str = planner._extract_json(text)
        
        assert json_str == text


class TestRetrieverAgent:
    """Tests for retriever agent."""
    
    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        return RetrieverAgent()
    
    def test_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.MAX_ITERATIONS == 3
        assert agent.MIN_RELEVANCE_SCORE == 0.5
    
    def test_validate_results_empty(self, agent):
        """Test validation with empty results."""
        assert agent._validate_results([], "test query") is False
    
    def test_merge_results(self, agent):
        """Test result merging and deduplication."""
        from app.models import RetrievedDocument, DocumentChunk, DocumentMetadata
        from datetime import datetime
        
        # Create mock documents
        meta = DocumentMetadata(
            ticker="AAPL",
            company_name="Apple",
            document_type=DocumentType.SEC_10K,
            filing_date=datetime.now(),
            source_url="http://example.com"
        )
        
        chunk1 = DocumentChunk(
            chunk_id="chunk1",
            document_id="doc1",
            content="Content 1",
            metadata=meta,
            chunk_index=0
        )
        
        chunk2 = DocumentChunk(
            chunk_id="chunk2",
            document_id="doc1",
            content="Content 2",
            metadata=meta,
            chunk_index=1
        )
        
        doc1 = RetrievedDocument(chunk=chunk1, score=0.9, retrieval_method="dense")
        doc2 = RetrievedDocument(chunk=chunk2, score=0.8, retrieval_method="dense")
        doc1_dup = RetrievedDocument(chunk=chunk1, score=0.85, retrieval_method="sparse")
        
        merged = agent._merge_results([[doc1, doc2], [doc1_dup]])
        
        # Should have 2 unique documents
        assert len(merged) == 2
        # Should be sorted by score
        assert merged[0].score >= merged[1].score


class TestValidator:
    """Tests for validator agent."""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return Validator()
    
    def test_initialization(self, validator):
        """Test validator initializes correctly."""
        assert validator.MIN_VALID_SCORE == 70
        assert validator.MIN_CITATION_COVERAGE == 0.5
    
    def test_check_citation_coverage(self, validator):
        """Test citation coverage checking."""
        # Response with cited numerical claims
        response = "Revenue was $100 billion [1]. Growth was 15% [2]."
        citations = [{"citation_id": "1"}, {"citation_id": "2"}]
        
        score, issues = validator.check_citation_coverage(response, citations)
        
        assert score == 100.0
        assert len(issues) == 0
    
    def test_check_citation_coverage_missing(self, validator):
        """Test citation coverage with missing citations."""
        response = "Revenue was $100 billion. Growth was 15%."
        citations = []
        
        score, issues = validator.check_citation_coverage(response, citations)
        
        assert score < 100.0
        assert len(issues) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
