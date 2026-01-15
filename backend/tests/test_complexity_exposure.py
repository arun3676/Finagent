"""
Test Complexity Classification Exposure

Tests that query complexity classification is properly exposed in streaming responses
with user-friendly metadata for frontend display.
"""

import pytest
from unittest.mock import patch, AsyncMock
from app.agents.router import QueryRouter
from app.models import QueryComplexity, ComplexityInfo, AgentState


class TestComplexityInfo:
    """Test ComplexityInfo model and creation."""

    def test_complexity_info_creation(self):
        """Test ComplexityInfo can be created with all required fields."""
        info = ComplexityInfo(
            level=QueryComplexity.COMPLEX,
            display_label="Deep Research",
            display_color="purple",
            estimated_time_seconds=25,
            reasoning="Multi-company comparison with deep analysis",
            features_enabled=["planner", "retriever", "analyst", "synthesizer", "validator"]
        )

        assert info.level == QueryComplexity.COMPLEX
        assert info.display_label == "Deep Research"
        assert info.display_color == "purple"
        assert info.estimated_time_seconds == 25
        assert len(info.features_enabled) == 5

    def test_complexity_info_serialization(self):
        """Test ComplexityInfo can be serialized for JSON response."""
        info = ComplexityInfo(
            level=QueryComplexity.SIMPLE,
            display_label="Quick Look",
            display_color="green",
            estimated_time_seconds=3,
            reasoning="Single fact lookup",
            features_enabled=["retriever", "synthesizer"]
        )

        data = info.model_dump()

        assert data["level"] == "simple"
        assert data["display_label"] == "Quick Look"
        assert data["display_color"] == "green"
        assert isinstance(data["estimated_time_seconds"], int)


class TestRouterComplexityInfo:
    """Test Router's complexity info building."""

    def test_build_complexity_info_simple(self):
        """Test building complexity info for simple query."""
        router = QueryRouter(use_llm=False)

        info = router._build_complexity_info(
            QueryComplexity.SIMPLE,
            "What is Apple's revenue?"
        )

        assert info.level == QueryComplexity.SIMPLE
        assert info.display_label == "Quick Look"
        assert info.display_color == "green"
        assert info.estimated_time_seconds == 3
        assert "retriever" in info.features_enabled
        assert "synthesizer" in info.features_enabled
        assert "planner" not in info.features_enabled

    def test_build_complexity_info_moderate(self):
        """Test building complexity info for moderate query."""
        router = QueryRouter(use_llm=False)

        info = router._build_complexity_info(
            QueryComplexity.MODERATE,
            "Compare Apple and Microsoft revenue"
        )

        assert info.level == QueryComplexity.MODERATE
        assert info.display_label == "Analysis"
        assert info.display_color == "blue"
        assert info.estimated_time_seconds == 10
        assert "analyst" in info.features_enabled
        assert "validator" in info.features_enabled

    def test_build_complexity_info_complex(self):
        """Test building complexity info for complex query."""
        router = QueryRouter(use_llm=False)

        info = router._build_complexity_info(
            QueryComplexity.COMPLEX,
            "Analyze the risk factors and their impact on revenue growth"
        )

        assert info.level == QueryComplexity.COMPLEX
        assert info.display_label == "Deep Research"
        assert info.display_color == "purple"
        assert info.estimated_time_seconds == 25
        assert "planner" in info.features_enabled
        assert len(info.features_enabled) == 5

    def test_generate_reasoning_simple(self):
        """Test reasoning generation for simple queries."""
        router = QueryRouter(use_llm=False)

        reasoning = router._generate_reasoning(
            QueryComplexity.SIMPLE,
            "What is Microsoft's CEO?"
        )

        assert "fact lookup" in reasoning.lower() or "retrieval" in reasoning.lower()

    def test_generate_reasoning_moderate_comparison(self):
        """Test reasoning generation for comparison queries."""
        router = QueryRouter(use_llm=False)

        reasoning = router._generate_reasoning(
            QueryComplexity.MODERATE,
            "Compare AAPL vs MSFT revenue"
        )

        assert "comparison" in reasoning.lower()

    def test_generate_reasoning_complex_risk(self):
        """Test reasoning generation for complex risk queries."""
        router = QueryRouter(use_llm=False)

        reasoning = router._generate_reasoning(
            QueryComplexity.COMPLEX,
            "Analyze risk factors and their impact"
        )

        assert "risk" in reasoning.lower() or "analysis" in reasoning.lower()

    @pytest.mark.asyncio
    async def test_route_returns_complexity_info(self):
        """Test that route method returns both complexity and complexity_info."""
        router = QueryRouter(use_llm=False)

        state = AgentState(original_query="What is Apple's revenue?")
        result = await router.route(state)

        assert "complexity" in result
        assert "complexity_info" in result
        assert isinstance(result["complexity_info"], ComplexityInfo)
        assert result["complexity_info"].level == result["complexity"]


class TestAgentStateComplexityInfo:
    """Test AgentState includes complexity_info."""

    def test_agent_state_with_complexity_info(self):
        """Test AgentState can store complexity_info."""
        info = ComplexityInfo(
            level=QueryComplexity.MODERATE,
            display_label="Analysis",
            display_color="blue",
            estimated_time_seconds=10,
            reasoning="Multi-step reasoning required",
            features_enabled=["retriever", "analyst", "synthesizer"]
        )

        state = AgentState(
            original_query="Test query",
            complexity=QueryComplexity.MODERATE,
            complexity_info=info
        )

        assert state.complexity_info is not None
        assert state.complexity_info.level == QueryComplexity.MODERATE
        assert state.complexity_info.display_label == "Analysis"

    def test_agent_state_without_complexity_info(self):
        """Test AgentState works without complexity_info (optional)."""
        state = AgentState(
            original_query="Test query",
            complexity=QueryComplexity.SIMPLE
        )

        assert state.complexity_info is None


class TestComplexityDisplayMetadata:
    """Test display metadata is correct for each complexity level."""

    def test_simple_query_metadata(self):
        """Test SIMPLE query display metadata."""
        router = QueryRouter(use_llm=False)
        info = router._build_complexity_info(QueryComplexity.SIMPLE, "test")

        # Quick Look with green color, ~3 seconds
        assert info.display_label == "Quick Look"
        assert info.display_color == "green"
        assert info.estimated_time_seconds <= 5
        # Should skip planner and analyst
        assert "planner" not in info.features_enabled
        assert "analyst" not in info.features_enabled

    def test_moderate_query_metadata(self):
        """Test MODERATE query display metadata."""
        router = QueryRouter(use_llm=False)
        info = router._build_complexity_info(QueryComplexity.MODERATE, "test")

        # Analysis with blue color, ~10 seconds
        assert info.display_label == "Analysis"
        assert info.display_color == "blue"
        assert 5 < info.estimated_time_seconds <= 15
        # Should include analyst but not planner
        assert "analyst" in info.features_enabled
        assert "planner" not in info.features_enabled

    def test_complex_query_metadata(self):
        """Test COMPLEX query display metadata."""
        router = QueryRouter(use_llm=False)
        info = router._build_complexity_info(QueryComplexity.COMPLEX, "test")

        # Deep Research with purple color, ~25 seconds
        assert info.display_label == "Deep Research"
        assert info.display_color == "purple"
        assert info.estimated_time_seconds > 15
        # Should include all features
        assert "planner" in info.features_enabled
        assert "analyst" in info.features_enabled
        assert "validator" in info.features_enabled


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
