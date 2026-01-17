"""
Test Response Length Feature

Tests the response length selection functionality across the FinAgent system.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from app.models import AgentState, ResponseLength, RetrievedDocument, DocumentChunk, DocumentMetadata, DocumentType
from app.agents.synthesizer import Synthesizer
from app.agents.fast_synthesizer import FastSynthesizer
from app.agents.prompts import (
    get_synthesizer_prompt_for_length,
    get_fast_synthesis_prompt_for_length,
    get_max_tokens_for_length
)
from datetime import datetime


class TestResponseLengthPrompts:
    """Test prompt selection based on response length."""
    
    def test_synthesizer_prompt_selection(self):
        """Test that correct synthesizer prompts are selected."""
        short_prompt = get_synthesizer_prompt_for_length("short")
        normal_prompt = get_synthesizer_prompt_for_length("normal")
        detailed_prompt = get_synthesizer_prompt_for_length("detailed")
        
        assert "2-3 sentences maximum" in short_prompt
        assert "3-5 sentences" in normal_prompt
        assert "400-800 words" in detailed_prompt
        
    def test_fast_synthesis_prompt_selection(self):
        """Test that correct fast synthesis prompts are selected."""
        short_prompt = get_fast_synthesis_prompt_for_length("short")
        normal_prompt = get_fast_synthesis_prompt_for_length("normal")
        detailed_prompt = get_fast_synthesis_prompt_for_length("detailed")
        
        assert "1-2 sentences maximum" in short_prompt
        assert "Keep response under 150 words" in normal_prompt
        assert "200-400 words" in detailed_prompt
        
    def test_token_limits(self):
        """Test that appropriate token limits are set."""
        assert get_max_tokens_for_length("short") == 150
        assert get_max_tokens_for_length("normal") == 400
        assert get_max_tokens_for_length("detailed") == 1000
        
    def test_fallback_to_normal(self):
        """Test that invalid lengths fallback to normal."""
        assert get_max_tokens_for_length("invalid") == 400
        normal_prompt = get_synthesizer_prompt_for_length("normal")
        assert get_synthesizer_prompt_for_length("invalid") == normal_prompt


class TestAgentStateIntegration:
    """Test AgentState integration with response length."""
    
    def test_agent_state_default_length(self):
        """Test that AgentState defaults to normal length."""
        state = AgentState(original_query="Test query")
        assert state.response_length == ResponseLength.NORMAL
        
    def test_agent_state_custom_length(self):
        """Test that AgentState accepts custom length."""
        state = AgentState(
            original_query="Test query",
            response_length=ResponseLength.SHORT
        )
        assert state.response_length == ResponseLength.SHORT


@pytest.mark.asyncio
class TestSynthesizerIntegration:
    """Test synthesizer integration with response length."""
    
    async def test_synthesizer_length_parameter(self):
        """Test that Synthesizer accepts and uses response_length parameter."""
        synthesizer = Synthesizer()
        synthesizer.llm = Mock()
        
        # Mock the LLM chain
        mock_chain = AsyncMock()
        mock_chain.ainvoke = AsyncMock(return_value="Test response [1]")
        
        # Create mock documents
        metadata = DocumentMetadata(
            ticker="AAPL",
            company_name="Apple Inc.",
            document_type=DocumentType.SEC_10K,
            filing_date=datetime.now(),
            source_url="https://example.com"
        )
        
        chunk = DocumentChunk(
            chunk_id="test_chunk",
            document_id="test_doc",
            content="Test content about Apple's revenue.",
            metadata=metadata,
            chunk_index=0
        )
        
        doc = RetrievedDocument(
            chunk=chunk,
            score=0.9,
            retrieval_method="dense"
        )
        
        # Test that different lengths are handled
        for length in ["short", "normal", "detailed"]:
            # Mock the chain creation to avoid LLM calls
            with pytest.MonkeyPatch.context() as m:
                m.setattr(synthesizer, "_generate_response", AsyncMock(return_value=f"Response for {length} [1]"))
                
                response = await synthesizer.synthesize(
                    query="Test query",
                    analysis={},
                    documents=[doc],
                    response_length=length
                )
                
                assert f"Response for {length}" in response.answer
                
    async def test_fast_synthesizer_length_parameter(self):
        """Test that FastSynthesizer accepts and uses response_length parameter."""
        fast_synthesizer = FastSynthesizer()
        
        # Mock the LLM client
        fast_synthesizer.llm_client = Mock()
        fast_synthesizer.llm_client.generate = AsyncMock(return_value="Fast response [1]")
        
        # Create mock documents
        metadata = DocumentMetadata(
            ticker="AAPL",
            company_name="Apple Inc.",
            document_type=DocumentType.SEC_10K,
            filing_date=datetime.now(),
            source_url="https://example.com"
        )
        
        chunk = DocumentChunk(
            chunk_id="test_chunk",
            document_id="test_doc",
            content="Test content about Apple's revenue.",
            metadata=metadata,
            chunk_index=0
        )
        
        doc = RetrievedDocument(
            chunk=chunk,
            score=0.9,
            retrieval_method="dense"
        )
        
        # Test different response lengths
        for length in ["short", "normal", "detailed"]:
            result = await fast_synthesizer.synthesize(
                query="Test query",
                documents=[doc],
                response_length=length
            )
            
            assert "draft_response" in result
            assert "citations" in result


class TestAPIIntegration:
    """Test API integration with response length."""
    
    def test_query_request_model(self):
        """Test that QueryRequest model accepts response_length."""
        from app.main import QueryRequest
        
        # Test default
        request = QueryRequest(query="Test query")
        assert request.response_length == ResponseLength.NORMAL
        
        # Test custom length
        request = QueryRequest(
            query="Test query",
            response_length=ResponseLength.SHORT
        )
        assert request.response_length == ResponseLength.SHORT


if __name__ == "__main__":
    pytest.main([__file__])
