"""
Tests for Retrieval Module

Tests embeddings, vector store, BM25, hybrid search, and reranker.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.retrieval.embeddings import EmbeddingService
from app.retrieval.bm25_index import BM25Index
from app.retrieval.hybrid_search import HybridSearcher
from app.retrieval.reranker import Reranker


class TestEmbeddingService:
    """Tests for embedding service."""
    
    @pytest.fixture
    def service(self):
        """Create service instance with mocked client."""
        with patch('app.retrieval.embeddings.AsyncOpenAI'):
            return EmbeddingService(model="text-embedding-3-small", api_key="test-key")
    
    def test_initialization(self, service):
        """Test service initializes correctly."""
        assert service.model == "text-embedding-3-small"
        assert service.dimension == 1536
        assert service.cache_enabled is True
    
    def test_get_cache_key(self, service):
        """Test cache key generation."""
        key1 = service._get_cache_key("test text")
        key2 = service._get_cache_key("test text")
        key3 = service._get_cache_key("different text")
        
        assert key1 == key2
        assert key1 != key3
    
    def test_estimate_cost(self, service):
        """Test cost estimation."""
        texts = ["Short text", "Another short text"]
        estimate = service.estimate_cost(texts)
        
        assert estimate["num_texts"] == 2
        assert estimate["estimated_tokens"] > 0
        assert estimate["estimated_cost_usd"] >= 0
        assert estimate["model"] == "text-embedding-3-small"
    
    def test_clear_cache(self, service):
        """Test cache clearing."""
        service._cache["key1"] = [0.1, 0.2]
        service._cache["key2"] = [0.3, 0.4]
        
        count = service.clear_cache()
        
        assert count == 2
        assert len(service._cache) == 0


class TestBM25Index:
    """Tests for BM25 index."""
    
    @pytest.fixture
    def index(self):
        """Create index instance."""
        return BM25Index(k1=1.5, b=0.75)
    
    def test_initialization(self, index):
        """Test index initializes correctly."""
        assert index.k1 == 1.5
        assert index.b == 0.75
        assert index._total_docs == 0
    
    def test_tokenize(self, index):
        """Test tokenization."""
        tokens = index.tokenize("Apple Inc. reported $100 billion revenue")
        
        assert "apple" in tokens
        assert "inc" in tokens
        assert "100" in tokens
        assert "billion" in tokens
        # Stopwords should be filtered
        assert "the" not in tokens
    
    def test_tokenize_preserves_tickers(self, index):
        """Test that short tokens (potential tickers) are preserved."""
        tokens = index.tokenize("AAPL stock price increased")
        
        # AAPL should be lowercased but preserved
        assert "aapl" in tokens
    
    def test_get_stats_empty(self, index):
        """Test stats on empty index."""
        stats = index.get_stats()
        
        assert stats["total_documents"] == 0
        assert stats["vocabulary_size"] == 0


class TestHybridSearcher:
    """Tests for hybrid search."""
    
    @pytest.fixture
    def searcher(self):
        """Create searcher with mocked components."""
        vector_store = MagicMock()
        bm25_index = MagicMock()
        embedding_service = MagicMock()
        
        return HybridSearcher(
            vector_store=vector_store,
            bm25_index=bm25_index,
            embedding_service=embedding_service,
            alpha=0.5
        )
    
    def test_initialization(self, searcher):
        """Test searcher initializes correctly."""
        assert searcher.alpha == 0.5
        assert searcher.RRF_K == 60
    
    def test_fuse_rrf(self, searcher):
        """Test RRF fusion."""
        dense_results = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        sparse_results = [("doc2", 0.95), ("doc1", 0.85), ("doc4", 0.75)]
        
        fused = searcher._fuse_rrf(dense_results, sparse_results, top_k=3)
        
        # doc1 and doc2 should be top since they appear in both
        fused_ids = [doc_id for doc_id, _ in fused]
        assert "doc1" in fused_ids[:2]
        assert "doc2" in fused_ids[:2]
    
    def test_normalize_scores(self, searcher):
        """Test score normalization."""
        results = [("doc1", 10.0), ("doc2", 5.0), ("doc3", 0.0)]
        normalized = searcher._normalize_scores(results)
        
        assert normalized["doc1"] == 1.0
        assert normalized["doc3"] == 0.0
        assert 0 < normalized["doc2"] < 1
    
    def test_normalize_scores_empty(self, searcher):
        """Test normalization with empty results."""
        normalized = searcher._normalize_scores([])
        assert normalized == {}


class TestReranker:
    """Tests for reranker."""
    
    @pytest.fixture
    def reranker(self):
        """Create reranker instance."""
        return Reranker(api_key="test-key", model="rerank-english-v3.0")
    
    def test_initialization(self, reranker):
        """Test reranker initializes correctly."""
        assert reranker.model == "rerank-english-v3.0"
    
    def test_is_available(self, reranker):
        """Test availability check."""
        assert reranker.is_available() is True
        
        no_key_reranker = Reranker(api_key=None)
        assert no_key_reranker.is_available() is False
    
    def test_estimate_cost(self, reranker):
        """Test cost estimation."""
        estimate = reranker.estimate_cost(num_documents=10)
        
        assert estimate["num_documents"] == 10
        assert estimate["estimated_cost_usd"] > 0
        assert estimate["model"] == "rerank-english-v3.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
