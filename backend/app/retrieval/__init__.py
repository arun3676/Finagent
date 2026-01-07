"""
Retrieval Module

Hybrid retrieval system combining dense and sparse methods:
- OpenAI embeddings for semantic search
- BM25 for keyword matching
- Reciprocal Rank Fusion for combining results
- Cohere reranking for final ordering
"""

from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_index import BM25Index
from app.retrieval.hybrid_search import HybridSearcher
from app.retrieval.reranker import Reranker

__all__ = [
    "EmbeddingService",
    "VectorStore", 
    "BM25Index",
    "HybridSearcher",
    "Reranker"
]
