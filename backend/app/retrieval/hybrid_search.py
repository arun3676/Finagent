"""
Hybrid Search Module

Combines dense (vector) and sparse (BM25) retrieval using
Reciprocal Rank Fusion (RRF) for optimal results.

Why hybrid search?
- Dense: Semantic understanding, handles paraphrasing
- Sparse: Exact matching, rare terms, specific entities
- Combined: Best of both worlds

Usage:
    searcher = HybridSearcher(vector_store, bm25_index)
    results = await searcher.search("Apple revenue growth YoY", top_k=10)
"""

from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict
import logging

from app.config import settings
from app.models import DocumentChunk, RetrievedDocument
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_index import BM25Index
from app.retrieval.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class HybridSearcher:
    """
    Hybrid search combining dense and sparse retrieval.
    
    Fusion methods:
    - RRF (Reciprocal Rank Fusion): Default, robust
    - Linear: Weighted combination of scores
    - Convex: Normalized score combination
    """
    
    # RRF constant (typically 60)
    RRF_K = 60
    
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        embedding_service: EmbeddingService,
        alpha: float = None
    ):
        """
        Initialize hybrid searcher.
        
        Args:
            vector_store: Dense vector store
            bm25_index: Sparse BM25 index
            embedding_service: Service for query embedding
            alpha: Weight for dense vs sparse (0=sparse, 1=dense)
        """
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedding_service = embedding_service
        self.alpha = alpha if alpha is not None else settings.HYBRID_ALPHA
    
    async def search(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None,
        fusion_method: str = "rrf"
    ) -> List[RetrievedDocument]:
        """
        Perform hybrid search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Metadata filters
            fusion_method: "rrf", "linear", or "convex"
            
        Returns:
            List of retrieved documents with fused scores
        """
        top_k = top_k or settings.RETRIEVAL_TOP_K
        
        logger.info(f"Hybrid search: '{query}' (top_k={top_k}, fusion={fusion_method})")
        
        dense_results = await self._dense_search(query, top_k * 2, filters)
        sparse_results = self._sparse_search(query, top_k * 2, filters)
        
        logger.debug(f"Dense: {len(dense_results)} results, Sparse: {len(sparse_results)} results")
        
        if fusion_method == "rrf":
            fused = self._fuse_rrf(dense_results, sparse_results, top_k)
        elif fusion_method == "linear":
            fused = self._fuse_linear(dense_results, sparse_results, top_k)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        documents = await self._get_documents(fused)
        
        logger.info(f"Returned {len(documents)} fused results")
        return documents
    
    async def _dense_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[Tuple[str, float]]:
        """
        Perform dense vector search.
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Metadata filters
            
        Returns:
            List of (chunk_id, score) tuples
        """
        query_embedding = await self.embedding_service.embed_query(query)
        
        results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters
        )
        
        return [(doc.chunk.chunk_id, doc.score) for doc in results]
    
    def _sparse_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[Tuple[str, float]]:
        """
        Perform sparse BM25 search.
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Metadata filters
            
        Returns:
            List of (chunk_id, score) tuples
        """
        results = self.bm25_index.search(
            query=query,
            top_k=top_k,
            filters=filters
        )
        
        return [(doc.chunk.chunk_id, doc.score) for doc in results]
    
    def _fuse_rrf(
        self,
        dense_results: List[Tuple[str, float]],
        sparse_results: List[Tuple[str, float]],
        top_k: int
    ) -> List[Tuple[str, float]]:
        """
        Fuse results using Reciprocal Rank Fusion.
        
        RRF score = Σ 1 / (k + rank)
        
        Args:
            dense_results: Dense search results
            sparse_results: Sparse search results
            top_k: Number of results to return
            
        Returns:
            Fused results sorted by RRF score
        """
        scores = defaultdict(float)
        
        # Score from dense results
        for rank, (chunk_id, _) in enumerate(dense_results):
            scores[chunk_id] += 1.0 / (self.RRF_K + rank + 1)
        
        # Score from sparse results
        for rank, (chunk_id, _) in enumerate(sparse_results):
            scores[chunk_id] += 1.0 / (self.RRF_K + rank + 1)
        
        # Sort by fused score
        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return fused[:top_k]
    
    def _fuse_linear(
        self,
        dense_results: List[Tuple[str, float]],
        sparse_results: List[Tuple[str, float]],
        top_k: int
    ) -> List[Tuple[str, float]]:
        """
        Fuse results using linear combination.
        
        score = alpha * dense_score + (1 - alpha) * sparse_score
        
        Args:
            dense_results: Dense search results
            sparse_results: Sparse search results
            top_k: Number of results to return
            
        Returns:
            Fused results sorted by combined score
        """
        # Normalize scores to [0, 1]
        dense_scores = self._normalize_scores(dense_results)
        sparse_scores = self._normalize_scores(sparse_results)
        
        # Combine scores
        all_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
        combined = {}
        
        for chunk_id in all_ids:
            d_score = dense_scores.get(chunk_id, 0.0)
            s_score = sparse_scores.get(chunk_id, 0.0)
            combined[chunk_id] = self.alpha * d_score + (1 - self.alpha) * s_score
        
        # Sort and return top-k
        fused = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return fused[:top_k]
    
    def _normalize_scores(
        self,
        results: List[Tuple[str, float]]
    ) -> Dict[str, float]:
        """
        Normalize scores to [0, 1] range.
        
        Args:
            results: List of (chunk_id, score) tuples
            
        Returns:
            Dictionary of normalized scores
        """
        if not results:
            return {}
        
        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return {chunk_id: 1.0 for chunk_id, _ in results}
        
        return {
            chunk_id: (score - min_score) / (max_score - min_score)
            for chunk_id, score in results
        }
    
    async def _get_documents(
        self,
        fused_results: List[Tuple[str, float]]
    ) -> List[RetrievedDocument]:
        """
        Retrieve full documents for fused results.
        
        Args:
            fused_results: List of (chunk_id, score) tuples
            
        Returns:
            List of RetrievedDocument objects
        """
        documents = []
        
        for chunk_id, score in fused_results:
            chunk = await self.vector_store.search_by_id(chunk_id)
            if chunk:
                documents.append(RetrievedDocument(
                    chunk=chunk,
                    score=score,
                    retrieval_method="hybrid"
                ))
        
        return documents
