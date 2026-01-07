"""
Reranker Module

Uses Cohere's cross-encoder model to rerank retrieved documents.
Cross-encoders provide more accurate relevance scores than bi-encoders.

Why reranking?
- Initial retrieval optimizes for recall
- Reranking optimizes for precision
- Cross-encoders see query and document together

Usage:
    reranker = Reranker()
    reranked = await reranker.rerank(query, documents, top_k=5)
"""

from typing import List, Optional, Dict, Any
import logging
import cohere

from app.config import settings
from app.models import RetrievedDocument

logger = logging.getLogger(__name__)


class Reranker:
    """
    Cohere-based document reranker.
    
    Models:
    - rerank-english-v3.0: Best quality
    - rerank-multilingual-v3.0: Multi-language support
    - rerank-english-v2.0: Faster, good quality
    """
    
    DEFAULT_MODEL = "rerank-english-v3.0"
    
    def __init__(
        self,
        api_key: str = None,
        model: str = None
    ):
        """
        Initialize Cohere reranker.
        
        Args:
            api_key: Cohere API key
            model: Reranking model name
        """
        self.api_key = api_key or settings.COHERE_API_KEY
        self.model = model or self.DEFAULT_MODEL
        self.client = cohere.Client(self.api_key) if self.api_key else None
    
    async def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: int = None
    ) -> List[RetrievedDocument]:
        """
        Rerank documents using Cohere cross-encoder.
        
        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Number of documents to return
            
        Returns:
            Reranked documents with updated scores
        """
        if not documents:
            return []
        
        if not self.client:
            logger.warning("Cohere API key not configured, skipping reranking")
            return documents[:top_k]
        
        top_k = top_k or settings.RERANK_TOP_K
        
        logger.info(f"Reranking {len(documents)} documents with {self.model}")
        
        doc_texts = self._prepare_documents(documents)
        
        try:
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=doc_texts,
                top_n=top_k,
                return_documents=False
            )
            
            reranked = []
            for result in response.results:
                idx = result.index
                reranked_doc = documents[idx]
                reranked_doc.score = result.relevance_score
                reranked.append(reranked_doc)
            
            logger.info(f"Reranked to top {len(reranked)} documents")
            return reranked
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return documents[:top_k]
    
    async def rerank_with_metadata(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: int = None,
        boost_recent: bool = True
    ) -> List[RetrievedDocument]:
        """
        Rerank with metadata-based score adjustments.
        
        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Number of documents to return
            boost_recent: Boost more recent documents
            
        Returns:
            Reranked documents with adjusted scores
        """
        reranked = await self.rerank(query, documents, top_k=len(documents))
        
        if boost_recent:
            scores = [doc.score for doc in reranked]
            adjusted_scores = self._apply_recency_boost(reranked, scores)
            
            for doc, adjusted_score in zip(reranked, adjusted_scores):
                doc.score = adjusted_score
            
            reranked.sort(key=lambda x: x.score, reverse=True)
        
        return reranked[:top_k or settings.RERANK_TOP_K]
    
    def _prepare_documents(
        self,
        documents: List[RetrievedDocument]
    ) -> List[str]:
        """
        Prepare document texts for reranking.
        
        Args:
            documents: Retrieved documents
            
        Returns:
            List of document texts
        """
        texts = []
        for doc in documents:
            # Include section context in text
            section = doc.chunk.section or "Unknown Section"
            text = f"[{section}] {doc.chunk.content}"
            texts.append(text)
        return texts
    
    def _apply_recency_boost(
        self,
        documents: List[RetrievedDocument],
        scores: List[float],
        decay_factor: float = 0.1
    ) -> List[float]:
        """
        Apply recency boost to scores.
        
        More recent documents get higher scores.
        
        Args:
            documents: Retrieved documents
            scores: Base reranking scores
            decay_factor: How much to boost recent docs
            
        Returns:
            Adjusted scores
        """
        from datetime import datetime
        
        adjusted = []
        now = datetime.now()
        
        for doc, score in zip(documents, scores):
            filing_date = doc.chunk.metadata.filing_date
            days_old = (now - filing_date).days
            
            # Exponential decay boost
            recency_boost = 1.0 / (1.0 + decay_factor * days_old / 365)
            adjusted.append(score * (1 + 0.1 * recency_boost))
        
        return adjusted
    
    def estimate_cost(self, num_documents: int) -> Dict[str, Any]:
        """
        Estimate cost for reranking.
        
        Args:
            num_documents: Number of documents to rerank
            
        Returns:
            Cost estimation
        """
        # Cohere pricing: $1 per 1000 searches
        # Each rerank call counts as one search
        cost_per_search = 0.001
        
        return {
            "num_documents": num_documents,
            "estimated_cost_usd": cost_per_search,
            "model": self.model
        }
    
    def is_available(self) -> bool:
        """
        Check if reranker is properly configured.
        
        Returns:
            True if API key is set
        """
        return bool(self.api_key)
