"""
Embedding Service

Generates vector embeddings using OpenAI's embedding models.
Handles batching, caching, and error recovery.

Supported models:
- text-embedding-3-small (1536 dims, recommended)
- text-embedding-3-large (3072 dims, higher quality)
- text-embedding-ada-002 (1536 dims, legacy)

Usage:
    service = EmbeddingService()
    embeddings = await service.embed_texts(["Hello world", "Financial analysis"])
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
import hashlib
from tenacity import retry, stop_after_attempt, wait_exponential

from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings via OpenAI API.
    
    Features:
    - Async batch processing
    - Automatic retry with exponential backoff
    - Optional caching layer
    - Token counting and cost estimation
    """
    
    # Model specifications
    MODEL_SPECS = {
        "text-embedding-3-small": {"dims": 1536, "max_tokens": 8191, "cost_per_1k": 0.00002},
        "text-embedding-3-large": {"dims": 3072, "max_tokens": 8191, "cost_per_1k": 0.00013},
        "text-embedding-ada-002": {"dims": 1536, "max_tokens": 8191, "cost_per_1k": 0.0001}
    }
    
    # Batch size for API calls
    BATCH_SIZE = 100
    
    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        cache_enabled: bool = True
    ):
        """
        Initialize embedding service.
        
        Args:
            model: OpenAI embedding model name
            api_key: OpenAI API key (uses env var if not provided)
            cache_enabled: Enable embedding cache
        """
        self.model = model or settings.EMBEDDING_MODEL
        self.client = AsyncOpenAI(api_key=api_key or settings.OPENAI_API_KEY)
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, List[float]] = {}
        
        if self.model not in self.MODEL_SPECS:
            raise ValueError(f"Unknown model: {self.model}")
        
        self.dimension = self.MODEL_SPECS[self.model]["dims"]
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        embeddings = await self.embed_texts([text])
        return embeddings[0]
    
    async def embed_texts(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            show_progress: Show progress bar
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        embeddings = [None] * len(texts)
        texts_to_embed = []
        indices_to_embed = []
        
        for i, text in enumerate(texts):
            cached = self._check_cache(text)
            if cached is not None:
                embeddings[i] = cached
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
        
        if texts_to_embed:
            logger.info(f"Embedding {len(texts_to_embed)} texts (cache hit: {len(texts) - len(texts_to_embed)})")
            
            for i in range(0, len(texts_to_embed), self.BATCH_SIZE):
                batch = texts_to_embed[i:i + self.BATCH_SIZE]
                batch_indices = indices_to_embed[i:i + self.BATCH_SIZE]
                
                batch_embeddings = await self._embed_batch(batch)
                
                for j, (text, embedding) in enumerate(zip(batch, batch_embeddings)):
                    self._add_to_cache(text, embedding)
                    embeddings[batch_indices[j]] = embedding
        
        return embeddings
    
    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        
        Note: Some models use different embeddings for queries vs documents.
        This method handles that distinction.
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding vector
        """
        # For OpenAI models, query and document embeddings are the same
        return await self.embed_text(query)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts via API.
        
        Args:
            texts: Batch of texts (max BATCH_SIZE)
            
        Returns:
            List of embeddings
        """
        if len(texts) > self.BATCH_SIZE:
            raise ValueError(f"Batch size {len(texts)} exceeds maximum {self.BATCH_SIZE}")
        
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts,
                encoding_format="float"
            )
            
            embeddings = [item.embedding for item in response.data]
            return embeddings
            
        except Exception as e:
            logger.error(f"Error embedding batch: {e}")
            raise
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generate cache key for a text.
        
        Args:
            text: Text to hash
            
        Returns:
            Cache key string
        """
        return hashlib.md5(f"{self.model}:{text}".encode()).hexdigest()
    
    def _check_cache(self, text: str) -> Optional[List[float]]:
        """
        Check if embedding is cached.
        
        Args:
            text: Text to look up
            
        Returns:
            Cached embedding or None
        """
        if not self.cache_enabled:
            return None
        return self._cache.get(self._get_cache_key(text))
    
    def _add_to_cache(self, text: str, embedding: List[float]) -> None:
        """
        Add embedding to cache.
        
        Args:
            text: Original text
            embedding: Embedding vector
        """
        if self.cache_enabled:
            self._cache[self._get_cache_key(text)] = embedding
    
    def estimate_cost(self, texts: List[str]) -> Dict[str, Any]:
        """
        Estimate cost for embedding texts.
        
        Args:
            texts: Texts to estimate
            
        Returns:
            Cost estimation details
        """
        # Rough token estimate: ~4 chars per token
        total_chars = sum(len(t) for t in texts)
        estimated_tokens = total_chars / 4
        cost_per_1k = self.MODEL_SPECS[self.model]["cost_per_1k"]
        
        return {
            "num_texts": len(texts),
            "estimated_tokens": int(estimated_tokens),
            "estimated_cost_usd": (estimated_tokens / 1000) * cost_per_1k,
            "model": self.model
        }
    
    def clear_cache(self) -> int:
        """
        Clear the embedding cache.
        
        Returns:
            Number of entries cleared
        """
        count = len(self._cache)
        self._cache.clear()
        return count
