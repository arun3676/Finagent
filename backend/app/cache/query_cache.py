"""
Semantic Query Cache

Caches query-response pairs with semantic similarity matching to avoid
redundant LLM calls for similar queries.

Features:
- Embeds incoming queries using OpenAI embedding service
- Compares against cached embeddings using cosine similarity
- LRU eviction when cache exceeds max size
- TTL-based expiration for data freshness

Usage:
    cache = SemanticQueryCache(embedding_service)
    cached = await cache.get(query)
    if cached:
        return cached
    # ... generate response ...
    await cache.set(query, response, citations)
"""

import asyncio
import logging
import time
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CachedResponse:
    """Response data returned from cache."""
    response: str
    citations: List[Dict[str, Any]]
    cache_hit: bool = True
    cached_at: float = field(default_factory=time.time)


@dataclass
class QueryCacheEntry:
    """A single cache entry with query embedding and response."""
    query_text: str
    query_embedding: List[float]
    response: str
    citations: List[Dict[str, Any]]
    timestamp: float
    ttl_hours: float
    access_time: float = field(default_factory=time.time)
    
    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL."""
        age_hours = (time.time() - self.timestamp) / 3600
        return age_hours > self.ttl_hours
    
    def touch(self):
        """Update access time for LRU tracking."""
        self.access_time = time.time()


class SemanticQueryCache:
    """
    Semantic cache for query-response pairs.
    
    Uses embedding similarity to match queries, with LRU eviction
    and TTL-based expiration.
    """
    
    def __init__(
        self,
        embedding_service,
        max_size: int = 1000,
        default_ttl_hours: float = 24.0,
        similarity_threshold: float = 0.92
    ):
        """
        Initialize semantic query cache.
        
        Args:
            embedding_service: Service for generating query embeddings
            max_size: Maximum number of entries in cache
            default_ttl_hours: Default time-to-live in hours (24h for SEC data)
            similarity_threshold: Cosine similarity threshold for cache hits (0.92)
        """
        self.embedding_service = embedding_service
        self.max_size = max_size
        self.default_ttl_hours = default_ttl_hours
        self.similarity_threshold = similarity_threshold
        
        # Cache storage: hash -> entry
        self._cache: Dict[str, QueryCacheEntry] = {}
        
        # Thread-safe lock for cache operations
        self._lock = asyncio.Lock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        
        logger.info(
            f"Initialized SemanticQueryCache: max_size={max_size}, "
            f"ttl={default_ttl_hours}h, threshold={similarity_threshold}"
        )
    
    async def get(self, query: str) -> Optional[CachedResponse]:
        """
        Check cache for a semantically similar query.
        
        Args:
            query: The incoming query text
            
        Returns:
            CachedResponse if similar query found and valid, None otherwise
        """
        async with self._lock:
            # First, evict expired entries
            self._evict_expired()
            
            if not self._cache:
                self._misses += 1
                return None
        
        try:
            # Embed the query (outside lock to avoid blocking)
            query_embedding = await self.embedding_service.embed_query(query)
            query_embedding_np = np.array(query_embedding)
            
            async with self._lock:
                best_match: Optional[QueryCacheEntry] = None
                best_similarity = 0.0
                
                # Find most similar cached query
                for entry in self._cache.values():
                    if entry.is_expired():
                        continue
                    
                    # Calculate cosine similarity
                    cached_embedding = np.array(entry.query_embedding)
                    similarity = self._cosine_similarity(
                        query_embedding_np, 
                        cached_embedding
                    )
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = entry
                
                # Check if similarity meets threshold
                if best_match and best_similarity >= self.similarity_threshold:
                    best_match.touch()  # Update LRU access time
                    self._hits += 1
                    
                    logger.info(
                        f"Cache HIT: similarity={best_similarity:.3f} "
                        f"for query '{query[:50]}...'"
                    )
                    
                    return CachedResponse(
                        response=best_match.response,
                        citations=best_match.citations,
                        cache_hit=True,
                        cached_at=best_match.timestamp
                    )
                
                self._misses += 1
                logger.debug(
                    f"Cache MISS: best_similarity={best_similarity:.3f} "
                    f"(threshold={self.similarity_threshold})"
                )
                return None
                
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            self._misses += 1
            return None
    
    async def set(
        self,
        query: str,
        response: str,
        citations: List[Dict[str, Any]],
        ttl_hours: Optional[float] = None
    ) -> None:
        """
        Store a query-response pair in the cache.
        
        Args:
            query: The query text
            response: The generated response
            citations: List of citation dictionaries
            ttl_hours: Optional custom TTL (uses default if not specified)
        """
        try:
            # Embed the query
            query_embedding = await self.embedding_service.embed_query(query)
            
            # Create cache key (hash of query for quick lookup)
            cache_key = self._get_cache_key(query)
            
            # Serialize citations if they're Pydantic models
            serialized_citations = []
            for c in citations:
                if hasattr(c, 'model_dump'):
                    serialized_citations.append(c.model_dump())
                elif isinstance(c, dict):
                    serialized_citations.append(c)
                else:
                    serialized_citations.append({"value": str(c)})
            
            entry = QueryCacheEntry(
                query_text=query,
                query_embedding=query_embedding,
                response=response,
                citations=serialized_citations,
                timestamp=time.time(),
                ttl_hours=ttl_hours or self.default_ttl_hours
            )
            
            async with self._lock:
                # Evict if at capacity
                if len(self._cache) >= self.max_size:
                    self._evict_lru()
                
                self._cache[cache_key] = entry
                
            logger.info(f"Cached response for query '{query[:50]}...'")
            
        except Exception as e:
            logger.error(f"Cache set failed: {e}")
    
    def _evict_expired(self) -> int:
        """
        Remove entries that have exceeded their TTL.
        
        Returns:
            Number of entries evicted
        """
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.debug(f"Evicted {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def _evict_lru(self) -> None:
        """Remove the least recently used entry."""
        if not self._cache:
            return
        
        # Find entry with oldest access time
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].access_time
        )
        
        del self._cache[lru_key]
        logger.debug(f"Evicted LRU cache entry: {lru_key[:16]}...")
    
    def _cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (0-1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _get_cache_key(self, query: str) -> str:
        """Generate a hash key for a query."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with hits, misses, hit_rate, and size
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": total,
            "hit_rate": round(hit_rate, 4),
            "hit_rate_percent": f"{hit_rate * 100:.1f}%",
            "size": len(self._cache),
            "max_size": self.max_size,
            "similarity_threshold": self.similarity_threshold,
            "default_ttl_hours": self.default_ttl_hours
        }
    
    async def clear(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of entries cleared
        """
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            logger.info(f"Cleared {count} cache entries")
            return count
    
    async def invalidate_for_ticker(self, ticker: str) -> int:
        """
        Invalidate cache entries related to a specific ticker.
        
        Useful when new data is ingested for a company.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Number of entries invalidated
        """
        async with self._lock:
            ticker_lower = ticker.lower()
            keys_to_remove = [
                key for key, entry in self._cache.items()
                if ticker_lower in entry.query_text.lower()
            ]
            
            for key in keys_to_remove:
                del self._cache[key]
            
            if keys_to_remove:
                logger.info(f"Invalidated {len(keys_to_remove)} cache entries for {ticker}")
            
            return len(keys_to_remove)
