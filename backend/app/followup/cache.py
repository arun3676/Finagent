"""
Chunk Cache for Follow-Up Questions

Provides in-memory caching of retrieved chunks with TTL
to enable fast follow-up question execution.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from app.models import DocumentChunk

logger = logging.getLogger(__name__)


class CacheEntry(BaseModel):
    """A cached query result with chunks and metadata."""
    query_id: str = Field(..., description="Unique query identifier")
    query_text: str = Field(..., description="Original query text")
    chunks: List[DocumentChunk] = Field(default_factory=list, description="Retrieved document chunks")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Query metadata (ticker, etc.)")
    created_at: datetime = Field(default_factory=datetime.now, description="Cache entry creation time")
    follow_up_questions: List[Any] = Field(default_factory=list, description="Generated follow-up questions")
    companies: List[str] = Field(default_factory=list, description="Companies mentioned in query")
    metrics: List[str] = Field(default_factory=list, description="Metrics mentioned in response")
    response_summary: str = Field(default="", description="Summary of the response")


class ChunkCache:
    """
    In-memory cache for retrieved chunks with TTL.

    Enables fast follow-up question execution by storing
    chunks from the parent query.
    """

    def __init__(self, ttl_seconds: int = 600, max_entries: int = 100):
        """
        Initialize the chunk cache.

        Args:
            ttl_seconds: Time-to-live for cache entries (default: 10 minutes)
            max_entries: Maximum number of cached queries
        """
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()

        # Stats tracking
        self._hits = 0
        self._misses = 0

        logger.info(f"ChunkCache initialized: TTL={ttl_seconds}s, max_entries={max_entries}")

    async def store(
        self,
        query_id: str,
        query_text: str,
        chunks: List[DocumentChunk],
        metadata: Optional[Dict[str, Any]] = None,
        companies: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        response_summary: str = ""
    ) -> None:
        """
        Store chunks with timestamp.

        Args:
            query_id: Unique identifier for the query
            query_text: The original query text
            chunks: Retrieved document chunks
            metadata: Additional metadata (ticker, filing_type, etc.)
            companies: List of company tickers
            metrics: List of financial metrics
            response_summary: Brief summary of the response
        """
        async with self._lock:
            # Enforce max entries by removing oldest
            if len(self._cache) >= self.max_entries:
                await self._evict_oldest()

            entry = CacheEntry(
                query_id=query_id,
                query_text=query_text,
                chunks=chunks,
                metadata=metadata or {},
                companies=companies or [],
                metrics=metrics or [],
                response_summary=response_summary
            )

            self._cache[query_id] = entry
            logger.info(f"Cached {len(chunks)} chunks for query {query_id}")

    async def get(self, query_id: str) -> Optional[CacheEntry]:
        """
        Retrieve cache entry if not expired.

        Args:
            query_id: The query identifier

        Returns:
            CacheEntry if found and not expired, None otherwise
        """
        async with self._lock:
            entry = self._cache.get(query_id)

            if entry is None:
                self._misses += 1
                return None

            # Check TTL
            age = (datetime.now() - entry.created_at).total_seconds()
            if age > self.ttl:
                # Expired - remove and return None
                del self._cache[query_id]
                self._misses += 1
                logger.info(f"Cache entry {query_id} expired (age: {age:.1f}s)")
                return None

            self._hits += 1
            return entry

    async def get_chunks(self, query_id: str) -> List[DocumentChunk]:
        """
        Get all chunks for a query.

        Args:
            query_id: The query identifier

        Returns:
            List of DocumentChunk objects
        """
        entry = await self.get(query_id)
        return entry.chunks if entry else []

    async def get_chunks_by_ids(
        self,
        query_id: str,
        chunk_ids: List[str]
    ) -> List[DocumentChunk]:
        """
        Get specific chunks for follow-up answering.

        Args:
            query_id: The query identifier
            chunk_ids: List of chunk IDs to retrieve

        Returns:
            List of matching DocumentChunk objects
        """
        entry = await self.get(query_id)
        if not entry:
            return []

        # If no specific chunk_ids, return all
        if not chunk_ids:
            return entry.chunks

        # Filter to requested chunks
        chunk_id_set = set(chunk_ids)
        return [c for c in entry.chunks if c.chunk_id in chunk_id_set]

    async def update_followups(
        self,
        query_id: str,
        follow_ups: List[Any]
    ) -> bool:
        """
        Update the follow-up questions for a cached query.

        Args:
            query_id: The query identifier
            follow_ups: List of FollowUpQuestion objects

        Returns:
            True if updated, False if entry not found
        """
        async with self._lock:
            if query_id not in self._cache:
                return False

            self._cache[query_id].follow_up_questions = follow_ups
            logger.info(f"Updated {len(follow_ups)} follow-ups for query {query_id}")
            return True

    async def cleanup_expired(self) -> int:
        """
        Remove expired entries.

        Call periodically to free memory.

        Returns:
            Number of entries removed
        """
        async with self._lock:
            now = datetime.now()
            expired = [
                qid for qid, entry in self._cache.items()
                if (now - entry.created_at).total_seconds() > self.ttl
            ]

            for qid in expired:
                del self._cache[qid]

            if expired:
                logger.info(f"Cleaned up {len(expired)} expired cache entries")

            return len(expired)

    async def _evict_oldest(self) -> None:
        """Evict oldest entry to make room for new ones."""
        if not self._cache:
            return

        # Find oldest entry
        oldest_id = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].created_at
        )
        del self._cache[oldest_id]
        logger.debug(f"Evicted oldest cache entry: {oldest_id}")

    async def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries removed
        """
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            logger.info(f"Cleared {count} cache entries")
            return count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache performance metrics
        """
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0

        return {
            "size": len(self._cache),
            "max_size": self.max_entries,
            "ttl_seconds": self.ttl,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": round(hit_rate, 2),
            "total_requests": total_requests
        }

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)


# Singleton instance
_chunk_cache_instance: Optional[ChunkCache] = None


def get_chunk_cache(ttl_seconds: int = 600, max_entries: int = 100) -> ChunkCache:
    """
    Get or create the singleton ChunkCache instance.

    Args:
        ttl_seconds: TTL for cache entries (only used on first call)
        max_entries: Maximum entries (only used on first call)

    Returns:
        ChunkCache singleton instance
    """
    global _chunk_cache_instance
    if _chunk_cache_instance is None:
        _chunk_cache_instance = ChunkCache(ttl_seconds, max_entries)
    return _chunk_cache_instance
