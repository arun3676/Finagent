"""
Cache Module

Provides semantic caching for query-response pairs to avoid redundant LLM calls.
"""

from app.cache.query_cache import SemanticQueryCache, CachedResponse, QueryCacheEntry

__all__ = ["SemanticQueryCache", "CachedResponse", "QueryCacheEntry"]
