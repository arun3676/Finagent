"""
Follow-up Questions Module

This module provides functionality for generating and executing
contextual follow-up questions after a query response.

Components:
- FollowUpGenerator: Generates contextual follow-up questions
- ChunkCache: Caches retrieved chunks for fast follow-up execution
- FollowUpExecutor: Executes follow-up queries using cached context
"""

from app.followup.generator import FollowUpGenerator, FollowUpQuestion
from app.followup.cache import ChunkCache, CacheEntry
from app.followup.executor import FollowUpExecutor, FollowUpResponse

__all__ = [
    "FollowUpGenerator",
    "FollowUpQuestion",
    "ChunkCache",
    "CacheEntry",
    "FollowUpExecutor",
    "FollowUpResponse",
]
