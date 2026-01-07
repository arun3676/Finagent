"""
Citations Module

Handles citation extraction, linking, and formatting:
- Extract claims from generated responses
- Link claims to source document chunks
- Format citations for various output formats
"""

from app.citations.extractor import ClaimExtractor
from app.citations.linker import CitationLinker
from app.citations.formatter import CitationFormatter

__all__ = ["ClaimExtractor", "CitationLinker", "CitationFormatter"]
