"""
Table Chunker

Specialized chunking for financial tables and structured data.
Keeps tables intact to preserve data relationships.

Key features:
- Detects tables in HTML and text formats
- Preserves table structure
- Adds table context (title, surrounding text)
- Handles multi-page tables

Usage:
    chunker = TableChunker()
    chunks = chunker.chunk_with_tables(document_text, metadata)
"""

import re
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

from app.config import settings
from app.models import DocumentChunk, DocumentMetadata


@dataclass
class Table:
    """Represents a detected table in the document."""
    content: str
    title: Optional[str]
    context_before: str
    context_after: str
    start_pos: int
    end_pos: int
    table_type: str  # "html", "markdown", "text"
    rows: int
    cols: int


class TableChunker:
    """
    Chunker that preserves financial tables.
    
    Strategy:
    1. Detect all tables in document
    2. Extract tables with surrounding context
    3. Chunk non-table text normally
    4. Keep tables as single chunks (unless very large)
    """
    
    # Table detection patterns
    HTML_TABLE_PATTERN = r'<table[^>]*>.*?</table>'
    MARKDOWN_TABLE_PATTERN = r'(?:\|[^\n]+\|\n)+(?:\|[-:| ]+\|\n)(?:\|[^\n]+\|\n)+'
    TEXT_TABLE_PATTERN = r'(?:(?:\s{2,}|\t)[^\n]+\n){3,}'  # Aligned columns
    
    # Financial table indicators
    FINANCIAL_TABLE_KEYWORDS = [
        "revenue", "income", "expense", "assets", "liabilities",
        "cash flow", "balance sheet", "earnings", "eps", "shares",
        "quarter", "fiscal", "year ended", "three months", "nine months"
    ]
    
    def __init__(
        self,
        max_table_size: int = 5000,
        context_chars: int = 200
    ):
        """
        Initialize table chunker.
        
        Args:
            max_table_size: Max chars before splitting a table
            context_chars: Characters of context to include
        """
        self.max_table_size = max_table_size
        self.context_chars = context_chars
    
    def chunk_with_tables(
        self,
        text: str,
        metadata: DocumentMetadata
    ) -> List[DocumentChunk]:
        """
        Chunk document while preserving tables.
        
        Args:
            text: Document text (may contain HTML tables)
            metadata: Document metadata
            
        Returns:
            List of DocumentChunk objects
        """
        # TODO: Implement table-aware chunking
        # 1. Detect all tables
        # 2. Extract tables with context
        # 3. Chunk remaining text
        # 4. Merge chunks in document order
        raise NotImplementedError("Table-aware chunking not yet implemented")
    
    def detect_tables(self, text: str) -> List[Table]:
        """
        Detect all tables in the document.
        
        Args:
            text: Document text
            
        Returns:
            List of detected Table objects
        """
        # TODO: Implement table detection
        # 1. Find HTML tables
        # 2. Find markdown tables
        # 3. Find text-aligned tables
        # 4. Deduplicate overlapping detections
        raise NotImplementedError("Table detection not yet implemented")
    
    def extract_table_with_context(
        self,
        text: str,
        table: Table
    ) -> str:
        """
        Extract table with surrounding context.
        
        Args:
            text: Full document text
            table: Detected table
            
        Returns:
            Table content with context
        """
        # TODO: Implement context extraction
        # 1. Get text before table (for title)
        # 2. Get text after table (for notes)
        # 3. Combine into single chunk
        raise NotImplementedError("Table context extraction not yet implemented")
    
    def parse_html_table(self, html: str) -> Dict[str, Any]:
        """
        Parse HTML table into structured format.
        
        Args:
            html: HTML table string
            
        Returns:
            Dictionary with headers, rows, and metadata
        """
        # TODO: Implement HTML table parsing
        # 1. Parse HTML
        # 2. Extract headers
        # 3. Extract rows
        # 4. Handle colspan/rowspan
        raise NotImplementedError("HTML table parsing not yet implemented")
    
    def table_to_markdown(self, table_data: Dict[str, Any]) -> str:
        """
        Convert parsed table to markdown format.
        
        Args:
            table_data: Parsed table dictionary
            
        Returns:
            Markdown table string
        """
        # TODO: Implement markdown conversion
        # 1. Format headers
        # 2. Add separator row
        # 3. Format data rows
        # 4. Align columns
        raise NotImplementedError("Markdown conversion not yet implemented")
    
    def split_large_table(
        self,
        table: Table,
        metadata: DocumentMetadata
    ) -> List[DocumentChunk]:
        """
        Split a table that exceeds max size.
        
        Args:
            table: Large table to split
            metadata: Document metadata
            
        Returns:
            List of chunks from the table
        """
        # TODO: Implement table splitting
        # 1. Parse table structure
        # 2. Split by rows (keep headers)
        # 3. Create chunks with table context
        raise NotImplementedError("Table splitting not yet implemented")
    
    def _detect_html_tables(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Find HTML tables in text.
        
        Args:
            text: Document text
            
        Returns:
            List of (start, end, content) tuples
        """
        tables = []
        for match in re.finditer(self.HTML_TABLE_PATTERN, text, re.DOTALL | re.IGNORECASE):
            tables.append((match.start(), match.end(), match.group()))
        return tables
    
    def _is_financial_table(self, table_text: str) -> bool:
        """
        Check if a table contains financial data.
        
        Args:
            table_text: Table content
            
        Returns:
            True if table appears to be financial
        """
        text_lower = table_text.lower()
        matches = sum(1 for kw in self.FINANCIAL_TABLE_KEYWORDS if kw in text_lower)
        return matches >= 2
    
    def _find_table_title(self, text: str, table_start: int) -> Optional[str]:
        """
        Find the title/header for a table.
        
        Args:
            text: Full document text
            table_start: Starting position of table
            
        Returns:
            Table title if found
        """
        # Look at text before table for title
        context_start = max(0, table_start - 500)
        context = text[context_start:table_start]
        
        # TODO: Implement title extraction
        # 1. Look for bold/header text
        # 2. Look for "Table X:" pattern
        # 3. Look for capitalized line
        return None
