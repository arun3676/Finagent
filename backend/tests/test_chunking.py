"""
Tests for Chunking Module

Tests SEC chunker, earnings chunker, and table chunker.
"""

import pytest
from datetime import datetime

from app.chunking.sec_chunker import SECChunker, Section
from app.chunking.earnings_chunker import EarningsChunker, QAPair
from app.chunking.table_chunker import TableChunker
from app.models import DocumentMetadata, DocumentType


class TestSECChunker:
    """Tests for SEC filing chunker."""
    
    @pytest.fixture
    def chunker(self):
        """Create chunker instance."""
        return SECChunker(chunk_size=1000, chunk_overlap=200)
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample document metadata."""
        return DocumentMetadata(
            ticker="AAPL",
            company_name="Apple Inc.",
            document_type=DocumentType.SEC_10K,
            filing_date=datetime(2023, 10, 27),
            source_url="https://sec.gov/example"
        )
    
    def test_initialization(self, chunker):
        """Test chunker initializes with correct parameters."""
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200
        assert chunker.preserve_tables is True
    
    def test_split_by_paragraphs(self, chunker):
        """Test paragraph splitting."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        paragraphs = chunker._split_by_paragraphs(text)
        
        assert len(paragraphs) == 3
        assert paragraphs[0] == "First paragraph."
        assert paragraphs[1] == "Second paragraph."
    
    def test_split_by_paragraphs_empty(self, chunker):
        """Test paragraph splitting with empty input."""
        paragraphs = chunker._split_by_paragraphs("")
        assert len(paragraphs) == 0
    
    def test_section_patterns_exist(self, chunker):
        """Test that section patterns are defined."""
        assert "item_1" in chunker.SECTION_10K_PATTERNS
        assert "item_1a" in chunker.SECTION_10K_PATTERNS
        assert "item_7" in chunker.SECTION_10K_PATTERNS
    
    def test_create_chunk(self, chunker, sample_metadata):
        """Test chunk creation."""
        chunk = chunker._create_chunk(
            content="Test content",
            metadata=sample_metadata,
            section="Item 1",
            chunk_index=0,
            document_id="doc_123"
        )
        
        assert chunk.content == "Test content"
        assert chunk.section == "Item 1"
        assert chunk.chunk_index == 0
        assert chunk.metadata.ticker == "AAPL"


class TestEarningsChunker:
    """Tests for earnings call chunker."""
    
    @pytest.fixture
    def chunker(self):
        """Create chunker instance."""
        return EarningsChunker()
    
    def test_initialization(self, chunker):
        """Test chunker initializes correctly."""
        assert chunker.keep_qa_pairs is True
        assert len(chunker.QA_START_PATTERNS) > 0
    
    def test_format_qa_chunk(self, chunker):
        """Test Q&A pair formatting."""
        qa_pair = QAPair(
            question="What is your revenue guidance?",
            questioner="John Analyst",
            questioner_affiliation="Goldman Sachs",
            answer="We expect revenue of $100 billion.",
            answerer="Tim Cook",
            answerer_title="CEO"
        )
        
        formatted = chunker._format_qa_chunk(qa_pair)
        
        assert "QUESTION from John Analyst" in formatted
        assert "Goldman Sachs" in formatted
        assert "ANSWER from Tim Cook" in formatted
        assert "CEO" in formatted


class TestTableChunker:
    """Tests for table-aware chunker."""
    
    @pytest.fixture
    def chunker(self):
        """Create chunker instance."""
        return TableChunker(max_table_size=5000, context_chars=200)
    
    def test_initialization(self, chunker):
        """Test chunker initializes correctly."""
        assert chunker.max_table_size == 5000
        assert chunker.context_chars == 200
    
    def test_is_financial_table(self, chunker):
        """Test financial table detection."""
        financial_table = "Revenue | 2023 | 2022\n$100M | $90M"
        assert chunker._is_financial_table(financial_table) is True
        
        non_financial = "Name | Age | City"
        assert chunker._is_financial_table(non_financial) is False
    
    def test_detect_html_tables(self, chunker):
        """Test HTML table detection."""
        html = "<p>Text</p><table><tr><td>Data</td></tr></table><p>More</p>"
        tables = chunker._detect_html_tables(html)
        
        assert len(tables) == 1
        assert "<table>" in tables[0][2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
