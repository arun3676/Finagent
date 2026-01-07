"""
Tests for Ingestion Module

Tests SEC EDGAR loader, earnings call loader, and XBRL parser.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.ingestion.sec_edgar_loader import SECEdgarLoader
from app.ingestion.earnings_loader import EarningsCallLoader
from app.ingestion.xbrl_parser import XBRLParser
from app.models import DocumentType


class TestSECEdgarLoader:
    """Tests for SEC EDGAR document loader."""
    
    @pytest.fixture
    def loader(self):
        """Create loader instance."""
        return SECEdgarLoader()
    
    def test_initialization(self, loader):
        """Test loader initializes correctly."""
        assert loader.BASE_URL == "https://data.sec.gov"
        assert loader.RATE_LIMIT_DELAY == 0.1
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, loader):
        """Test rate limiting is enforced."""
        # TODO: Implement when rate limiting is implemented
        pass
    
    def test_create_document_metadata(self, loader):
        """Test metadata creation."""
        metadata = loader._create_document_metadata(
            ticker="AAPL",
            company_name="Apple Inc.",
            filing_type="10-K",
            filing_date=datetime(2023, 10, 27),
            accession_number="0000320193-23-000077"
        )
        
        assert metadata.ticker == "AAPL"
        assert metadata.company_name == "Apple Inc."
        assert metadata.document_type == DocumentType.SEC_10K
    
    @pytest.mark.asyncio
    async def test_get_company_cik_not_implemented(self, loader):
        """Test CIK lookup raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await loader.get_company_cik("AAPL")
    
    @pytest.mark.asyncio
    async def test_get_filings_not_implemented(self, loader):
        """Test filing retrieval raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await loader.get_filings("AAPL", "10-K", limit=5)


class TestEarningsCallLoader:
    """Tests for earnings call transcript loader."""
    
    @pytest.fixture
    def loader(self):
        """Create loader instance."""
        return EarningsCallLoader()
    
    def test_initialization(self, loader):
        """Test loader initializes correctly."""
        assert len(loader.EXECUTIVE_PATTERNS) > 0
        assert len(loader.ANALYST_PATTERNS) > 0
    
    def test_classify_speaker_role_executive(self, loader):
        """Test executive role classification."""
        role = loader._classify_speaker_role("Tim Cook", "Chief Executive Officer")
        assert role == "executive"
    
    def test_classify_speaker_role_analyst(self, loader):
        """Test analyst role classification."""
        role = loader._classify_speaker_role("John Smith", "Morgan Stanley Research")
        assert role == "analyst"
    
    def test_classify_speaker_role_operator(self, loader):
        """Test operator role classification."""
        role = loader._classify_speaker_role("Conference Operator", "Operator")
        assert role == "operator"
    
    @pytest.mark.asyncio
    async def test_load_transcript_not_implemented(self, loader):
        """Test transcript loading raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await loader.load_transcript("AAPL", "2024", "Q1")


class TestXBRLParser:
    """Tests for XBRL parser."""
    
    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return XBRLParser()
    
    def test_initialization(self, parser):
        """Test parser initializes correctly."""
        assert "us-gaap" in parser.NAMESPACES
        assert len(parser.INCOME_STATEMENT_CONCEPTS) > 0
    
    def test_normalize_concept_name(self, parser):
        """Test concept name normalization."""
        normalized = parser._normalize_concept_name("us-gaap:Revenues")
        assert normalized == "Revenues"
    
    def test_normalize_concept_name_no_prefix(self, parser):
        """Test normalization without namespace prefix."""
        normalized = parser._normalize_concept_name("NetIncomeLoss")
        assert normalized == "NetIncomeLoss"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
