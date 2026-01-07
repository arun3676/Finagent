"""
Ingestion Module

Handles loading and preprocessing of financial documents:
- SEC EDGAR filings (10-K, 10-Q, 8-K)
- Earnings call transcripts
- XBRL structured data extraction
"""

from app.ingestion.sec_edgar_loader import SECEdgarLoader
from app.ingestion.earnings_loader import EarningsCallLoader
from app.ingestion.xbrl_parser import XBRLParser

__all__ = ["SECEdgarLoader", "EarningsCallLoader", "XBRLParser"]
