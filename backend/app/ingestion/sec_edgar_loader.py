"""
SEC EDGAR Document Loader

Fetches SEC filings (10-K, 10-Q, 8-K) from the SEC EDGAR database.
Handles rate limiting, caching, and document parsing.

SEC EDGAR API Documentation: https://www.sec.gov/developer

Usage:
    loader = SECEdgarLoader()
    filings = await loader.get_filings("AAPL", filing_type="10-K", limit=5)
"""

import asyncio
import httpx
import re
import time
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup

from app.config import settings
from app.models import DocumentMetadata, DocumentType

logger = logging.getLogger(__name__)


class SECEdgarLoader:
    """
    Loader for SEC EDGAR filings.
    
    Supports:
    - 10-K (Annual Reports)
    - 10-Q (Quarterly Reports)
    - 8-K (Current Reports / Material Events)
    
    Implements SEC rate limiting (10 requests/second) and caching.
    """
    
    BASE_URL = "https://data.sec.gov"
    SUBMISSIONS_URL = f"{BASE_URL}/submissions"
    ARCHIVES_URL = f"{BASE_URL}/Archives/edgar/data"
    
    # SEC requires User-Agent header with contact info
    HEADERS = {
        "User-Agent": "FinAgent/1.0 (arun.kumar@finagent.ai)",
        "Accept-Encoding": "gzip, deflate"
    }
    
    # Rate limiting: SEC allows 10 requests/second
    RATE_LIMIT_DELAY = 0.1  # 100ms between requests
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize SEC EDGAR loader.
        
        Args:
            cache_dir: Directory to cache downloaded filings
        """
        self.cache_dir = cache_dir or Path("./data/cache/sec")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request_time = 0.0
        
    async def _rate_limit(self) -> None:
        """Enforce SEC rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self.RATE_LIMIT_DELAY:
            sleep_time = self.RATE_LIMIT_DELAY - time_since_last
            await asyncio.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    async def get_company_cik(self, ticker: str) -> Optional[str]:
        """
        Get CIK (Central Index Key) for a company ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            
        Returns:
            10-digit CIK string or None if not found
        """
        return await self._lookup_cik_from_tickers(ticker)
    
    async def _lookup_cik_from_tickers(self, ticker: str) -> Optional[str]:
        """CIK lookup from company tickers JSON."""
        await self._rate_limit()
        
        ticker = ticker.upper()
        
        cik_map = {
            "AAPL": "0000320193",
            "MSFT": "0000789019",
            "GOOGL": "0001652044",
            "AMZN": "0001018724",
            "TSLA": "0001318605",
            "META": "0001326801",
            "NVDA": "0001045810"
        }
        
        if ticker in cik_map:
            cik = cik_map[ticker]
            logger.info(f"Found CIK {cik} for ticker {ticker}")
            return cik
        
        url = f"{self.BASE_URL}/files/company_tickers.json"
        
        try:
            async with httpx.AsyncClient(headers=self.HEADERS, timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                
                for entry in data.values():
                    if isinstance(entry, dict) and entry.get("ticker", "").upper() == ticker:
                        cik = str(entry.get("cik_str", "")).zfill(10)
                        logger.info(f"Found CIK {cik} for ticker {ticker} via tickers endpoint")
                        return cik
                
                logger.warning(f"Ticker {ticker} not found in company tickers")
                return None
                
        except Exception as e:
            logger.error(f"Error in CIK lookup for {ticker}: {e}")
            return None
    
    async def get_filings(
        self,
        ticker: str,
        filing_type: str = "10-K",
        limit: int = 5,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch SEC filings for a company.
        
        Args:
            ticker: Stock ticker symbol
            filing_type: Type of filing (10-K, 10-Q, 8-K)
            limit: Maximum number of filings to return
            start_date: Filter filings after this date
            end_date: Filter filings before this date
            
        Returns:
            List of filing metadata dictionaries
        """
        cik = await self.get_company_cik(ticker)
        if not cik:
            logger.error(f"Could not find CIK for ticker {ticker}")
            return []
        
        await self._rate_limit()
        
        url = f"{self.SUBMISSIONS_URL}/CIK{cik}.json"
        
        try:
            async with httpx.AsyncClient(headers=self.HEADERS, timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                
                company_name = data.get("name", "")
                filings_data = data.get("filings", {}).get("recent", {})
                
                forms = filings_data.get("form", [])
                filing_dates = filings_data.get("filingDate", [])
                accession_numbers = filings_data.get("accessionNumber", [])
                primary_documents = filings_data.get("primaryDocument", [])
                
                filings = []
                for i, form in enumerate(forms):
                    if form != filing_type:
                        continue
                    
                    filing_date = datetime.strptime(filing_dates[i], "%Y-%m-%d")
                    
                    if start_date and filing_date < start_date:
                        continue
                    if end_date and filing_date > end_date:
                        continue
                    
                    accession = accession_numbers[i].replace("-", "")
                    primary_doc = primary_documents[i]
                    
                    filings.append({
                        "ticker": ticker,
                        "company_name": company_name,
                        "filing_type": form,
                        "filing_date": filing_date,
                        "accession_number": accession_numbers[i],
                        "cik": cik,
                        "primary_document": primary_doc,
                        "url": f"{self.ARCHIVES_URL}/{cik}/{accession}/{primary_doc}"
                    })
                    
                    if len(filings) >= limit:
                        break
                
                logger.info(f"Found {len(filings)} {filing_type} filings for {ticker}")
                return filings
                
        except Exception as e:
            logger.error(f"Error fetching filings for {ticker}: {e}")
            return []
    
    async def download_filing(
        self,
        accession_number: str,
        cik: str,
        primary_document: Optional[str] = None
    ) -> str:
        """
        Download the full text of a filing.
        
        Args:
            accession_number: SEC accession number (with dashes)
            cik: Company CIK
            primary_document: Primary document filename (optional)
            
        Returns:
            Raw filing text content
        """
        cache_file = self.cache_dir / f"{cik}_{accession_number.replace('-', '')}.txt"
        
        if cache_file.exists():
            logger.info(f"Loading cached filing: {cache_file}")
            return cache_file.read_text(encoding="utf-8")
        
        await self._rate_limit()
        
        accession_clean = accession_number.replace("-", "")
        
        if primary_document:
            url = f"{self.ARCHIVES_URL}/{cik}/{accession_clean}/{primary_document}"
        else:
            url = f"{self.ARCHIVES_URL}/{cik}/{accession_clean}/{accession_number}-index.html"
        
        try:
            async with httpx.AsyncClient(headers=self.HEADERS, timeout=60.0, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                content = response.text
                
                soup = BeautifulSoup(content, "lxml")
                text = soup.get_text(separator="\n", strip=True)
                
                cache_file.write_text(text, encoding="utf-8")
                logger.info(f"Downloaded and cached filing: {cache_file}")
                
                return text
                
        except Exception as e:
            logger.error(f"Error downloading filing {accession_number}: {e}")
            raise
    
    async def parse_filing_sections(
        self,
        filing_text: str,
        filing_type: str
    ) -> Dict[str, str]:
        """
        Parse a filing into its constituent sections.
        
        For 10-K filings, extracts:
        - Item 1: Business
        - Item 1A: Risk Factors
        - Item 7: MD&A
        - Item 7A: Quantitative Disclosures
        - Item 8: Financial Statements
        
        Args:
            filing_text: Raw filing text
            filing_type: Type of filing for section mapping
            
        Returns:
            Dictionary mapping section names to content
        """
        if filing_type == "10-K":
            return self._parse_10k_sections(filing_text)
        elif filing_type == "10-Q":
            return self._parse_10q_sections(filing_text)
        else:
            logger.warning(f"Section parsing not implemented for {filing_type}")
            return {"full_text": filing_text}
    
    def _parse_10k_sections(self, text: str) -> Dict[str, str]:
        """Parse 10-K filing into sections."""
        sections = {}
        
        section_patterns = [
            ("item_1", r"(?i)item\s*1[.\s]+business", r"(?i)item\s*1a"),
            ("item_1a", r"(?i)item\s*1a[.\s]+risk\s*factors", r"(?i)item\s*1b"),
            ("item_1b", r"(?i)item\s*1b[.\s]+unresolved\s*staff", r"(?i)item\s*2"),
            ("item_2", r"(?i)item\s*2[.\s]+properties", r"(?i)item\s*3"),
            ("item_3", r"(?i)item\s*3[.\s]+legal\s*proceedings", r"(?i)item\s*4"),
            ("item_7", r"(?i)item\s*7[.\s]+management.*discussion", r"(?i)item\s*7a"),
            ("item_7a", r"(?i)item\s*7a[.\s]+quantitative", r"(?i)item\s*8"),
            ("item_8", r"(?i)item\s*8[.\s]+financial\s*statements", r"(?i)item\s*9"),
        ]
        
        for section_name, start_pattern, end_pattern in section_patterns:
            start_match = re.search(start_pattern, text)
            if not start_match:
                continue
            
            start_pos = start_match.start()
            
            end_match = re.search(end_pattern, text[start_pos:])
            if end_match:
                end_pos = start_pos + end_match.start()
            else:
                end_pos = len(text)
            
            section_content = text[start_pos:end_pos]
            section_content = self._clean_section_text(section_content)
            
            if section_content:
                sections[section_name] = section_content
                logger.debug(f"Extracted {section_name}: {len(section_content)} chars")
        
        return sections
    
    def _parse_10q_sections(self, text: str) -> Dict[str, str]:
        """Parse 10-Q filing into sections."""
        sections = {}
        
        section_patterns = [
            ("part_i_item_1", r"(?i)part\s*i.*item\s*1[.\s]+financial\s*statements", r"(?i)item\s*2"),
            ("part_i_item_2", r"(?i)item\s*2[.\s]+management.*discussion", r"(?i)item\s*3"),
            ("part_i_item_3", r"(?i)item\s*3[.\s]+quantitative", r"(?i)item\s*4"),
        ]
        
        for section_name, start_pattern, end_pattern in section_patterns:
            start_match = re.search(start_pattern, text)
            if not start_match:
                continue
            
            start_pos = start_match.start()
            end_match = re.search(end_pattern, text[start_pos:])
            
            if end_match:
                end_pos = start_pos + end_match.start()
            else:
                end_pos = len(text)
            
            section_content = text[start_pos:end_pos]
            section_content = self._clean_section_text(section_content)
            
            if section_content:
                sections[section_name] = section_content
        
        return sections
    
    def _clean_section_text(self, text: str) -> str:
        """Clean and normalize section text."""
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        text = text.strip()
        return text
    
    def _create_document_metadata(
        self,
        ticker: str,
        company_name: str,
        filing_type: str,
        filing_date: datetime,
        accession_number: str
    ) -> DocumentMetadata:
        """
        Create DocumentMetadata from filing information.
        
        Args:
            ticker: Stock ticker
            company_name: Company name
            filing_type: SEC filing type
            filing_date: Date of filing
            accession_number: SEC accession number
            
        Returns:
            DocumentMetadata instance
        """
        doc_type_map = {
            "10-K": DocumentType.SEC_10K,
            "10-Q": DocumentType.SEC_10Q,
            "8-K": DocumentType.SEC_8K
        }
        
        return DocumentMetadata(
            ticker=ticker,
            company_name=company_name,
            document_type=doc_type_map.get(filing_type, DocumentType.SEC_10K),
            filing_date=filing_date,
            source_url=f"{self.ARCHIVES_URL}/{accession_number}",
            accession_number=accession_number
        )
