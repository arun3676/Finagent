"""
Stock Price Lookup Tool

LangChain-compatible tool for fetching stock prices.
Uses free APIs for real-time and historical price data.

Data sources:
- Yahoo Finance (via yfinance)
- Alpha Vantage (requires API key)

Usage:
    tool = PriceLookupTool()
    result = await tool.run("AAPL current")
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from langchain.tools import BaseTool
from pydantic import Field


class PriceLookupTool(BaseTool):
    """
    Tool for looking up stock prices.
    
    Provides current and historical stock price data
    for financial analysis.
    """
    
    name: str = "price_lookup"
    description: str = """
    Look up stock prices. Supported queries:
    
    1. Current price: "TICKER current"
       Example: "AAPL current"
       
    2. Historical price: "TICKER date YYYY-MM-DD"
       Example: "MSFT date 2023-12-31"
       
    3. Price range: "TICKER range START_DATE END_DATE"
       Example: "GOOGL range 2023-01-01 2023-12-31"
       
    4. Price change: "TICKER change PERIOD"
       Periods: 1d, 5d, 1m, 3m, 6m, 1y, ytd
       Example: "NVDA change 1y"
    
    Returns price data including open, high, low, close, volume.
    """
    
    api_key: Optional[str] = Field(default=None, exclude=True)
    
    def __init__(self, api_key: str = None, **kwargs):
        """
        Initialize price lookup tool.
        
        Args:
            api_key: Optional API key for premium data sources
        """
        super().__init__(**kwargs)
        self.api_key = api_key
    
    def _run(self, query: str) -> str:
        """
        Synchronous price lookup.
        
        Args:
            query: Price lookup query
            
        Returns:
            Price information
        """
        # TODO: Implement synchronous lookup
        raise NotImplementedError("Use async _arun instead")
    
    async def _arun(self, query: str) -> str:
        """
        Async price lookup.
        
        Args:
            query: Price lookup query
            
        Returns:
            Price information
        """
        try:
            parts = query.strip().split()
            if len(parts) < 2:
                return "Error: Query must include ticker and operation"
            
            ticker = parts[0].upper()
            operation = parts[1].lower()
            
            if operation == "current":
                return await self._get_current_price(ticker)
            elif operation == "date" and len(parts) >= 3:
                return await self._get_historical_price(ticker, parts[2])
            elif operation == "range" and len(parts) >= 4:
                return await self._get_price_range(ticker, parts[2], parts[3])
            elif operation == "change" and len(parts) >= 3:
                return await self._get_price_change(ticker, parts[2])
            else:
                return f"Error: Unknown operation '{operation}'"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def _get_current_price(self, ticker: str) -> str:
        """
        Get current stock price.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Current price information
        """
        # TODO: Implement using yfinance or API
        # 1. Fetch current quote
        # 2. Format response
        raise NotImplementedError("Current price lookup not yet implemented")
    
    async def _get_historical_price(
        self,
        ticker: str,
        date_str: str
    ) -> str:
        """
        Get historical stock price for a specific date.
        
        Args:
            ticker: Stock ticker symbol
            date_str: Date in YYYY-MM-DD format
            
        Returns:
            Historical price information
        """
        # TODO: Implement historical lookup
        # 1. Parse date
        # 2. Fetch historical data
        # 3. Format response
        raise NotImplementedError("Historical price lookup not yet implemented")
    
    async def _get_price_range(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> str:
        """
        Get stock prices for a date range.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Price range summary
        """
        # TODO: Implement range lookup
        # 1. Parse dates
        # 2. Fetch historical data
        # 3. Calculate summary stats
        # 4. Format response
        raise NotImplementedError("Price range lookup not yet implemented")
    
    async def _get_price_change(
        self,
        ticker: str,
        period: str
    ) -> str:
        """
        Get price change over a period.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period (1d, 5d, 1m, 3m, 6m, 1y, ytd)
            
        Returns:
            Price change information
        """
        # TODO: Implement price change calculation
        # 1. Determine date range from period
        # 2. Fetch start and end prices
        # 3. Calculate change
        # 4. Format response
        raise NotImplementedError("Price change lookup not yet implemented")
    
    def _parse_period(self, period: str) -> timedelta:
        """
        Parse period string to timedelta.
        
        Args:
            period: Period string (1d, 5d, 1m, 3m, 6m, 1y)
            
        Returns:
            timedelta object
        """
        period_map = {
            "1d": timedelta(days=1),
            "5d": timedelta(days=5),
            "1m": timedelta(days=30),
            "3m": timedelta(days=90),
            "6m": timedelta(days=180),
            "1y": timedelta(days=365)
        }
        return period_map.get(period.lower(), timedelta(days=30))
    
    def _format_price_data(self, data: Dict[str, Any]) -> str:
        """
        Format price data for output.
        
        Args:
            data: Price data dictionary
            
        Returns:
            Formatted string
        """
        return (
            f"Ticker: {data.get('ticker', 'N/A')}\n"
            f"Date: {data.get('date', 'N/A')}\n"
            f"Open: ${data.get('open', 0):.2f}\n"
            f"High: ${data.get('high', 0):.2f}\n"
            f"Low: ${data.get('low', 0):.2f}\n"
            f"Close: ${data.get('close', 0):.2f}\n"
            f"Volume: {data.get('volume', 0):,}"
        )
