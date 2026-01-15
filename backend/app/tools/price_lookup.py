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

from typing import Optional, Dict, Any, ClassVar
from datetime import datetime, timedelta
from langchain_core.tools import BaseTool
from pydantic import Field


class PriceLookupTool(BaseTool):
    """
    Tool for looking up stock prices.

    Provides current and historical stock price data
    for financial analysis.
    """

    name: str = "price_lookup"
    INVALID_TICKERS: ClassVar[set] = {"PRICE", "TICKER", "SYMBOL"}
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
            
            ticker = parts[0].upper().lstrip("$")
            if not ticker or ticker in self.INVALID_TICKERS:
                return "Error: Invalid ticker placeholder"
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
        import yfinance as yf
        import asyncio
        
        try:
            # yfinance is synchronous, run in executor
            loop = asyncio.get_event_loop()
            
            def fetch():
                t = yf.Ticker(ticker)
                # fast_info is faster than info
                info = t.fast_info
                # fallback to info if fast_info missing key data
                if not info or not hasattr(info, 'last_price'):
                    return t.info
                return info

            data = await loop.run_in_executor(None, fetch)
            
            if not data:
                return f"Error: No data found for {ticker}"
            
            # Handle different data structures between info and fast_info
            if hasattr(data, 'last_price'): # fast_info
                price = data.last_price
                open_p = data.open
                previous_close = data.previous_close
                # fast_info doesn't always have high/low/volume easily accessible in same way
                # We can try to get today's history for that
                try:
                    hist = yf.Ticker(ticker).history(period="1d")
                    if not hist.empty:
                        high = hist['High'].iloc[0]
                        low = hist['Low'].iloc[0]
                        vol = hist['Volume'].iloc[0]
                    else:
                        high = low = vol = 0
                except:
                    high = low = vol = 0
            else: # info dict
                price = data.get('currentPrice') or data.get('regularMarketPrice')
                open_p = data.get('open')
                high = data.get('dayHigh')
                low = data.get('dayLow')
                vol = data.get('volume')
                previous_close = data.get('previousClose')

            response = (
                f"Ticker: {ticker}\n"
                f"Current Price: ${price:.2f}\n"
                f"Open: ${open_p:.2f}\n"
                f"High: ${high:.2f}\n"
                f"Low: ${low:.2f}\n"
                f"Volume: {vol:,}\n"
                f"Previous Close: ${previous_close:.2f}"
            )
            return response
            
        except Exception as e:
            return f"Error fetching price for {ticker}: {str(e)}"
    
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
        import yfinance as yf
        import asyncio
        
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d")
            start_date = target_date
            end_date = target_date + timedelta(days=1)
            
            loop = asyncio.get_event_loop()
            
            def fetch():
                t = yf.Ticker(ticker)
                return t.history(start=start_date, end=end_date)
                
            hist = await loop.run_in_executor(None, fetch)
            
            if hist.empty:
                return f"No data found for {ticker} on {date_str} (market might be closed)"
            
            row = hist.iloc[0]
            return self._format_price_data({
                "ticker": ticker,
                "date": date_str,
                "open": row['Open'],
                "high": row['High'],
                "low": row['Low'],
                "close": row['Close'],
                "volume": int(row['Volume'])
            })
            
        except Exception as e:
            return f"Error fetching historical price: {str(e)}"
    
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
        import yfinance as yf
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
            
            def fetch():
                t = yf.Ticker(ticker)
                return t.history(start=start_date, end=end_date)
                
            hist = await loop.run_in_executor(None, fetch)
            
            if hist.empty:
                return f"No data found for {ticker} between {start_date} and {end_date}"
            
            # Calculate summary stats
            avg_close = hist['Close'].mean()
            max_high = hist['High'].max()
            min_low = hist['Low'].min()
            total_vol = hist['Volume'].sum()
            
            first_row = hist.iloc[0]
            last_row = hist.iloc[-1]
            change = last_row['Close'] - first_row['Open']
            change_pct = (change / first_row['Open']) * 100
            
            return (
                f"Ticker: {ticker} ({start_date} to {end_date})\n"
                f"Days Traded: {len(hist)}\n"
                f"Average Close: ${avg_close:.2f}\n"
                f"Highest Price: ${max_high:.2f}\n"
                f"Lowest Price: ${min_low:.2f}\n"
                f"Total Volume: {total_vol:,}\n"
                f"Price Change: ${change:.2f} ({change_pct:.2f}%)"
            )
            
        except Exception as e:
            return f"Error fetching price range: {str(e)}"
    
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
        import yfinance as yf
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
            
            def fetch():
                t = yf.Ticker(ticker)
                return t.history(period=period)
                
            hist = await loop.run_in_executor(None, fetch)
            
            if hist.empty:
                return f"No data found for {ticker} over {period}"
            
            first_row = hist.iloc[0]
            last_row = hist.iloc[-1]
            
            current_price = last_row['Close']
            start_price = first_row['Open'] # Or Close? Usually Open of start period or Close of prev.
            
            change = current_price - start_price
            change_pct = (change / start_price) * 100
            
            return (
                f"Ticker: {ticker} (Period: {period})\n"
                f"Current Price: ${current_price:.2f}\n"
                f"Start Price: ${start_price:.2f}\n"
                f"Change: ${change:.2f}\n"
                f"Percent Change: {change_pct:.2f}%"
            )
            
        except Exception as e:
            return f"Error fetching price change: {str(e)}"
    
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
