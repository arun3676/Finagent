"""
Financial Calculator Tool

LangChain-compatible tool for financial calculations.
Handles common financial metrics and ratios.

Supported calculations:
- Growth rates (YoY, QoQ, CAGR)
- Margins (gross, operating, net)
- Ratios (P/E, P/S, debt-to-equity)
- Comparisons and percentages

Usage:
    tool = FinancialCalculator()
    result = tool.run("growth_rate current=150 previous=100")
"""

from typing import Dict, Any, Optional, List
from decimal import Decimal, InvalidOperation
from langchain_core.tools import BaseTool
from pydantic import Field
import re


class FinancialCalculator(BaseTool):
    """
    Tool for financial calculations.
    
    Provides accurate financial metric calculations
    with proper handling of edge cases.
    """
    
    name: str = "financial_calculator"
    description: str = """
    Perform financial calculations. Supported operations:
    
    1. growth_rate: Calculate percentage growth
       Input: "growth_rate current=VALUE previous=VALUE"
       
    2. margin: Calculate margin/ratio as percentage
       Input: "margin numerator=VALUE denominator=VALUE"
       
    3. cagr: Compound Annual Growth Rate
       Input: "cagr start=VALUE end=VALUE years=N"
       
    4. percentage_change: Simple percentage change
       Input: "percentage_change from=VALUE to=VALUE"
       
    5. ratio: Calculate ratio between two values
       Input: "ratio a=VALUE b=VALUE"
    
    Values can include units like M (million), B (billion), K (thousand).
    Example: "growth_rate current=150M previous=120M"
    """
    
    def _run(self, query: str) -> str:
        """
        Execute financial calculation.
        
        Args:
            query: Calculation query
            
        Returns:
            Calculation result
        """
        try:
            # Parse the operation and parameters
            parts = query.strip().split()
            if not parts:
                return "Error: Empty query"
            
            operation = parts[0].lower()
            params = self._parse_params(" ".join(parts[1:]))
            
            # Route to appropriate calculation
            if operation == "growth_rate":
                return self._calc_growth_rate(params)
            elif operation == "margin":
                return self._calc_margin(params)
            elif operation == "cagr":
                return self._calc_cagr(params)
            elif operation == "percentage_change":
                return self._calc_percentage_change(params)
            elif operation == "ratio":
                return self._calc_ratio(params)
            else:
                return f"Error: Unknown operation '{operation}'"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """Async version (just calls sync)."""
        return self._run(query)
    
    def _parse_params(self, param_str: str) -> Dict[str, Decimal]:
        """
        Parse parameter string into values.
        
        Args:
            param_str: String like "current=150M previous=100M"
            
        Returns:
            Dictionary of parameter names to Decimal values
        """
        params = {}
        
        # Find all key=value pairs
        pattern = r'(\w+)=([\d.,]+)([KMBkmb])?'
        matches = re.findall(pattern, param_str)
        
        for key, value, unit in matches:
            # Parse numeric value
            numeric = Decimal(value.replace(',', ''))
            
            # Apply unit multiplier
            multipliers = {
                'K': Decimal('1000'),
                'k': Decimal('1000'),
                'M': Decimal('1000000'),
                'm': Decimal('1000000'),
                'B': Decimal('1000000000'),
                'b': Decimal('1000000000')
            }
            
            if unit:
                numeric *= multipliers.get(unit, Decimal('1'))
            
            params[key.lower()] = numeric
        
        return params
    
    def _calc_growth_rate(self, params: Dict[str, Decimal]) -> str:
        """Calculate growth rate."""
        current = params.get('current')
        previous = params.get('previous')
        
        if current is None or previous is None:
            return "Error: growth_rate requires 'current' and 'previous' parameters"
        
        if previous == 0:
            return "Error: Cannot calculate growth rate with previous value of 0"
        
        growth = ((current - previous) / previous) * 100
        return f"Growth Rate: {growth:.2f}%"
    
    def _calc_margin(self, params: Dict[str, Decimal]) -> str:
        """Calculate margin/ratio as percentage."""
        numerator = params.get('numerator')
        denominator = params.get('denominator')
        
        if numerator is None or denominator is None:
            return "Error: margin requires 'numerator' and 'denominator' parameters"
        
        if denominator == 0:
            return "Error: Cannot calculate margin with denominator of 0"
        
        margin = (numerator / denominator) * 100
        return f"Margin: {margin:.2f}%"
    
    def _calc_cagr(self, params: Dict[str, Decimal]) -> str:
        """Calculate Compound Annual Growth Rate."""
        start = params.get('start')
        end = params.get('end')
        years = params.get('years')
        
        if start is None or end is None or years is None:
            return "Error: cagr requires 'start', 'end', and 'years' parameters"
        
        if start <= 0 or years <= 0:
            return "Error: start value and years must be positive"
        
        # CAGR = (end/start)^(1/years) - 1
        cagr = (float(end / start) ** (1 / float(years)) - 1) * 100
        return f"CAGR: {cagr:.2f}%"
    
    def _calc_percentage_change(self, params: Dict[str, Decimal]) -> str:
        """Calculate simple percentage change."""
        from_val = params.get('from')
        to_val = params.get('to')
        
        if from_val is None or to_val is None:
            return "Error: percentage_change requires 'from' and 'to' parameters"
        
        if from_val == 0:
            return "Error: Cannot calculate percentage change from 0"
        
        change = ((to_val - from_val) / from_val) * 100
        direction = "increase" if change > 0 else "decrease"
        return f"Percentage Change: {abs(change):.2f}% {direction}"
    
    def _calc_ratio(self, params: Dict[str, Decimal]) -> str:
        """Calculate ratio between two values."""
        a = params.get('a')
        b = params.get('b')
        
        if a is None or b is None:
            return "Error: ratio requires 'a' and 'b' parameters"
        
        if b == 0:
            return "Error: Cannot calculate ratio with b=0"
        
        ratio = a / b
        return f"Ratio: {ratio:.2f}x"
    
    def get_supported_operations(self) -> List[str]:
        """Get list of supported operations."""
        return [
            "growth_rate",
            "margin",
            "cagr",
            "percentage_change",
            "ratio"
        ]
