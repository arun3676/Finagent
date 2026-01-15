"""
SEC Search Tool

LangChain-compatible tool for searching SEC filings.
Used by agents to find specific information in SEC documents.

Usage:
    tool = SECSearchTool(vector_store)
    results = await tool.run("Apple revenue 2023")
"""

from typing import List, Optional, Dict, Any
from langchain_core.tools import BaseTool
from pydantic import Field

from app.retrieval.vector_store import VectorStore
from app.retrieval.hybrid_search import HybridSearcher
from app.models import RetrievedDocument


class SECSearchTool(BaseTool):
    """
    Tool for searching SEC filings.
    
    Integrates with the hybrid search system to find
    relevant passages in SEC documents.
    """
    
    name: str = "sec_search"
    description: str = """
    Search SEC filings (10-K, 10-Q, 8-K) for financial information.
    Input should be a search query describing what you're looking for.
    Optionally include ticker symbol and date range.
    
    Examples:
    - "Apple revenue growth 2023"
    - "MSFT risk factors cybersecurity"
    - "Tesla gross margin Q4 2023"
    """
    
    hybrid_searcher: Optional[HybridSearcher] = Field(default=None, exclude=True)
    top_k: int = Field(default=5)
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(
        self,
        hybrid_searcher: HybridSearcher = None,
        top_k: int = 5,
        **kwargs
    ):
        """
        Initialize SEC search tool.
        
        Args:
            hybrid_searcher: Hybrid search component
            top_k: Number of results to return
        """
        super().__init__(**kwargs)
        self.hybrid_searcher = hybrid_searcher
        self.top_k = top_k
    
    def _run(self, query: str) -> str:
        """
        Synchronous search (not recommended).
        
        Args:
            query: Search query
            
        Returns:
            Formatted search results
        """
        raise NotImplementedError("Use async _arun instead")
    
    async def _arun(self, query: str) -> str:
        """
        Async search for SEC filings.
        
        Args:
            query: Search query
            
        Returns:
            Formatted search results
        """
        try:
            # Initialize searcher if needed
            if not self.hybrid_searcher:
                from app.retrieval.hybrid_search import HybridSearcher
                from app.retrieval.vector_store import VectorStore
                from app.retrieval.bm25_index import BM25Index
                from app.retrieval.embeddings import EmbeddingService
                import os
                
                vector_store = VectorStore()
                bm25_index = BM25Index()
                
                # Try to load existing index
                index_path = os.path.join(os.getcwd(), "data", "indexes", "bm25.pkl")
                if os.path.exists(index_path):
                    bm25_index.load_index(index_path)
                
                embedding_service = EmbeddingService()
                
                self.hybrid_searcher = HybridSearcher(
                    vector_store=vector_store,
                    bm25_index=bm25_index,
                    embedding_service=embedding_service
                )
            
            # Parse query for filters
            parsed = self._parse_query(query)
            filters = {}
            
            if parsed["ticker"]:
                filters["ticker"] = parsed["ticker"]
            
            # Construct search query
            search_query = parsed["query"]
            
            # Execute search
            results = await self.hybrid_searcher.search(
                query=search_query,
                top_k=self.top_k,
                filters=filters if filters else None
            )
            
            return self._format_results(results)
            
        except Exception as e:
            return f"Error searching SEC filings: {str(e)}"
    
    def _parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse query to extract filters.
        
        Args:
            query: Raw query string
            
        Returns:
            Dictionary with query and filters
        """
        import re
        
        # Extract ticker (uppercase 1-5 letters)
        ticker_match = re.search(r'\b([A-Z]{1,5})\b', query)
        ticker = ticker_match.group(1) if ticker_match else None
        
        # Extract year
        year_match = re.search(r'\b(20\d{2})\b', query)
        year = int(year_match.group(1)) if year_match else None
        
        # Extract quarter
        quarter_match = re.search(r'\b[Qq]([1-4])\b', query)
        quarter = int(quarter_match.group(1)) if quarter_match else None
        
        return {
            "query": query,
            "ticker": ticker,
            "year": year,
            "quarter": quarter
        }
    
    def _format_results(
        self,
        results: List[RetrievedDocument]
    ) -> str:
        """
        Format search results for agent.
        
        Args:
            results: Retrieved documents
            
        Returns:
            Formatted string
        """
        if not results:
            return "No relevant SEC filings found."
        
        formatted = []
        for i, doc in enumerate(results, 1):
            meta = doc.chunk.metadata
            formatted.append(
                f"[{i}] {meta.ticker} {meta.document_type.value} "
                f"({meta.filing_date.strftime('%Y-%m-%d')})\n"
                f"Section: {doc.chunk.section or 'N/A'}\n"
                f"Score: {doc.score:.3f}\n"
                f"Content: {doc.chunk.content[:500]}...\n"
            )
        
        return "\n---\n".join(formatted)
