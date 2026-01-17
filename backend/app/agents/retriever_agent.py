"""
Retriever Agent

Orchestrates document retrieval using hybrid search.
Handles query reformulation and iterative retrieval.

Retrieval strategy:
1. Extract ticker from query
2. Check if data exists in Qdrant
3. If not, auto-ingest from SEC EDGAR
4. Execute hybrid search (dense + sparse)
5. Rerank results
6. Validate relevance

Usage:
    agent = RetrieverAgent()
    docs = await agent.retrieve(query, filters)
"""

import re
import logging
from typing import List, Optional, Dict, Any, Tuple
from openai import AsyncOpenAI
from datetime import datetime

from app.config import settings
from app.models import AgentState, RetrievedDocument, SubQuery, DocumentChunk, DocumentMetadata, DocumentType, AgentRole, StepEvent
from app.retrieval.hybrid_search import HybridSearcher
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_index import BM25Index
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.reranker import Reranker
from app.tools.price_lookup import PriceLookupTool
from app.agents.prompts import RETRIEVER_SYSTEM_PROMPT, RETRIEVER_USER_TEMPLATE
from app.utils.temporal import (
    derive_fiscal_metadata,
    extract_temporal_constraints,
    merge_temporal_filters,
)

logger = logging.getLogger(__name__)


# Common ticker mappings for company name resolution
TICKER_MAPPINGS = {
    "microsoft": "MSFT",
    "apple": "AAPL",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "netflix": "NFLX",
    "intel": "INTC",
    "amd": "AMD",
    "salesforce": "CRM",
    "oracle": "ORCL",
    "ibm": "IBM",
    "cisco": "CSCO",
    "adobe": "ADBE",
    "paypal": "PYPL",
    "uber": "UBER",
    "airbnb": "ABNB",
    "spotify": "SPOT",
    "zoom": "ZM",
    "shopify": "SHOP",
    "snowflake": "SNOW",
    "palantir": "PLTR",
    "coinbase": "COIN",
    "robinhood": "HOOD",
}


class RetrieverAgent:
    """
    Document retrieval agent with auto-ingestion.
    
    Features:
    - Automatic ticker extraction from queries
    - Auto-ingestion from SEC EDGAR if data not in Qdrant
    - Hybrid search orchestration (dense + sparse)
    - Relevance validation
    """
    
    MAX_ITERATIONS = 3
    MIN_RELEVANCE_SCORE = 0.3
    
    def __init__(
        self,
        hybrid_searcher: HybridSearcher = None,
        reranker: Reranker = None,
        model: str = None
    ):
        """
        Initialize retriever agent with all required components.
        
        Args:
            hybrid_searcher: Hybrid search component (auto-created if None)
            reranker: Reranking component
            model: LLM model for query reformulation
        """
        # Initialize core components
        self.vector_store = VectorStore()
        self.embedding_service = EmbeddingService()
        self.bm25_index = BM25Index()
        self.price_tool = PriceLookupTool()
        self.hybrid_searcher = hybrid_searcher
        self.reranker = reranker
        self.model = model or settings.LLM_MODEL
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        
    async def initialize(self):
        """
        Perform async initialization tasks.
        - Load BM25 index
        - Create vector store collection
        - Initialize hybrid searcher if needed
        """
        # Load BM25 index if exists
        import os
        self.index_path = os.path.join(os.getcwd(), "data", "indexes", "bm25.pkl")
        try:
            if os.path.exists(self.index_path):
                self.bm25_index.load_index(self.index_path)
            else:
                logger.info("No existing BM25 index found, starting fresh")
        except Exception as e:
            logger.warning(f"Failed to load BM25 index: {e}")
        
        # Create hybrid searcher if not provided
        if self.hybrid_searcher is None:
            self.hybrid_searcher = HybridSearcher(
                vector_store=self.vector_store,
                bm25_index=self.bm25_index,
                embedding_service=self.embedding_service
            )
            
        # Ensure collection exists
        try:
            self.vector_store.create_collection()
            logger.info("Vector store collection ready")
        except Exception as e:
            logger.warning(f"Could not create collection: {e}")
    
    def extract_ticker_from_query(self, query: str) -> Optional[str]:
        """
        Extract stock ticker from a natural language query.
        
        Args:
            query: User's query text
            
        Returns:
            Ticker symbol if found, None otherwise
        """
        query_lower = query.lower()
        invalid_tickers = {"PRICE", "TICKER", "SYMBOL"}
        
        # Check for explicit ticker patterns (e.g., $MSFT, MSFT:, ticker MSFT)
        ticker_patterns = [
            r'\$([A-Z]{1,5})\b',  # $MSFT
            r'\bticker[:\s]+([A-Z]{1,5})\b',  # ticker: MSFT
            r'\b([A-Z]{2,5})\s+stock\b',  # MSFT stock
            r'\bstock\s+([A-Z]{2,5})\b',  # stock MSFT
        ]
        
        for pattern in ticker_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                candidate = match.group(1).upper()
                if candidate in invalid_tickers:
                    logger.info(f"Ignoring placeholder ticker {candidate}")
                    continue
                return candidate
        
        # Check company name mappings
        for company, ticker in TICKER_MAPPINGS.items():
            if company in query_lower:
                logger.info(f"Extracted ticker {ticker} from company name '{company}'")
                return ticker
        
        # Check for standalone uppercase tickers (2-5 chars)
        ticker_match = re.search(r'\b([A-Z]{2,5})\b', query)
        if ticker_match:
            potential_ticker = ticker_match.group(1)
            # Filter out common words that look like tickers
            common_words = {
                "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HAD",
                "HER", "WAS", "ONE", "OUR", "OUT", "PRICE", "STOCK", "MARKET", "CURRENT"
            }
            common_words |= invalid_tickers
            if potential_ticker not in common_words:
                return potential_ticker
        
        return None
    
    async def check_data_exists(self, ticker: str) -> bool:
        """
        Check if we have data for a ticker in Qdrant.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            True if data exists, False otherwise
        """
        try:
            # Try to search for any documents with this ticker
            query_embedding = await self.embedding_service.embed_query(f"{ticker} company")
            results = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=1,
                filters={"ticker": ticker}
            )
            exists = len(results) > 0
            logger.info(f"Data exists for {ticker}: {exists}")
            return exists
        except Exception as e:
            logger.warning(f"Error checking data for {ticker}: {e}")
            return False
    
    async def auto_ingest_company(self, ticker: str) -> bool:
        """
        Automatically ingest SEC filings for a company.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            True if ingestion successful, False otherwise
        """
        logger.info(f"Auto-ingesting SEC filings for {ticker}")
        
        try:
            from app.ingestion.sec_edgar_loader import SECEdgarLoader
            from app.chunking.sec_chunker import SECChunker
            from datetime import datetime
            
            loader = SECEdgarLoader()
            chunker = SECChunker()
            
            # Fetch recent 10-K and 10-Q filings
            all_filings = []
            for doc_type in ["10-K", "10-Q"]:
                filings = await loader.get_filings(
                    ticker=ticker,
                    filing_type=doc_type,
                    limit=2  # Get last 2 of each type
                )
                all_filings.extend(filings)
            
            if not all_filings:
                logger.warning(f"No SEC filings found for {ticker}")
                return False
            
            logger.info(f"Found {len(all_filings)} filings for {ticker}")
            
            total_chunks = 0
            all_chunks = []
            
            for filing in all_filings:
                try:
                    # Download filing content
                    content = await loader.download_filing(
                        accession_number=filing["accession_number"],
                        cik=filing["cik"],
                        primary_document=filing.get("primary_document")
                    )
                    
                    if not content or len(content) < 100:
                        continue
                    
                    # Parse filing date
                    filing_date = filing.get("filing_date")
                    if isinstance(filing_date, str):
                        filing_date = datetime.strptime(filing_date, "%Y-%m-%d")
                    elif not isinstance(filing_date, datetime):
                        filing_date = datetime.now()

                    report_date = filing.get("report_date")
                    if isinstance(report_date, str):
                        try:
                            report_date = datetime.strptime(report_date, "%Y-%m-%d")
                        except ValueError:
                            report_date = None

                    # Map filing type
                    doc_type_map = {
                        "10-K": DocumentType.SEC_10K,
                        "10-Q": DocumentType.SEC_10Q,
                        "8-K": DocumentType.SEC_8K,
                    }
                    doc_type = doc_type_map.get(filing.get("filing_type", "10-K"), DocumentType.SEC_10K)
                    
                    derived = derive_fiscal_metadata(
                        report_date=report_date,
                        fiscal_year_end_mmdd=filing.get("fiscal_year_end"),
                        document_type=doc_type
                    )

                    metadata = DocumentMetadata(
                        ticker=ticker,
                        company_name=filing.get("company_name", ticker),
                        document_type=doc_type,
                        filing_date=filing_date,
                        fiscal_year=derived.fiscal_year,
                        fiscal_quarter=derived.fiscal_quarter,
                        fiscal_period=derived.fiscal_period,
                        period_end_date=derived.period_end_date,
                        source_url=filing.get("url", ""),
                        accession_number=filing.get("accession_number")
                    )
                    
                    # Chunk the document
                    chunks = chunker.chunk_document(content, metadata)
                    logger.info(f"Created {len(chunks)} chunks from {filing.get('filing_type')}")
                    
                    # Generate embeddings
                    texts = [chunk.content for chunk in chunks]
                    embeddings = await self.embedding_service.embed_texts(texts)
                    
                    for chunk, embedding in zip(chunks, embeddings):
                        chunk.embedding = embedding
                    
                    all_chunks.extend(chunks)
                    total_chunks += len(chunks)
                    
                except Exception as e:
                    logger.warning(f"Failed to process filing: {e}")
                    continue
            
            # Store all chunks in Qdrant
            if all_chunks:
                await self.vector_store.upsert_chunks(all_chunks)
                # Also update BM25 index
                self.bm25_index.add_documents(all_chunks)
                # Save BM25 index to disk
                self.bm25_index.save_index(self.index_path)
                logger.info(f"Successfully ingested {total_chunks} chunks for {ticker}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Auto-ingestion failed for {ticker}: {e}", exc_info=True)
            return False
    
    async def fast_retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[RetrievedDocument]:
        """
        Fast retrieval for simple queries - bypasses planning.
        
        Directly retrieves documents for the raw query without
        decomposition or complex filtering. Optimized for speed.
        
        Args:
            query: Raw search query
            top_k: Number of documents to return (default 5 for speed)
            
        Returns:
            List of retrieved documents
        """
        logger.info(f"Fast retrieving for: '{query[:50]}...'")
        
        # Extract ticker for filtering
        ticker = self.extract_ticker_from_query(query)
        filters = {"ticker": ticker} if ticker else None
        
        # Check if we need to auto-ingest
        if ticker:
            has_data = await self.check_data_exists(ticker)
            if not has_data:
                logger.info(f"No data for {ticker}, auto-ingesting...")
                await self.auto_ingest_company(ticker)
        
        # Direct hybrid search without reranking for speed
        try:
            results = await self.hybrid_searcher.search(
                query=query,
                top_k=top_k,
                filters=filters
            )
            logger.info(f"Fast retrieve found {len(results)} documents")
            return results
        except Exception as e:
            logger.error(f"Fast retrieve failed: {e}")
            return []
    
    async def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = None
    ) -> List[RetrievedDocument]:
        """
        Retrieve relevant documents for a query.
        
        This method:
        1. Extracts ticker from query
        2. Checks if data exists in Qdrant
        3. Auto-ingests from SEC if needed
        4. Performs hybrid search
        
        Args:
            query: Search query or information need
            filters: Metadata filters
            top_k: Number of documents to return
            
        Returns:
            List of retrieved documents
        """
        top_k = top_k or settings.RETRIEVAL_TOP_K
        logger.info(f"Retrieving documents for query: '{query[:50]}...'")
        
        results = []
        
        # Extract ticker from query
        ticker = self.extract_ticker_from_query(query)
        
        if ticker:
            logger.info(f"Extracted ticker: {ticker}")
            
            # 1. Fetch Market Data if relevant
            if self._is_market_data_query(query):
                logger.info(f"Detected market data query for {ticker}")
                try:
                    market_docs = await self._fetch_market_data(ticker, query)
                    results.extend(market_docs)
                except Exception as e:
                    logger.error(f"Failed to fetch market data: {e}")

            # 2. Check if we have data for this ticker
            has_data = await self.check_data_exists(ticker)
            
            if not has_data:
                logger.info(f"No data found for {ticker}, initiating auto-ingestion")
                ingestion_success = await self.auto_ingest_company(ticker)
                
                if not ingestion_success:
                    logger.warning(f"Auto-ingestion failed for {ticker}")
                    # If we have market data, return that at least
                    if results:
                        return results
                    return []
            
            # Add ticker filter if extracted
            if filters is None:
                filters = {}
            filters["ticker"] = ticker

        filters = self._apply_temporal_filters(query, filters)
        
        # 3. Perform hybrid search with reranking
        try:
            doc_results = await self._search_and_rerank(
                query=query,
                filters=filters,
                top_k=top_k
            )
            
            logger.info(f"Retrieved {len(doc_results)} documents from search")
            results.extend(doc_results)
            
            # If no results and we have a ticker, try without filter
            if not doc_results and ticker and not results:
                logger.info("No filtered results, trying broader search")
                broad_results = await self._search_and_rerank(
                    query=query,
                    filters=None,
                    top_k=top_k
                )
                results.extend(broad_results)
            
            # Deduplicate just in case
            return self._merge_results([results])
            
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return results

    def _apply_temporal_filters(
        self,
        query: str,
        filters: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Apply fiscal period filters derived from the query.
        """
        constraints = extract_temporal_constraints(query)
        merged = merge_temporal_filters(filters, constraints)
        return merged if merged else None

    def _is_market_data_query(self, query: str) -> bool:
        """Check if query is asking for market data (price, volume, etc.)."""
        keywords = {
            "price", "stock", "share", "value", "valuation", "performance",
            "open", "close", "high", "low", "volume", "market cap", "cap",
            "trading", "today", "current", "history", "trend"
        }
        query_lower = query.lower()
        return any(k in query_lower for k in keywords)

    async def _fetch_market_data(self, ticker: str, query: str) -> List[RetrievedDocument]:
        """Fetch market data using PriceLookupTool."""
        import uuid
        from datetime import datetime
        
        # Determine operation
        query_lower = query.lower()
        op = "current"
        if "history" in query_lower or "trend" in query_lower or "year" in query_lower:
            op = "change 1y"
        elif "month" in query_lower:
            op = "change 1m"
            
        tool_query = f"{ticker} {op}"
        logger.info(f"Calling PriceLookupTool with: {tool_query}")
        
        result_text = await self.price_tool._arun(tool_query)
        
        if "Error" in result_text and "No data" not in result_text:
            logger.warning(f"Price tool returned error: {result_text}")
            return []
            
        # Create a synthetic document chunk
        chunk = DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            document_id=f"market_data_{ticker}",
            content=f"MARKET DATA FOR {ticker}:\n{result_text}",
            metadata=DocumentMetadata(
                ticker=ticker,
                company_name=ticker, # We might not have full name
                document_type=DocumentType.MARKET_DATA,
                filing_date=datetime.now(),
                source_url=f"https://finance.yahoo.com/quote/{ticker}",
                accession_number=None
            ),
            section="Market Data",
            chunk_index=0
        )
        
        return [RetrievedDocument(
            chunk=chunk,
            score=1.0, # High confidence for direct data
            retrieval_method="tool"
        )]
    
    def _extract_parallel_queries(self, state: AgentState) -> List[Tuple[str, Optional[Dict]]]:
        """
        Extract queries that can run in parallel from planner output.
        
        Args:
            state: Current agent state with sub_queries
            
        Returns:
            List of (query_text, filter_dict) tuples
        """
        parallel_queries = []
        
        if not state.sub_queries:
            return [(state.original_query, state.filters)]
        
        for sub_query in state.sub_queries:
            query_text = sub_query.sub_query
            
            # Extract ticker from sub-query for filtering
            ticker = self.extract_ticker_from_query(query_text)
            filters = {"ticker": ticker} if ticker else state.filters
            
            parallel_queries.append((query_text, filters))
        
        return parallel_queries
    
    def _merge_parallel_results(
        self,
        results: Dict[str, List[RetrievedDocument]]
    ) -> List[RetrievedDocument]:
        """
        Merge results from parallel queries.
        
        - Deduplicates by chunk_id
        - Preserves highest score when duplicates found
        - Maintains source query attribution for citations
        
        Args:
            results: Dict mapping query -> list of documents
            
        Returns:
            Merged and deduplicated list of documents
        """
        seen_chunks: Dict[str, RetrievedDocument] = {}
        
        for query, docs in results.items():
            for doc in docs:
                chunk_id = doc.chunk.chunk_id
                if chunk_id not in seen_chunks:
                    seen_chunks[chunk_id] = doc
                elif doc.score > seen_chunks[chunk_id].score:
                    # Keep higher-scored version
                    seen_chunks[chunk_id] = doc
        
        # Sort by score descending
        merged = sorted(seen_chunks.values(), key=lambda x: x.score, reverse=True)
        return merged
    
    async def retrieve_for_state(self, state: AgentState) -> Dict[str, Any]:
        """
        Retrieve documents and update agent state.

        LangGraph-compatible interface - returns dict with updated fields.
        Uses parallel retrieval for multi-entity queries.

        Args:
            state: Current agent state

        Returns:
            Dict with retrieved_docs field for state update
        """
        new_events = list(state.step_events) if state.step_events else []

        # Handle sub-queries if present - use parallel retrieval
        if state.sub_queries and len(state.sub_queries) > 1:
            # Extract parallelizable queries
            parallel_queries = self._extract_parallel_queries(state)
            
            # Emit parallel retrieval start event
            retrieval_start_event = StepEvent(
                event_type="retrieval_progress",
                agent=AgentRole.RETRIEVER,
                timestamp=datetime.now(),
                data={
                    "status": "parallel_searching",
                    "num_queries": len(parallel_queries),
                    "queries": [q[0][:50] for q in parallel_queries]
                }
            )
            new_events.append(retrieval_start_event)
            
            # Use parallel retrieval if hybrid_searcher supports it
            if hasattr(self.hybrid_searcher, 'retrieve_parallel') and len(parallel_queries) > 1:
                queries, filters = zip(*parallel_queries)
                results = await self.hybrid_searcher.retrieve_parallel(
                    list(queries),
                    list(filters),
                    top_k=5
                )
                all_docs = self._merge_parallel_results(results)
            else:
                # Fallback to sequential retrieval
                all_docs = []
                for query_text, filters in parallel_queries:
                    docs = await self.retrieve(query_text, filters=filters)
                    all_docs.extend(docs)
                all_docs = self._merge_results([all_docs])
            
            # Emit retrieval complete event
            if all_docs:
                sources_found = set()
                for doc in all_docs:
                    source_name = f"{doc.chunk.metadata.company_name} {doc.chunk.metadata.document_type.value} {doc.chunk.metadata.filing_date.year}"
                    sources_found.add(source_name)

                retrieval_complete_event = StepEvent(
                    event_type="retrieval_progress",
                    agent=AgentRole.RETRIEVER,
                    timestamp=datetime.now(),
                    data={
                        "status": "parallel_complete",
                        "chunks_found": len(all_docs),
                        "sources": list(sources_found),
                        "avg_score": sum(d.score for d in all_docs) / len(all_docs) if all_docs else 0
                    }
                )
                new_events.append(retrieval_complete_event)

            return {
                "retrieved_docs": all_docs,
                "step_events": new_events
            }
        
        # Single sub-query or no sub-queries - use standard retrieval
        elif state.sub_queries:
            all_docs = []
            for idx, sub_query in enumerate(state.sub_queries, 1):
                # Emit retrieval start event
                retrieval_start_event = StepEvent(
                    event_type="retrieval_progress",
                    agent=AgentRole.RETRIEVER,
                    timestamp=datetime.now(),
                    data={
                        "status": "searching",
                        "query": sub_query.sub_query,
                        "sub_query_index": idx,
                        "total_sub_queries": len(state.sub_queries)
                    }
                )
                new_events.append(retrieval_start_event)

                docs = await self.retrieve(
                    sub_query.sub_query,
                    filters=state.filters
                )
                all_docs.extend(docs)

                # Emit retrieval complete event with results
                if docs:
                    # Get unique sources from retrieved docs
                    sources_found = set()
                    for doc in docs:
                        source_name = f"{doc.chunk.metadata.company_name} {doc.chunk.metadata.document_type.value} {doc.chunk.metadata.filing_date.year}"
                        sources_found.add(source_name)

                    retrieval_complete_event = StepEvent(
                        event_type="retrieval_progress",
                        agent=AgentRole.RETRIEVER,
                        timestamp=datetime.now(),
                        data={
                            "status": "found",
                            "query": sub_query.sub_query,
                            "chunks_found": len(docs),
                            "sources": list(sources_found),
                            "avg_score": sum(d.score for d in docs) / len(docs) if docs else 0
                        }
                    )
                    new_events.append(retrieval_complete_event)

            # Deduplicate by chunk_id
            seen = set()
            unique_docs = []
            for doc in all_docs:
                if doc.chunk.chunk_id not in seen:
                    seen.add(doc.chunk.chunk_id)
                    unique_docs.append(doc)

            return {
                "retrieved_docs": unique_docs,
                "step_events": new_events
            }
        else:
            # Single query retrieval
            # Emit retrieval start event
            retrieval_start_event = StepEvent(
                event_type="retrieval_progress",
                agent=AgentRole.RETRIEVER,
                timestamp=datetime.now(),
                data={
                    "status": "searching",
                    "query": state.original_query
                }
            )
            new_events.append(retrieval_start_event)

            docs = await self.retrieve(
                state.original_query,
                filters=state.filters
            )

            # Emit retrieval complete event
            if docs:
                sources_found = set()
                for doc in docs:
                    source_name = f"{doc.chunk.metadata.company_name} {doc.chunk.metadata.document_type.value} {doc.chunk.metadata.filing_date.year}"
                    sources_found.add(source_name)

                retrieval_complete_event = StepEvent(
                    event_type="retrieval_progress",
                    agent=AgentRole.RETRIEVER,
                    timestamp=datetime.now(),
                    data={
                        "status": "found",
                        "chunks_found": len(docs),
                        "sources": list(sources_found),
                        "avg_score": sum(d.score for d in docs) / len(docs) if docs else 0
                    }
                )
                new_events.append(retrieval_complete_event)

            return {
                "retrieved_docs": docs,
                "step_events": new_events
            }
    
    async def reformulate_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Generate alternative query formulations.
        
        Args:
            query: Original query
            context: Additional context
            
        Returns:
            List of query variations
        """
        try:
            # Simple few-shot prompt for reformulation
            system_prompt = """You are a financial research assistant. 
Your goal is to generate 3 alternative search queries for the given user question.
Focus on financial terminology, synonyms, and specific SEC filing sections.
Output a JSON array of strings."""
            
            user_prompt = f"Original query: {query}"
            if context:
                user_prompt += f"\nContext: {context}"
                
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            import json
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # Expecting {"queries": [...]} or just [...]
            if isinstance(result, list):
                variations = result
            elif isinstance(result, dict):
                variations = result.get("queries", list(result.values())[0] if result else [])
            else:
                variations = []
                
            # Ensure they are strings
            variations = [str(v) for v in variations if v]
            
            # Always include original
            if query not in variations:
                variations.insert(0, query)
                
            return variations[:5] # Limit to 5
            
        except Exception as e:
            logger.warning(f"Query reformulation failed: {e}")
            return [query]
    
    async def _search_and_rerank(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        top_k: int
    ) -> List[RetrievedDocument]:
        """
        Execute search and reranking pipeline.
        
        Args:
            query: Search query
            filters: Metadata filters
            top_k: Number of results
            
        Returns:
            Reranked documents
        """
        # Get more results for reranking
        results = await self.hybrid_searcher.search(
            query=query,
            top_k=top_k * 2,
            filters=filters
        )
        
        # Apply reranking if available
        if self.reranker and results:
            try:
                results = await self.reranker.rerank(
                    query=query,
                    documents=results,
                    top_k=top_k
                )
            except Exception as e:
                logger.warning(f"Reranking failed, using original order: {e}")
                results = results[:top_k]
        else:
            results = results[:top_k]

        return self._apply_score_threshold(results)

    def _apply_score_threshold(
        self,
        documents: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """
        Filter documents below the retrieval score threshold.
        """
        threshold = getattr(settings, "RETRIEVAL_SCORE_THRESHOLD", 0.0)
        if threshold <= 0:
            return documents
        filtered = [doc for doc in documents if doc.score >= threshold]
        if not filtered:
            logger.info(f"All retrieved documents below threshold {threshold}")
        return filtered
    
    def _validate_results(
        self,
        documents: List[RetrievedDocument],
        query: str
    ) -> bool:
        """
        Validate that results are relevant to query.
        
        Args:
            documents: Retrieved documents
            query: Original query
            
        Returns:
            True if results are sufficiently relevant
        """
        if not documents:
            return False
        
        # Check average relevance score
        avg_score = sum(d.score for d in documents) / len(documents)
        return avg_score >= self.MIN_RELEVANCE_SCORE
    
    def _merge_results(
        self,
        results_list: List[List[RetrievedDocument]]
    ) -> List[RetrievedDocument]:
        """
        Merge results from multiple queries.
        
        Args:
            results_list: List of result lists
            
        Returns:
            Merged and deduplicated results
        """
        seen = set()
        merged = []
        
        for results in results_list:
            for doc in results:
                if doc.chunk.chunk_id not in seen:
                    seen.add(doc.chunk.chunk_id)
                    merged.append(doc)
        
        # Sort by score
        merged.sort(key=lambda x: x.score, reverse=True)
        return merged
    
    def get_retrieval_stats(
        self,
        documents: List[RetrievedDocument]
    ) -> Dict[str, Any]:
        """
        Get statistics about retrieved documents.
        
        Args:
            documents: Retrieved documents
            
        Returns:
            Statistics dictionary
        """
        if not documents:
            return {"count": 0}
        
        return {
            "count": len(documents),
            "avg_score": sum(d.score for d in documents) / len(documents),
            "max_score": max(d.score for d in documents),
            "min_score": min(d.score for d in documents),
            "retrieval_methods": list(set(d.retrieval_method for d in documents)),
            "document_types": list(set(
                d.chunk.metadata.document_type.value for d in documents
            ))
        }
