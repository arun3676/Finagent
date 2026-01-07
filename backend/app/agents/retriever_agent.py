"""
Retriever Agent

Orchestrates document retrieval using hybrid search.
Handles query reformulation and iterative retrieval.

Retrieval strategy:
1. Generate search queries from information need
2. Execute hybrid search (dense + sparse)
3. Rerank results
4. Validate relevance
5. Iterate if needed

Usage:
    agent = RetrieverAgent(hybrid_searcher, reranker)
    docs = await agent.retrieve(query, filters)
"""

from typing import List, Optional, Dict, Any
from openai import AsyncOpenAI

from app.config import settings
from app.models import AgentState, RetrievedDocument, SubQuery
from app.retrieval.hybrid_search import HybridSearcher
from app.retrieval.reranker import Reranker
from app.agents.prompts import RETRIEVER_SYSTEM_PROMPT, RETRIEVER_USER_TEMPLATE


class RetrieverAgent:
    """
    Document retrieval agent.
    
    Features:
    - Query reformulation for better retrieval
    - Hybrid search orchestration
    - Relevance validation
    - Iterative retrieval for complex queries
    """
    
    MAX_ITERATIONS = 3
    MIN_RELEVANCE_SCORE = 0.5
    
    def __init__(
        self,
        hybrid_searcher: HybridSearcher = None,
        reranker: Reranker = None,
        model: str = None
    ):
        """
        Initialize retriever agent.
        
        Args:
            hybrid_searcher: Hybrid search component
            reranker: Reranking component
            model: LLM model for query reformulation
        """
        self.hybrid_searcher = hybrid_searcher
        self.reranker = reranker
        self.model = model or settings.LLM_MODEL
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    
    async def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = None
    ) -> List[RetrievedDocument]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query or information need
            filters: Metadata filters
            top_k: Number of documents to return
            
        Returns:
            List of retrieved documents
        """
        top_k = top_k or settings.RETRIEVAL_TOP_K
        
        # TODO: Implement retrieval pipeline
        # 1. Reformulate query if needed
        # 2. Execute hybrid search
        # 3. Rerank results
        # 4. Validate relevance
        # 5. Iterate if insufficient results
        raise NotImplementedError("Retrieval pipeline not yet implemented")
    
    async def retrieve_for_state(self, state: AgentState) -> AgentState:
        """
        Retrieve documents and update agent state.
        
        LangGraph-compatible interface.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with retrieved documents
        """
        # Handle sub-queries if present
        if state.sub_queries:
            all_docs = []
            for sub_query in state.sub_queries:
                docs = await self.retrieve(
                    sub_query.sub_query,
                    filters=state.filters
                )
                all_docs.extend(docs)
            
            # Deduplicate by chunk_id
            seen = set()
            unique_docs = []
            for doc in all_docs:
                if doc.chunk.chunk_id not in seen:
                    seen.add(doc.chunk.chunk_id)
                    unique_docs.append(doc)
            
            state.retrieved_docs = unique_docs
        else:
            # Single query retrieval
            state.retrieved_docs = await self.retrieve(
                state.original_query,
                filters=state.filters
            )
        
        return state
    
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
        # TODO: Implement query reformulation
        # 1. Call LLM to generate variations
        # 2. Include financial terminology variations
        # 3. Return original + variations
        raise NotImplementedError("Query reformulation not yet implemented")
    
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
        # TODO: Implement search + rerank
        # 1. Call hybrid searcher
        # 2. Call reranker
        # 3. Return reranked results
        raise NotImplementedError("Search and rerank not yet implemented")
    
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
