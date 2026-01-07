"""
BM25 Sparse Index

Implements BM25 (Best Matching 25) for keyword-based retrieval.
Complements dense retrieval for exact term matching.

BM25 excels at:
- Exact ticker/company name matching
- Specific financial terms
- Rare words not well-represented in embeddings

Usage:
    index = BM25Index()
    index.build_index(documents)
    results = index.search("AAPL revenue Q4 2023", top_k=10)
"""

import math
import logging
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter, defaultdict
import re

from app.models import DocumentChunk, RetrievedDocument

logger = logging.getLogger(__name__)


class BM25Index:
    """
    BM25 sparse retrieval index.
    
    Parameters:
    - k1: Term frequency saturation (default 1.5)
    - b: Length normalization (default 0.75)
    
    Features:
    - Custom tokenization for financial text
    - Incremental index updates
    - Filtered search support
    """
    
    # BM25 parameters
    DEFAULT_K1 = 1.5
    DEFAULT_B = 0.75
    
    # Financial domain stopwords to remove
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can", "this",
        "that", "these", "those", "it", "its", "we", "our", "they", "their"
    }
    
    def __init__(
        self,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B
    ):
        """
        Initialize BM25 index.
        
        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        
        # Index structures
        self._documents: Dict[str, DocumentChunk] = {}  # chunk_id -> chunk
        self._doc_lengths: Dict[str, int] = {}  # chunk_id -> token count
        self._avg_doc_length: float = 0.0
        self._doc_freqs: Dict[str, int] = {}  # term -> document frequency
        self._inverted_index: Dict[str, Dict[str, int]] = {}  # term -> {chunk_id -> term_freq}
        self._total_docs: int = 0
    
    def build_index(self, chunks: List[DocumentChunk]) -> None:
        """
        Build BM25 index from document chunks.
        
        Args:
            chunks: List of document chunks to index
        """
        logger.info(f"Building BM25 index for {len(chunks)} documents")
        
        self._documents = {}
        self._doc_lengths = {}
        self._inverted_index = defaultdict(lambda: defaultdict(int))
        self._doc_freqs = defaultdict(int)
        
        total_length = 0
        
        for chunk in chunks:
            chunk_id = chunk.chunk_id
            self._documents[chunk_id] = chunk
            
            tokens = self.tokenize(chunk.content)
            self._doc_lengths[chunk_id] = len(tokens)
            total_length += len(tokens)
            
            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                self._inverted_index[term][chunk_id] = count
            
            unique_terms = set(tokens)
            for term in unique_terms:
                self._doc_freqs[term] += 1
        
        self._total_docs = len(chunks)
        self._avg_doc_length = total_length / self._total_docs if self._total_docs > 0 else 0
        
        logger.info(f"Index built: {self._total_docs} docs, {len(self._doc_freqs)} unique terms, avg length {self._avg_doc_length:.1f}")
    
    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """
        Add documents to existing index.
        
        Args:
            chunks: New chunks to add
        """
        for chunk in chunks:
            chunk_id = chunk.chunk_id
            self._documents[chunk_id] = chunk
            
            tokens = self.tokenize(chunk.content)
            self._doc_lengths[chunk_id] = len(tokens)
            
            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                self._inverted_index[term][chunk_id] = count
            
            unique_terms = set(tokens)
            for term in unique_terms:
                self._doc_freqs[term] += 1
        
        self._total_docs = len(self._documents)
        total_length = sum(self._doc_lengths.values())
        self._avg_doc_length = total_length / self._total_docs if self._total_docs > 0 else 0
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedDocument]:
        """
        Search index using BM25 scoring.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of retrieved documents with BM25 scores
        """
        query_terms = self.tokenize(query)
        
        if not query_terms:
            return []
        
        scores = {}
        for chunk_id in self._documents.keys():
            if filters and not self._apply_filters(chunk_id, filters):
                continue
            
            score = self._score_document(query_terms, chunk_id)
            if score > 0:
                scores[chunk_id] = score
        
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for chunk_id, score in sorted_results:
            chunk = self._documents[chunk_id]
            results.append(RetrievedDocument(
                chunk=chunk,
                score=score,
                retrieval_method="sparse"
            ))
        
        return results
    
    def _score_document(
        self,
        query_terms: List[str],
        chunk_id: str
    ) -> float:
        """
        Calculate BM25 score for a document.
        
        BM25 formula:
        score = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D|/avgdl))
        
        Args:
            query_terms: Tokenized query terms
            chunk_id: Document chunk ID
            
        Returns:
            BM25 score
        """
        score = 0.0
        doc_length = self._doc_lengths.get(chunk_id, 0)
        
        for term in query_terms:
            if term not in self._inverted_index:
                continue
            
            # Term frequency in document
            tf = self._inverted_index[term].get(chunk_id, 0)
            if tf == 0:
                continue
            
            # Inverse document frequency
            df = self._doc_freqs.get(term, 0)
            idf = math.log((self._total_docs - df + 0.5) / (df + 0.5) + 1)
            
            # BM25 term score
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self._avg_doc_length)
            
            score += idf * (numerator / denominator)
        
        return score
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing.
        
        Handles:
        - Lowercase conversion
        - Punctuation removal
        - Stopword filtering
        - Financial term preservation (tickers, numbers, acronyms)
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        text = text.lower()
        
        financial_terms = re.findall(r'\b(?:ebitda|ebit|gaap|eps|roe|roa|p/e|yoy|qoq|cagr|fcf)\b', text)
        
        tokens = re.findall(r'\b[a-z0-9]+(?:[.-][a-z0-9]+)*\b', text)
        
        tokens = [t for t in tokens if t not in self.STOPWORDS or len(t) <= 4]
        
        tokens.extend(financial_terms)
        
        return tokens
    
    def _apply_filters(
        self,
        chunk_id: str,
        filters: Dict[str, Any]
    ) -> bool:
        """
        Check if document passes filters.
        
        Args:
            chunk_id: Document chunk ID
            filters: Filter conditions
            
        Returns:
            True if document passes all filters
        """
        if not filters:
            return True
        
        chunk = self._documents.get(chunk_id)
        if not chunk:
            return False
        
        if "ticker" in filters and chunk.metadata.ticker != filters["ticker"]:
            return False
        
        if "document_type" in filters and chunk.metadata.document_type.value != filters["document_type"]:
            return False
        
        if "section" in filters and chunk.section != filters["section"]:
            return False
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.
        
        Returns:
            Index statistics dictionary
        """
        return {
            "total_documents": self._total_docs,
            "vocabulary_size": len(self._doc_freqs),
            "avg_document_length": self._avg_doc_length,
            "k1": self.k1,
            "b": self.b
        }
    
    def save_index(self, path: str) -> None:
        """
        Save index to disk.
        
        Args:
            path: File path to save to
        """
        # TODO: Implement index serialization
        raise NotImplementedError("Index saving not yet implemented")
    
    def load_index(self, path: str) -> None:
        """
        Load index from disk.
        
        Args:
            path: File path to load from
        """
        # TODO: Implement index deserialization
        raise NotImplementedError("Index loading not yet implemented")
