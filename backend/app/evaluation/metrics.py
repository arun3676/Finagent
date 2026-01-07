"""
Evaluation Metrics

Implements metrics for evaluating RAG system performance:

Retrieval Metrics:
- Recall@K: Fraction of relevant docs in top K
- Precision@K: Fraction of top K that are relevant
- MRR: Mean Reciprocal Rank
- NDCG: Normalized Discounted Cumulative Gain

Generation Metrics:
- Faithfulness: Is answer supported by sources?
- Answer Relevance: Does answer address the question?
- Context Relevance: Are retrieved docs relevant?

Usage:
    metrics = RetrievalMetrics()
    recall = metrics.recall_at_k(retrieved, relevant, k=5)
"""

from typing import List, Dict, Any, Set, Optional
import math


class RetrievalMetrics:
    """
    Metrics for evaluating retrieval quality.
    
    All metrics return values between 0 and 1,
    where 1 is perfect performance.
    """
    
    def recall_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int = 10
    ) -> float:
        """
        Calculate Recall@K.
        
        Recall@K = |Retrieved ∩ Relevant| / |Relevant|
        
        Args:
            retrieved_ids: List of retrieved document IDs (ordered)
            relevant_ids: Set of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Recall score (0 to 1)
        """
        if not relevant_ids:
            return 1.0  # No relevant docs means perfect recall
        
        top_k = set(retrieved_ids[:k])
        hits = len(top_k & relevant_ids)
        
        return hits / len(relevant_ids)
    
    def precision_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int = 10
    ) -> float:
        """
        Calculate Precision@K.
        
        Precision@K = |Retrieved ∩ Relevant| / K
        
        Args:
            retrieved_ids: List of retrieved document IDs (ordered)
            relevant_ids: Set of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Precision score (0 to 1)
        """
        if k == 0:
            return 0.0
        
        top_k = set(retrieved_ids[:k])
        hits = len(top_k & relevant_ids)
        
        return hits / k
    
    def mrr(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank.
        
        MRR = 1 / rank of first relevant document
        
        Args:
            retrieved_ids: List of retrieved document IDs (ordered)
            relevant_ids: Set of relevant document IDs
            
        Returns:
            MRR score (0 to 1)
        """
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def ndcg_at_k(
        self,
        retrieved_ids: List[str],
        relevance_scores: Dict[str, float],
        k: int = 10
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain.
        
        NDCG = DCG / IDCG
        
        Args:
            retrieved_ids: List of retrieved document IDs (ordered)
            relevance_scores: Dict mapping doc IDs to relevance scores
            k: Number of top results to consider
            
        Returns:
            NDCG score (0 to 1)
        """
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            rel = relevance_scores.get(doc_id, 0.0)
            dcg += rel / math.log2(i + 2)  # +2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG)
        ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def hit_rate(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int = 10
    ) -> float:
        """
        Calculate Hit Rate (binary recall).
        
        1 if any relevant doc in top K, else 0.
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: Set of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            1.0 or 0.0
        """
        top_k = set(retrieved_ids[:k])
        return 1.0 if (top_k & relevant_ids) else 0.0
    
    def average_precision(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str]
    ) -> float:
        """
        Calculate Average Precision.
        
        AP = (1/|Relevant|) * Σ Precision@k * rel(k)
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: Set of relevant document IDs
            
        Returns:
            Average precision score
        """
        if not relevant_ids:
            return 1.0
        
        hits = 0
        sum_precision = 0.0
        
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                hits += 1
                precision_at_i = hits / (i + 1)
                sum_precision += precision_at_i
        
        return sum_precision / len(relevant_ids)


class GenerationMetrics:
    """
    Metrics for evaluating generation quality.
    
    Uses LLM-as-judge for semantic evaluation.
    """
    
    def __init__(self, judge_model: str = "gpt-4-turbo-preview"):
        """
        Initialize generation metrics.
        
        Args:
            judge_model: LLM model to use as judge
        """
        self.judge_model = judge_model
    
    async def faithfulness(
        self,
        answer: str,
        contexts: List[str]
    ) -> float:
        """
        Evaluate if answer is faithful to source contexts.
        
        Checks that all claims in answer are supported by contexts.
        
        Args:
            answer: Generated answer
            contexts: Source context passages
            
        Returns:
            Faithfulness score (0 to 1)
        """
        # TODO: Implement LLM-based faithfulness evaluation
        # 1. Extract claims from answer
        # 2. For each claim, check if supported by contexts
        # 3. Return fraction of supported claims
        raise NotImplementedError("Faithfulness evaluation not yet implemented")
    
    async def answer_relevance(
        self,
        question: str,
        answer: str
    ) -> float:
        """
        Evaluate if answer is relevant to the question.
        
        Args:
            question: Original question
            answer: Generated answer
            
        Returns:
            Relevance score (0 to 1)
        """
        # TODO: Implement LLM-based relevance evaluation
        # 1. Ask LLM to rate how well answer addresses question
        # 2. Parse and return score
        raise NotImplementedError("Answer relevance evaluation not yet implemented")
    
    async def context_relevance(
        self,
        question: str,
        contexts: List[str]
    ) -> float:
        """
        Evaluate if retrieved contexts are relevant to question.
        
        Args:
            question: Original question
            contexts: Retrieved context passages
            
        Returns:
            Context relevance score (0 to 1)
        """
        # TODO: Implement context relevance evaluation
        # 1. For each context, check relevance to question
        # 2. Return average relevance
        raise NotImplementedError("Context relevance evaluation not yet implemented")
    
    def answer_similarity(
        self,
        generated: str,
        reference: str
    ) -> float:
        """
        Calculate similarity between generated and reference answers.
        
        Uses simple token overlap (F1 score).
        
        Args:
            generated: Generated answer
            reference: Reference/ground truth answer
            
        Returns:
            Similarity score (0 to 1)
        """
        gen_tokens = set(generated.lower().split())
        ref_tokens = set(reference.lower().split())
        
        if not gen_tokens or not ref_tokens:
            return 0.0
        
        overlap = len(gen_tokens & ref_tokens)
        precision = overlap / len(gen_tokens)
        recall = overlap / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def citation_precision(
        self,
        citations: List[Dict[str, Any]],
        relevant_chunks: Set[str]
    ) -> float:
        """
        Calculate precision of citations.
        
        Args:
            citations: List of citation objects with chunk_ids
            relevant_chunks: Set of actually relevant chunk IDs
            
        Returns:
            Citation precision (0 to 1)
        """
        if not citations:
            return 0.0
        
        cited_chunks = {c.get("source_chunk_id") for c in citations}
        correct = len(cited_chunks & relevant_chunks)
        
        return correct / len(citations)
