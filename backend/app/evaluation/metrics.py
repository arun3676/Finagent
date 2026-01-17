"""
Evaluation Metrics

Implements metrics for evaluating RAG system performance using DeepEval.

Retrieval Metrics:
- Recall@K: Fraction of relevant docs in top K
- Precision@K: Fraction of top K that are relevant
- MRR: Mean Reciprocal Rank
- NDCG: Normalized Discounted Cumulative Gain

Generation Metrics (DeepEval):
- Faithfulness: Is answer supported by sources?
- Answer Relevance: Does answer address the question?
- Contextual Precision: Are retrieved docs relevant and well-ranked?

Usage:
    # Retrieval metrics
    metrics = RetrievalMetrics()
    recall = metrics.recall_at_k(retrieved, relevant, k=5)

    # Generation metrics (DeepEval)
    gen_metrics = GenerationMetrics()
    faithfulness = await gen_metrics.faithfulness(answer, contexts)

DeepEval Integration:
    # Run with pytest
    pytest tests/evaluation/ -v

    # Or use DeepEval CLI
    deepeval test run tests/evaluation/
"""

from typing import List, Dict, Any, Set, Optional
import math
import logging
import os

logger = logging.getLogger(__name__)

# Check if DeepEval is available
DEEPEVAL_AVAILABLE = False
try:
    from deepeval import evaluate
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        HallucinationMetric,
    )
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
    logger.info("DeepEval metrics loaded successfully")
except ImportError:
    logger.warning(
        "DeepEval not installed. Generation metrics will use fallback implementations. "
        "Install with: pip install deepeval"
    )


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
    Metrics for evaluating generation quality using DeepEval.

    Uses LLM-as-judge for semantic evaluation with industry-standard metrics:
    - FaithfulnessMetric: Checks if answer is grounded in context
    - AnswerRelevancyMetric: Checks if answer addresses the question
    - ContextualPrecisionMetric: Checks if retrieved contexts are relevant

    Requires: pip install deepeval
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        threshold: float = 0.7,
        include_reason: bool = True
    ):
        """
        Initialize generation metrics.

        Args:
            model: LLM model to use as judge (default: gpt-4o-mini for cost efficiency)
            threshold: Minimum score threshold for passing (default: 0.7)
            include_reason: Include reasoning in metric output
        """
        self.model = model
        self.threshold = threshold
        self.include_reason = include_reason

        # Initialize DeepEval metrics if available
        if DEEPEVAL_AVAILABLE:
            self._faithfulness_metric = FaithfulnessMetric(
                threshold=threshold,
                model=model,
                include_reason=include_reason
            )
            self._relevancy_metric = AnswerRelevancyMetric(
                threshold=threshold,
                model=model,
                include_reason=include_reason
            )
            self._contextual_precision_metric = ContextualPrecisionMetric(
                threshold=threshold,
                model=model,
                include_reason=include_reason
            )
            self._contextual_recall_metric = ContextualRecallMetric(
                threshold=threshold,
                model=model,
                include_reason=include_reason
            )
            self._hallucination_metric = HallucinationMetric(
                threshold=threshold,
                model=model,
                include_reason=include_reason
            )
            logger.info(f"DeepEval metrics initialized with model={model}, threshold={threshold}")

    async def faithfulness(
        self,
        answer: str,
        contexts: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate if answer is faithful to source contexts (grounded).

        Uses DeepEval FaithfulnessMetric which:
        1. Extracts claims from the answer
        2. For each claim, verifies it's supported by the contexts
        3. Returns the fraction of supported claims

        Args:
            answer: Generated answer
            contexts: Source context passages

        Returns:
            Dict with score, passed, and reason
        """
        if not DEEPEVAL_AVAILABLE:
            return self._fallback_faithfulness(answer, contexts)

        try:
            test_case = LLMTestCase(
                input="",  # Not needed for faithfulness
                actual_output=answer,
                retrieval_context=contexts
            )

            self._faithfulness_metric.measure(test_case)

            return {
                "score": self._faithfulness_metric.score,
                "passed": self._faithfulness_metric.is_successful(),
                "reason": self._faithfulness_metric.reason if self.include_reason else None,
                "metric": "faithfulness"
            }
        except Exception as e:
            logger.error(f"Faithfulness evaluation failed: {e}")
            return {"score": 0.0, "passed": False, "reason": str(e), "metric": "faithfulness"}

    async def answer_relevance(
        self,
        question: str,
        answer: str,
        contexts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate if answer is relevant to the question.

        Uses DeepEval AnswerRelevancyMetric which:
        1. Generates N hypothetical questions from the answer
        2. Measures semantic similarity between generated questions and original
        3. Returns average similarity score

        Args:
            question: Original question
            answer: Generated answer
            contexts: Optional retrieval contexts

        Returns:
            Dict with score, passed, and reason
        """
        if not DEEPEVAL_AVAILABLE:
            return self._fallback_relevance(question, answer)

        try:
            test_case = LLMTestCase(
                input=question,
                actual_output=answer,
                retrieval_context=contexts or []
            )

            self._relevancy_metric.measure(test_case)

            return {
                "score": self._relevancy_metric.score,
                "passed": self._relevancy_metric.is_successful(),
                "reason": self._relevancy_metric.reason if self.include_reason else None,
                "metric": "answer_relevance"
            }
        except Exception as e:
            logger.error(f"Answer relevance evaluation failed: {e}")
            return {"score": 0.0, "passed": False, "reason": str(e), "metric": "answer_relevance"}

    async def contextual_precision(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        expected_output: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate if retrieved contexts are relevant and well-ranked.

        Uses DeepEval ContextualPrecisionMetric which:
        1. Checks if relevant contexts appear before irrelevant ones
        2. Penalizes irrelevant contexts that rank highly
        3. Returns precision score accounting for ranking

        Args:
            question: Original question
            answer: Generated answer
            contexts: Retrieved context passages (in ranked order)
            expected_output: Optional expected/reference answer

        Returns:
            Dict with score, passed, and reason
        """
        if not DEEPEVAL_AVAILABLE:
            return self._fallback_context_relevance(question, contexts)

        try:
            test_case = LLMTestCase(
                input=question,
                actual_output=answer,
                retrieval_context=contexts,
                expected_output=expected_output or answer  # Use answer as reference if not provided
            )

            self._contextual_precision_metric.measure(test_case)

            return {
                "score": self._contextual_precision_metric.score,
                "passed": self._contextual_precision_metric.is_successful(),
                "reason": self._contextual_precision_metric.reason if self.include_reason else None,
                "metric": "contextual_precision"
            }
        except Exception as e:
            logger.error(f"Contextual precision evaluation failed: {e}")
            return {"score": 0.0, "passed": False, "reason": str(e), "metric": "contextual_precision"}

    async def contextual_recall(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        expected_output: str
    ) -> Dict[str, Any]:
        """
        Evaluate if contexts contain all information needed for the expected answer.

        Uses DeepEval ContextualRecallMetric which:
        1. Extracts claims from expected output
        2. Checks if each claim can be attributed to a context
        3. Returns fraction of attributable claims

        Args:
            question: Original question
            answer: Generated answer
            contexts: Retrieved context passages
            expected_output: Expected/reference answer

        Returns:
            Dict with score, passed, and reason
        """
        if not DEEPEVAL_AVAILABLE:
            return {"score": 0.0, "passed": False, "reason": "DeepEval not installed", "metric": "contextual_recall"}

        try:
            test_case = LLMTestCase(
                input=question,
                actual_output=answer,
                retrieval_context=contexts,
                expected_output=expected_output
            )

            self._contextual_recall_metric.measure(test_case)

            return {
                "score": self._contextual_recall_metric.score,
                "passed": self._contextual_recall_metric.is_successful(),
                "reason": self._contextual_recall_metric.reason if self.include_reason else None,
                "metric": "contextual_recall"
            }
        except Exception as e:
            logger.error(f"Contextual recall evaluation failed: {e}")
            return {"score": 0.0, "passed": False, "reason": str(e), "metric": "contextual_recall"}

    async def hallucination(
        self,
        answer: str,
        contexts: List[str]
    ) -> Dict[str, Any]:
        """
        Detect hallucinations in the answer.

        Uses DeepEval HallucinationMetric which:
        1. Identifies claims in the answer not supported by contexts
        2. Returns hallucination score (lower is better)

        Args:
            answer: Generated answer
            contexts: Source context passages

        Returns:
            Dict with score (0 = no hallucination), passed, and reason
        """
        if not DEEPEVAL_AVAILABLE:
            return {"score": 1.0, "passed": False, "reason": "DeepEval not installed", "metric": "hallucination"}

        try:
            test_case = LLMTestCase(
                input="",
                actual_output=answer,
                context=contexts
            )

            self._hallucination_metric.measure(test_case)

            return {
                "score": self._hallucination_metric.score,
                "passed": self._hallucination_metric.is_successful(),
                "reason": self._hallucination_metric.reason if self.include_reason else None,
                "metric": "hallucination"
            }
        except Exception as e:
            logger.error(f"Hallucination detection failed: {e}")
            return {"score": 1.0, "passed": False, "reason": str(e), "metric": "hallucination"}

    async def evaluate_response(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        expected_output: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run all metrics on a response and return comprehensive evaluation.

        Args:
            question: Original question
            answer: Generated answer
            contexts: Retrieved context passages
            expected_output: Optional expected/reference answer

        Returns:
            Dict with all metric scores and overall assessment
        """
        results = {}

        # Run all metrics
        results["faithfulness"] = await self.faithfulness(answer, contexts)
        results["answer_relevance"] = await self.answer_relevance(question, answer, contexts)
        results["contextual_precision"] = await self.contextual_precision(
            question, answer, contexts, expected_output
        )

        if expected_output:
            results["contextual_recall"] = await self.contextual_recall(
                question, answer, contexts, expected_output
            )

        # Calculate overall score
        scores = [r["score"] for r in results.values() if r.get("score") is not None]
        overall_score = sum(scores) / len(scores) if scores else 0.0
        all_passed = all(r.get("passed", False) for r in results.values())

        return {
            "metrics": results,
            "overall_score": overall_score,
            "all_passed": all_passed,
            "threshold": self.threshold
        }

    # Fallback implementations when DeepEval is not available
    def _fallback_faithfulness(self, answer: str, contexts: List[str]) -> Dict[str, Any]:
        """Simple keyword overlap fallback for faithfulness."""
        if not contexts:
            return {"score": 0.0, "passed": False, "reason": "No contexts provided", "metric": "faithfulness"}

        answer_words = set(answer.lower().split())
        context_words = set()
        for ctx in contexts:
            context_words.update(ctx.lower().split())

        overlap = len(answer_words & context_words)
        score = overlap / len(answer_words) if answer_words else 0.0

        return {
            "score": min(score, 1.0),
            "passed": score >= self.threshold,
            "reason": "Fallback: keyword overlap (install deepeval for LLM-based evaluation)",
            "metric": "faithfulness"
        }

    def _fallback_relevance(self, question: str, answer: str) -> Dict[str, Any]:
        """Simple keyword overlap fallback for relevance."""
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())

        overlap = len(question_words & answer_words)
        score = overlap / len(question_words) if question_words else 0.0

        return {
            "score": min(score, 1.0),
            "passed": score >= self.threshold,
            "reason": "Fallback: keyword overlap (install deepeval for LLM-based evaluation)",
            "metric": "answer_relevance"
        }

    def _fallback_context_relevance(self, question: str, contexts: List[str]) -> Dict[str, Any]:
        """Simple keyword overlap fallback for context relevance."""
        if not contexts:
            return {"score": 0.0, "passed": False, "reason": "No contexts provided", "metric": "contextual_precision"}

        question_words = set(question.lower().split())
        scores = []

        for ctx in contexts:
            ctx_words = set(ctx.lower().split())
            overlap = len(question_words & ctx_words)
            scores.append(overlap / len(question_words) if question_words else 0.0)

        avg_score = sum(scores) / len(scores) if scores else 0.0

        return {
            "score": min(avg_score, 1.0),
            "passed": avg_score >= self.threshold,
            "reason": "Fallback: keyword overlap (install deepeval for LLM-based evaluation)",
            "metric": "contextual_precision"
        }
    
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
