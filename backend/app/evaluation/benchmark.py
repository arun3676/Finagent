"""
Benchmark Runner

Runs evaluation benchmarks on the RAG system.
Produces detailed reports with metrics and analysis.

Features:
- Batch evaluation of test sets
- Metric aggregation and reporting
- Error analysis
- Performance tracking over time

Usage:
    runner = BenchmarkRunner(workflow)
    results = await runner.run_benchmark(dataset)
    runner.generate_report(results, "benchmark_report.json")
"""

import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path

from app.evaluation.metrics import RetrievalMetrics, GenerationMetrics
from app.evaluation.datasets import FinancialQADataset, QAExample
from app.agents.workflow import FinAgentWorkflow
from app.models import QueryResponse


@dataclass
class EvaluationResult:
    """Result for a single evaluation example."""
    question_id: str
    question: str
    expected_answer: str
    generated_answer: str
    retrieval_metrics: Dict[str, float]
    generation_metrics: Dict[str, float]
    latency_ms: int
    passed: bool
    error: Optional[str] = None


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    timestamp: str
    dataset_name: str
    total_examples: int
    passed_examples: int
    failed_examples: int
    avg_latency_ms: float
    retrieval_metrics: Dict[str, float]
    generation_metrics: Dict[str, float]
    results_by_type: Dict[str, Dict[str, float]]
    results_by_difficulty: Dict[str, Dict[str, float]]
    errors: List[Dict[str, str]]


class BenchmarkRunner:
    """
    Runs benchmarks on the RAG system.
    
    Evaluates retrieval and generation quality
    across a test dataset.
    """
    
    # Thresholds for passing
    PASS_THRESHOLDS = {
        "recall_at_5": 0.6,
        "answer_similarity": 0.5,
        "faithfulness": 0.7
    }
    
    def __init__(
        self,
        workflow: FinAgentWorkflow = None,
        retrieval_metrics: RetrievalMetrics = None,
        generation_metrics: GenerationMetrics = None
    ):
        """
        Initialize benchmark runner.
        
        Args:
            workflow: FinAgent workflow to evaluate
            retrieval_metrics: Retrieval metrics calculator
            generation_metrics: Generation metrics calculator
        """
        self.workflow = workflow
        self.retrieval_metrics = retrieval_metrics or RetrievalMetrics()
        self.generation_metrics = generation_metrics or GenerationMetrics()
    
    async def run_benchmark(
        self,
        dataset: FinancialQADataset,
        dataset_name: str = "default"
    ) -> BenchmarkReport:
        """
        Run benchmark on a dataset.
        
        Args:
            dataset: Dataset to evaluate
            dataset_name: Name for the benchmark
            
        Returns:
            BenchmarkReport with results
        """
        results = []
        errors = []
        
        for example in dataset:
            try:
                result = await self._evaluate_example(example)
                results.append(result)
            except Exception as e:
                errors.append({
                    "question_id": example.question_id,
                    "error": str(e)
                })
        
        # Aggregate results
        report = self._aggregate_results(results, dataset_name, errors)
        return report
    
    async def _evaluate_example(self, example: QAExample) -> EvaluationResult:
        """
        Evaluate a single example.
        
        Args:
            example: QA example to evaluate
            
        Returns:
            EvaluationResult
        """
        start_time = time.time()
        
        # Run the workflow
        # TODO: Implement actual workflow call
        # response = await self.workflow.run(example.question)
        
        # Placeholder for now
        response = QueryResponse(
            query=example.question,
            answer="Placeholder answer",
            citations=[],
            sources=[],
            confidence=0.0,
            processing_time_ms=0
        )
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Calculate retrieval metrics
        retrieval_scores = self._calculate_retrieval_metrics(
            response,
            example.supporting_facts
        )
        
        # Calculate generation metrics
        generation_scores = await self._calculate_generation_metrics(
            example.question,
            example.answer,
            response.answer
        )
        
        # Determine if passed
        passed = self._check_passed(retrieval_scores, generation_scores)
        
        return EvaluationResult(
            question_id=example.question_id,
            question=example.question,
            expected_answer=example.answer,
            generated_answer=response.answer,
            retrieval_metrics=retrieval_scores,
            generation_metrics=generation_scores,
            latency_ms=latency_ms,
            passed=passed
        )
    
    def _calculate_retrieval_metrics(
        self,
        response: QueryResponse,
        relevant_ids: List[str]
    ) -> Dict[str, float]:
        """
        Calculate retrieval metrics for a response.
        
        Args:
            response: System response
            relevant_ids: Ground truth relevant chunk IDs
            
        Returns:
            Dictionary of retrieval metrics
        """
        # Extract retrieved chunk IDs from citations
        retrieved_ids = [c.source_chunk_id for c in response.citations]
        relevant_set = set(relevant_ids)
        
        return {
            "recall_at_5": self.retrieval_metrics.recall_at_k(retrieved_ids, relevant_set, k=5),
            "precision_at_5": self.retrieval_metrics.precision_at_k(retrieved_ids, relevant_set, k=5),
            "mrr": self.retrieval_metrics.mrr(retrieved_ids, relevant_set),
            "hit_rate": self.retrieval_metrics.hit_rate(retrieved_ids, relevant_set, k=5)
        }
    
    async def _calculate_generation_metrics(
        self,
        question: str,
        expected: str,
        generated: str
    ) -> Dict[str, float]:
        """
        Calculate generation metrics.
        
        Args:
            question: Original question
            expected: Expected answer
            generated: Generated answer
            
        Returns:
            Dictionary of generation metrics
        """
        return {
            "answer_similarity": self.generation_metrics.answer_similarity(generated, expected),
            # TODO: Add LLM-based metrics when implemented
            # "faithfulness": await self.generation_metrics.faithfulness(generated, contexts),
            # "answer_relevance": await self.generation_metrics.answer_relevance(question, generated)
        }
    
    def _check_passed(
        self,
        retrieval_scores: Dict[str, float],
        generation_scores: Dict[str, float]
    ) -> bool:
        """
        Check if example passed thresholds.
        
        Args:
            retrieval_scores: Retrieval metric scores
            generation_scores: Generation metric scores
            
        Returns:
            True if passed all thresholds
        """
        all_scores = {**retrieval_scores, **generation_scores}
        
        for metric, threshold in self.PASS_THRESHOLDS.items():
            if metric in all_scores and all_scores[metric] < threshold:
                return False
        
        return True
    
    def _aggregate_results(
        self,
        results: List[EvaluationResult],
        dataset_name: str,
        errors: List[Dict[str, str]]
    ) -> BenchmarkReport:
        """
        Aggregate individual results into report.
        
        Args:
            results: List of evaluation results
            dataset_name: Name of dataset
            errors: List of errors encountered
            
        Returns:
            BenchmarkReport
        """
        if not results:
            return BenchmarkReport(
                timestamp=datetime.now().isoformat(),
                dataset_name=dataset_name,
                total_examples=0,
                passed_examples=0,
                failed_examples=0,
                avg_latency_ms=0,
                retrieval_metrics={},
                generation_metrics={},
                results_by_type={},
                results_by_difficulty={},
                errors=errors
            )
        
        # Aggregate metrics
        retrieval_agg = self._aggregate_metrics([r.retrieval_metrics for r in results])
        generation_agg = self._aggregate_metrics([r.generation_metrics for r in results])
        
        # Calculate pass/fail
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        
        # Average latency
        avg_latency = sum(r.latency_ms for r in results) / len(results)
        
        return BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            dataset_name=dataset_name,
            total_examples=len(results),
            passed_examples=passed,
            failed_examples=failed,
            avg_latency_ms=avg_latency,
            retrieval_metrics=retrieval_agg,
            generation_metrics=generation_agg,
            results_by_type={},  # TODO: Implement grouping
            results_by_difficulty={},  # TODO: Implement grouping
            errors=errors
        )
    
    def _aggregate_metrics(
        self,
        metrics_list: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Aggregate metrics across examples.
        
        Args:
            metrics_list: List of metric dictionaries
            
        Returns:
            Aggregated metrics (averages)
        """
        if not metrics_list:
            return {}
        
        aggregated = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m]
            if values:
                aggregated[key] = sum(values) / len(values)
        
        return aggregated
    
    def generate_report(
        self,
        report: BenchmarkReport,
        output_path: str
    ) -> None:
        """
        Save benchmark report to file.
        
        Args:
            report: Benchmark report
            output_path: Path to save report
        """
        with open(output_path, 'w') as f:
            json.dump(asdict(report), f, indent=2)
    
    def print_summary(self, report: BenchmarkReport) -> None:
        """
        Print a summary of the benchmark report.
        
        Args:
            report: Benchmark report to summarize
        """
        print("\n" + "=" * 60)
        print(f"BENCHMARK REPORT: {report.dataset_name}")
        print("=" * 60)
        print(f"Timestamp: {report.timestamp}")
        print(f"Total Examples: {report.total_examples}")
        print(f"Passed: {report.passed_examples} ({report.passed_examples/report.total_examples*100:.1f}%)")
        print(f"Failed: {report.failed_examples}")
        print(f"Average Latency: {report.avg_latency_ms:.0f}ms")
        print("\nRetrieval Metrics:")
        for metric, value in report.retrieval_metrics.items():
            print(f"  {metric}: {value:.3f}")
        print("\nGeneration Metrics:")
        for metric, value in report.generation_metrics.items():
            print(f"  {metric}: {value:.3f}")
        if report.errors:
            print(f"\nErrors: {len(report.errors)}")
        print("=" * 60 + "\n")
