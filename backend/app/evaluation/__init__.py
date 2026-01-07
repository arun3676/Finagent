"""
Evaluation Module

Metrics and benchmarking for RAG system evaluation:
- Retrieval metrics (recall, precision, MRR)
- Generation metrics (faithfulness, relevance)
- End-to-end evaluation datasets
- Benchmark runner
"""

from app.evaluation.metrics import RetrievalMetrics, GenerationMetrics
from app.evaluation.datasets import FinancialQADataset
from app.evaluation.benchmark import BenchmarkRunner

__all__ = [
    "RetrievalMetrics",
    "GenerationMetrics",
    "FinancialQADataset",
    "BenchmarkRunner"
]
