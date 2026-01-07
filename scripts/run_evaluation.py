#!/usr/bin/env python3
"""
Evaluation Runner Script

Runs benchmarks on the FinAgent RAG system and generates reports.

Usage:
    python scripts/run_evaluation.py --dataset data/evaluation/test_set.json
    python scripts/run_evaluation.py --sample  # Run on sample dataset
"""

import argparse
import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.evaluation.benchmark import BenchmarkRunner, BenchmarkReport
from app.evaluation.datasets import FinancialQADataset
from app.evaluation.metrics import RetrievalMetrics, GenerationMetrics


async def run_evaluation(
    dataset_path: str = None,
    use_sample: bool = False,
    output_dir: str = "data/evaluation/results"
):
    """
    Run evaluation benchmark.
    
    Args:
        dataset_path: Path to evaluation dataset JSON
        use_sample: Use built-in sample dataset
        output_dir: Directory to save results
    """
    print("=" * 60)
    print("🧪 FinAgent Evaluation Suite")
    print("=" * 60)
    
    # Load dataset
    if use_sample:
        print("📊 Loading sample dataset...")
        dataset = FinancialQADataset.create_sample_dataset()
        dataset_name = "sample_dataset"
    else:
        print(f"📊 Loading dataset from {dataset_path}...")
        dataset = FinancialQADataset.load(dataset_path)
        dataset_name = Path(dataset_path).stem
    
    # Print dataset statistics
    stats = dataset.get_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Total Examples: {stats['total_examples']}")
    print(f"  By Type: {stats['by_type']}")
    print(f"  By Difficulty: {stats['by_difficulty']}")
    
    # Initialize benchmark runner
    # TODO: Initialize with actual workflow when implemented
    runner = BenchmarkRunner(
        workflow=None,  # TODO: Add actual workflow
        retrieval_metrics=RetrievalMetrics(),
        generation_metrics=GenerationMetrics()
    )
    
    # Run benchmark
    print("\n🏃 Running benchmark...")
    print("=" * 60)
    
    try:
        report = await runner.run_benchmark(dataset, dataset_name)
    except NotImplementedError:
        print("⚠️  Workflow not yet implemented. Running mock evaluation...")
        # Create mock report for demonstration
        report = BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            dataset_name=dataset_name,
            total_examples=len(dataset),
            passed_examples=0,
            failed_examples=len(dataset),
            avg_latency_ms=0,
            retrieval_metrics={
                "recall_at_5": 0.0,
                "precision_at_5": 0.0,
                "mrr": 0.0
            },
            generation_metrics={
                "answer_similarity": 0.0
            },
            results_by_type={},
            results_by_difficulty={},
            errors=[{"error": "Workflow not implemented"}]
        )
    
    # Print summary
    runner.print_summary(report)
    
    # Save report
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_path / f"benchmark_{dataset_name}_{timestamp}.json"
    
    runner.generate_report(report, str(report_file))
    print(f"📄 Report saved to: {report_file}")
    
    return report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run FinAgent evaluation benchmarks"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to evaluation dataset JSON file"
    )
    
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use built-in sample dataset"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/evaluation/results",
        help="Directory to save evaluation results"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if not args.dataset and not args.sample:
        print("Error: Must specify --dataset or --sample")
        sys.exit(1)
    
    asyncio.run(run_evaluation(
        dataset_path=args.dataset,
        use_sample=args.sample,
        output_dir=args.output_dir
    ))
