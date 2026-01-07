"""
Evaluation Datasets

Financial QA datasets for evaluating the RAG system.
Includes curated test cases with ground truth answers.

Dataset types:
- Simple factual questions
- Multi-hop reasoning questions
- Comparative analysis questions
- Numerical calculation questions

Usage:
    dataset = FinancialQADataset.load("test_set_v1")
    for example in dataset:
        result = await system.query(example.question)
        score = evaluate(result, example.answer)
"""

import json
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum


class QuestionType(Enum):
    """Types of financial questions."""
    FACTUAL = "factual"
    MULTI_HOP = "multi_hop"
    COMPARATIVE = "comparative"
    NUMERICAL = "numerical"
    TREND = "trend"


@dataclass
class QAExample:
    """A single QA example for evaluation."""
    question_id: str
    question: str
    question_type: QuestionType
    answer: str
    supporting_facts: List[str]  # Chunk IDs that support the answer
    ticker: Optional[str] = None
    difficulty: str = "medium"  # easy, medium, hard
    metadata: Optional[Dict[str, Any]] = None


class FinancialQADataset:
    """
    Dataset of financial QA examples for evaluation.
    
    Supports loading from JSON files and programmatic creation.
    """
    
    def __init__(self, examples: List[QAExample] = None):
        """
        Initialize dataset.
        
        Args:
            examples: List of QA examples
        """
        self.examples = examples or []
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __iter__(self) -> Iterator[QAExample]:
        return iter(self.examples)
    
    def __getitem__(self, idx: int) -> QAExample:
        return self.examples[idx]
    
    @classmethod
    def load(cls, path: str) -> "FinancialQADataset":
        """
        Load dataset from JSON file.
        
        Args:
            path: Path to JSON file
            
        Returns:
            FinancialQADataset instance
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        examples = []
        for item in data.get("examples", []):
            example = QAExample(
                question_id=item["question_id"],
                question=item["question"],
                question_type=QuestionType(item["question_type"]),
                answer=item["answer"],
                supporting_facts=item.get("supporting_facts", []),
                ticker=item.get("ticker"),
                difficulty=item.get("difficulty", "medium"),
                metadata=item.get("metadata")
            )
            examples.append(example)
        
        return cls(examples)
    
    def save(self, path: str) -> None:
        """
        Save dataset to JSON file.
        
        Args:
            path: Path to save to
        """
        data = {
            "examples": [
                {
                    **asdict(ex),
                    "question_type": ex.question_type.value
                }
                for ex in self.examples
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def filter_by_type(self, question_type: QuestionType) -> "FinancialQADataset":
        """
        Filter examples by question type.
        
        Args:
            question_type: Type to filter by
            
        Returns:
            Filtered dataset
        """
        filtered = [ex for ex in self.examples if ex.question_type == question_type]
        return FinancialQADataset(filtered)
    
    def filter_by_ticker(self, ticker: str) -> "FinancialQADataset":
        """
        Filter examples by ticker.
        
        Args:
            ticker: Ticker symbol to filter by
            
        Returns:
            Filtered dataset
        """
        filtered = [ex for ex in self.examples if ex.ticker == ticker]
        return FinancialQADataset(filtered)
    
    def filter_by_difficulty(self, difficulty: str) -> "FinancialQADataset":
        """
        Filter examples by difficulty.
        
        Args:
            difficulty: Difficulty level (easy, medium, hard)
            
        Returns:
            Filtered dataset
        """
        filtered = [ex for ex in self.examples if ex.difficulty == difficulty]
        return FinancialQADataset(filtered)
    
    def add_example(self, example: QAExample) -> None:
        """
        Add an example to the dataset.
        
        Args:
            example: QA example to add
        """
        self.examples.append(example)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Statistics dictionary
        """
        type_counts = {}
        difficulty_counts = {}
        ticker_counts = {}
        
        for ex in self.examples:
            # Count by type
            type_name = ex.question_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            
            # Count by difficulty
            difficulty_counts[ex.difficulty] = difficulty_counts.get(ex.difficulty, 0) + 1
            
            # Count by ticker
            if ex.ticker:
                ticker_counts[ex.ticker] = ticker_counts.get(ex.ticker, 0) + 1
        
        return {
            "total_examples": len(self.examples),
            "by_type": type_counts,
            "by_difficulty": difficulty_counts,
            "by_ticker": ticker_counts,
            "unique_tickers": len(ticker_counts)
        }
    
    @classmethod
    def create_sample_dataset(cls) -> "FinancialQADataset":
        """
        Create a sample dataset for testing.
        
        Returns:
            Sample FinancialQADataset
        """
        examples = [
            QAExample(
                question_id="q001",
                question="What was Apple's total revenue in fiscal year 2023?",
                question_type=QuestionType.FACTUAL,
                answer="Apple's total revenue in fiscal year 2023 was $383.3 billion.",
                supporting_facts=["aapl_10k_2023_revenue"],
                ticker="AAPL",
                difficulty="easy"
            ),
            QAExample(
                question_id="q002",
                question="How did Microsoft's cloud revenue grow from 2022 to 2023?",
                question_type=QuestionType.NUMERICAL,
                answer="Microsoft's Intelligent Cloud revenue grew from $75.3 billion in FY2022 to $87.9 billion in FY2023, representing a 17% year-over-year increase.",
                supporting_facts=["msft_10k_2022_cloud", "msft_10k_2023_cloud"],
                ticker="MSFT",
                difficulty="medium"
            ),
            QAExample(
                question_id="q003",
                question="Compare the gross margins of Apple and Microsoft in their most recent fiscal years.",
                question_type=QuestionType.COMPARATIVE,
                answer="Apple's gross margin was 44.1% in FY2023, while Microsoft's gross margin was 69.4% in FY2023. Microsoft's higher gross margin reflects its software-focused business model compared to Apple's hardware-heavy revenue mix.",
                supporting_facts=["aapl_10k_2023_margin", "msft_10k_2023_margin"],
                ticker=None,
                difficulty="hard"
            ),
            QAExample(
                question_id="q004",
                question="What are the main risk factors Tesla disclosed related to production?",
                question_type=QuestionType.FACTUAL,
                answer="Tesla's main production-related risk factors include: supply chain disruptions, dependency on key suppliers for battery cells, manufacturing complexity at scale, and potential production delays at new facilities.",
                supporting_facts=["tsla_10k_2023_risks"],
                ticker="TSLA",
                difficulty="medium"
            ),
            QAExample(
                question_id="q005",
                question="What was the trend in Amazon's operating income over the past three years?",
                question_type=QuestionType.TREND,
                answer="Amazon's operating income showed significant volatility: $24.9B in 2021, declining to $12.2B in 2022 due to increased costs, then recovering to $36.9B in 2023 as efficiency measures took effect.",
                supporting_facts=["amzn_10k_2021_oi", "amzn_10k_2022_oi", "amzn_10k_2023_oi"],
                ticker="AMZN",
                difficulty="hard"
            )
        ]
        
        return cls(examples)
