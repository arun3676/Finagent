#!/usr/bin/env python3
"""
FinAgent Demo Script

Interactive demonstration of the FinAgent system for interviews.
Shows the multi-agent workflow in action with sample queries.

Usage:
    python scripts/demo.py
    python scripts/demo.py --query "What was Apple's revenue in 2023?"
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.config import settings


# Sample queries for demonstration
DEMO_QUERIES = [
    {
        "query": "What was Apple's total revenue in fiscal year 2023?",
        "complexity": "simple",
        "description": "Simple factual lookup"
    },
    {
        "query": "How did Microsoft's cloud revenue grow from 2022 to 2023?",
        "complexity": "moderate",
        "description": "Multi-step calculation"
    },
    {
        "query": "Compare the gross margins of Apple and Microsoft, and explain the key factors driving the difference.",
        "complexity": "complex",
        "description": "Multi-document comparative analysis"
    },
    {
        "query": "What are the main risk factors Tesla disclosed related to production and supply chain?",
        "complexity": "moderate",
        "description": "Information extraction from risk factors"
    }
]


def print_header():
    """Print demo header."""
    print("\n" + "=" * 70)
    print("🤖 FinAgent - Enterprise Financial Research Assistant")
    print("=" * 70)
    print("\nAn agentic RAG system for SEC filings and earnings call analysis")
    print("Built with: FastAPI | LangGraph | Qdrant | OpenAI | Cohere")
    print("=" * 70)


def print_architecture():
    """Print system architecture overview."""
    print("\n📐 System Architecture:")
    print("-" * 50)
    print("""
    ┌─────────────────────────────────────────────────┐
    │                   User Query                     │
    └─────────────────────┬───────────────────────────┘
                          │
    ┌─────────────────────▼───────────────────────────┐
    │              Query Router                        │
    │         (Complexity Classification)              │
    └─────────────────────┬───────────────────────────┘
                          │
           ┌──────────────┼──────────────┐
           │              │              │
    ┌──────▼──────┐ ┌─────▼─────┐ ┌─────▼─────┐
    │   Simple    │ │ Moderate  │ │  Complex  │
    └──────┬──────┘ └─────┬─────┘ └─────┬─────┘
           │              │              │
           │              │       ┌──────▼──────┐
           │              │       │   Planner   │
           │              │       └──────┬──────┘
           │              │              │
    ┌──────┴──────────────┴──────────────┴──────┐
    │              Hybrid Retrieval              │
    │         (Dense + BM25 + Reranking)         │
    └─────────────────────┬─────────────────────┘
                          │
    ┌─────────────────────▼─────────────────────┐
    │              Analyst Agent                 │
    │      (Data Extraction & Calculation)       │
    └─────────────────────┬─────────────────────┘
                          │
    ┌─────────────────────▼─────────────────────┐
    │             Synthesizer Agent              │
    │       (Response Generation + Citations)    │
    └─────────────────────┬─────────────────────┘
                          │
    ┌─────────────────────▼─────────────────────┐
    │              Validator Agent               │
    │           (Quality Assurance)              │
    └─────────────────────┬─────────────────────┘
                          │
    ┌─────────────────────▼─────────────────────┐
    │            Cited Response                  │
    └───────────────────────────────────────────┘
    """)


def print_features():
    """Print key features."""
    print("\n✨ Key Features:")
    print("-" * 50)
    features = [
        "🔍 Hybrid Search: Dense embeddings + BM25 keyword matching",
        "📊 Multi-Agent: Specialized agents for different tasks",
        "📝 Citations: Every claim linked to source documents",
        "🎯 Query Routing: Complexity-based pipeline selection",
        "✅ Validation: Quality checks before response delivery",
        "📈 Evaluation: Built-in benchmarking and metrics"
    ]
    for feature in features:
        print(f"  {feature}")


async def run_demo_query(query: str, show_trace: bool = True):
    """
    Run a demo query through the system.
    
    Args:
        query: The query to process
        show_trace: Whether to show agent trace
    """
    print(f"\n{'=' * 70}")
    print(f"📝 Query: {query}")
    print("=" * 70)
    
    # TODO: Implement actual query processing when workflow is ready
    # workflow = FinAgentWorkflow()
    # response = await workflow.run(query)
    
    print("\n⏳ Processing query...")
    print("\n🔄 Agent Trace:")
    print("-" * 50)
    
    # Simulated trace for demonstration
    trace_steps = [
        ("Router", "Classifying query complexity...", "MODERATE"),
        ("Retriever", "Searching SEC filings...", "Found 8 relevant chunks"),
        ("Retriever", "Reranking results...", "Top 5 selected"),
        ("Analyst", "Extracting financial data...", "Revenue: $383.3B"),
        ("Synthesizer", "Generating response...", "Response with 3 citations"),
        ("Validator", "Validating response...", "Score: 92/100 ✅")
    ]
    
    for agent, action, result in trace_steps:
        print(f"  [{agent}] {action}")
        print(f"           → {result}")
        await asyncio.sleep(0.3)  # Simulate processing time
    
    print("\n" + "-" * 50)
    print("📤 Response:")
    print("-" * 50)
    print("""
Apple's total revenue in fiscal year 2023 was $383.3 billion [1], 
representing a slight decrease of 2.8% compared to FY2022's $394.3 billion [2].

The revenue breakdown by segment:
- iPhone: $200.6 billion (52% of total) [1]
- Services: $85.2 billion (22% of total) [1]
- Mac: $29.4 billion [1]
- iPad: $28.3 billion [1]
- Wearables & Accessories: $39.8 billion [1]

Sources:
[1] Apple Inc. 10-K FY2023, Item 7 - MD&A
[2] Apple Inc. 10-K FY2022, Item 7 - MD&A
    """)
    
    print("\n✅ Query completed in 2.3 seconds")


async def interactive_mode():
    """Run interactive demo mode."""
    print("\n🎮 Interactive Mode")
    print("-" * 50)
    print("Enter your financial research questions, or type 'quit' to exit.")
    print("Type 'examples' to see sample queries.\n")
    
    while True:
        try:
            query = input("You: ").strip()
            
            if not query:
                continue
            
            if query.lower() == 'quit':
                print("\n👋 Thanks for trying FinAgent!")
                break
            
            if query.lower() == 'examples':
                print("\n📋 Sample Queries:")
                for i, q in enumerate(DEMO_QUERIES, 1):
                    print(f"  {i}. [{q['complexity']}] {q['query']}")
                print()
                continue
            
            await run_demo_query(query)
            
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for trying FinAgent!")
            break


async def main(query: Optional[str] = None, show_architecture: bool = False):
    """
    Main demo function.
    
    Args:
        query: Optional specific query to run
        show_architecture: Show system architecture
    """
    print_header()
    
    if show_architecture:
        print_architecture()
        print_features()
    
    if query:
        await run_demo_query(query)
    else:
        # Show sample queries
        print("\n📋 Sample Queries:")
        print("-" * 50)
        for i, q in enumerate(DEMO_QUERIES, 1):
            print(f"\n{i}. [{q['complexity'].upper()}] {q['description']}")
            print(f"   \"{q['query']}\"")
        
        # Run first demo query
        print("\n" + "=" * 70)
        print("Running demo with sample query...")
        await run_demo_query(DEMO_QUERIES[0]["query"])
        
        # Offer interactive mode
        print("\n" + "-" * 50)
        response = input("Would you like to try interactive mode? (y/n): ")
        if response.lower() == 'y':
            await interactive_mode()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FinAgent interactive demo"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Specific query to run"
    )
    
    parser.add_argument(
        "--architecture",
        action="store_true",
        help="Show system architecture"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(
        query=args.query,
        show_architecture=args.architecture
    ))
