#!/usr/bin/env python3
"""
FinAgent Flow Verification Script

This script verifies the complete end-to-end flow of the FinAgent system.
It tests each component and the full pipeline.

Usage:
    python tests/verify_flow.py                    # Run all tests
    python tests/verify_flow.py --quick            # Run quick tests only
    python tests/verify_flow.py --component router # Test specific component
    python tests/verify_flow.py --live             # Test against live server

Requirements:
    - Qdrant running on localhost:6333
    - Backend server running on localhost:8010 (for --live tests)
    - OPENAI_API_KEY set in environment
"""

import os
import sys
import time
import json
import asyncio
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    WARN = "WARN"


@dataclass
class TestResult:
    name: str
    status: TestStatus
    message: str
    duration_ms: int


class FlowVerifier:
    """Verifies the FinAgent flow end-to-end."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[TestResult] = []

    def log(self, message: str):
        """Log a message."""
        if self.verbose:
            print(f"  {message}")

    def add_result(self, name: str, status: TestStatus, message: str, duration_ms: int = 0):
        """Add a test result."""
        result = TestResult(name, status, message, duration_ms)
        self.results.append(result)

        # Print result
        icon = {"PASS": "[OK]", "FAIL": "[XX]", "SKIP": "[--]", "WARN": "[!!]"}[status.value]
        time_str = f" ({duration_ms}ms)" if duration_ms > 0 else ""
        print(f"{icon} {name}: {message}{time_str}")

    # =========================================================================
    # Environment Tests
    # =========================================================================

    def test_environment(self) -> bool:
        """Test environment configuration."""
        print("\n=== Environment Tests ===")
        all_pass = True

        # Check Python version
        start = time.time()
        py_version = sys.version_info
        if py_version >= (3, 11):
            self.add_result("Python version", TestStatus.PASS, f"Python {py_version.major}.{py_version.minor}")
        else:
            self.add_result("Python version", TestStatus.WARN, f"Python {py_version.major}.{py_version.minor} (3.11+ recommended)")

        # Check OpenAI API key
        start = time.time()
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if openai_key and openai_key.startswith("sk-"):
            self.add_result("OpenAI API Key", TestStatus.PASS, "Configured")
        else:
            self.add_result("OpenAI API Key", TestStatus.FAIL, "Not configured or invalid")
            all_pass = False

        # Check Qdrant connection
        start = time.time()
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(url="http://localhost:6333", timeout=5)
            collections = client.get_collections()
            duration = int((time.time() - start) * 1000)
            self.add_result("Qdrant connection", TestStatus.PASS, f"Connected, {len(collections.collections)} collections", duration)
        except Exception as e:
            duration = int((time.time() - start) * 1000)
            self.add_result("Qdrant connection", TestStatus.FAIL, str(e), duration)
            all_pass = False

        return all_pass

    # =========================================================================
    # Component Tests
    # =========================================================================

    async def test_router(self) -> bool:
        """Test router agent."""
        print("\n=== Router Agent Tests ===")
        all_pass = True

        try:
            from app.agents.router import QueryRouter
            from app.models import QueryComplexity

            router = QueryRouter(use_llm=False)

            # Test simple query
            start = time.time()
            result = router._heuristic_classify("What is Apple's revenue?")
            duration = int((time.time() - start) * 1000)
            if result == QueryComplexity.SIMPLE:
                self.add_result("Router: Simple query", TestStatus.PASS, "Classified as SIMPLE", duration)
            else:
                self.add_result("Router: Simple query", TestStatus.FAIL, f"Got {result}, expected SIMPLE", duration)
                all_pass = False

            # Test complex query
            start = time.time()
            result = router._heuristic_classify("Analyze the risk factors and their impact on revenue growth strategy")
            duration = int((time.time() - start) * 1000)
            if result == QueryComplexity.COMPLEX:
                self.add_result("Router: Complex query", TestStatus.PASS, "Classified as COMPLEX", duration)
            else:
                self.add_result("Router: Complex query", TestStatus.FAIL, f"Got {result}, expected COMPLEX", duration)
                all_pass = False

            # Test pipeline selection
            start = time.time()
            pipeline = router.get_pipeline_for_complexity(QueryComplexity.COMPLEX)
            duration = int((time.time() - start) * 1000)
            if "planner" in pipeline and "validator" in pipeline:
                self.add_result("Router: Pipeline selection", TestStatus.PASS, f"Pipeline: {pipeline}", duration)
            else:
                self.add_result("Router: Pipeline selection", TestStatus.FAIL, f"Missing agents in pipeline", duration)
                all_pass = False

        except Exception as e:
            self.add_result("Router: Import/Init", TestStatus.FAIL, str(e))
            all_pass = False

        return all_pass

    async def test_retriever(self) -> bool:
        """Test retriever agent."""
        print("\n=== Retriever Agent Tests ===")
        all_pass = True

        try:
            from app.agents.retriever_agent import RetrieverAgent

            agent = RetrieverAgent()

            # Test ticker extraction
            test_cases = [
                ("What is $AAPL revenue?", "AAPL"),
                ("Microsoft earnings", "MSFT"),
                ("Tell me about Apple", "AAPL"),
            ]

            for query, expected in test_cases:
                start = time.time()
                result = agent.extract_ticker_from_query(query)
                duration = int((time.time() - start) * 1000)

                if result == expected:
                    self.add_result(f"Retriever: Extract '{query}'", TestStatus.PASS, f"Got {result}", duration)
                else:
                    self.add_result(f"Retriever: Extract '{query}'", TestStatus.FAIL, f"Got {result}, expected {expected}", duration)
                    all_pass = False

            # Test market data query detection
            start = time.time()
            is_market = agent._is_market_data_query("What is the current price of AAPL?")
            duration = int((time.time() - start) * 1000)
            if is_market:
                self.add_result("Retriever: Market query detection", TestStatus.PASS, "Detected market query", duration)
            else:
                self.add_result("Retriever: Market query detection", TestStatus.FAIL, "Failed to detect", duration)
                all_pass = False

        except Exception as e:
            self.add_result("Retriever: Import/Init", TestStatus.FAIL, str(e))
            all_pass = False

        return all_pass

    async def test_validator(self) -> bool:
        """Test validator agent."""
        print("\n=== Validator Agent Tests ===")
        all_pass = True

        try:
            from app.agents.validator import Validator

            validator = Validator()

            # Test citation coverage
            start = time.time()
            response = "Revenue was $100 billion [1]. Growth was 15% [2]."
            citations = [{"citation_id": "1"}, {"citation_id": "2"}]
            score, issues = validator.check_citation_coverage(response, citations)
            duration = int((time.time() - start) * 1000)

            if score >= 90:
                self.add_result("Validator: Citation coverage", TestStatus.PASS, f"Score: {score}%", duration)
            else:
                self.add_result("Validator: Citation coverage", TestStatus.FAIL, f"Score: {score}%", duration)
                all_pass = False

            # Test missing citations
            start = time.time()
            response_no_cites = "Revenue was $100 billion. Growth was 15%."
            score2, issues2 = validator.check_citation_coverage(response_no_cites, [])
            duration = int((time.time() - start) * 1000)

            if score2 < 100 and len(issues2) > 0:
                self.add_result("Validator: Missing citation detection", TestStatus.PASS, f"Found {len(issues2)} issues", duration)
            else:
                self.add_result("Validator: Missing citation detection", TestStatus.FAIL, f"Should detect missing citations", duration)
                all_pass = False

        except Exception as e:
            self.add_result("Validator: Import/Init", TestStatus.FAIL, str(e))
            all_pass = False

        return all_pass

    async def test_workflow(self) -> bool:
        """Test workflow orchestration."""
        print("\n=== Workflow Tests ===")
        all_pass = True

        try:
            from app.agents.workflow import FinAgentWorkflow
            from app.models import AgentState, QueryComplexity

            # Test workflow initialization
            start = time.time()
            workflow = FinAgentWorkflow()
            duration = int((time.time() - start) * 1000)

            if workflow.graph is not None:
                self.add_result("Workflow: Initialization", TestStatus.PASS, "Graph built", duration)
            else:
                self.add_result("Workflow: Initialization", TestStatus.FAIL, "Graph is None", duration)
                all_pass = False

            # Test routing logic
            start = time.time()
            state_simple = AgentState(original_query="test", complexity=QueryComplexity.SIMPLE)
            route = workflow._route_after_classification(state_simple)
            duration = int((time.time() - start) * 1000)

            if route == "retriever":
                self.add_result("Workflow: Simple routing", TestStatus.PASS, f"Routes to {route}", duration)
            else:
                self.add_result("Workflow: Simple routing", TestStatus.FAIL, f"Got {route}, expected retriever", duration)
                all_pass = False

            # Test validation loop logic
            start = time.time()
            state_invalid = AgentState(original_query="test", is_valid=False, iteration_count=1)
            route = workflow._route_after_validation(state_invalid)
            duration = int((time.time() - start) * 1000)

            if route == "retriever":
                self.add_result("Workflow: Validation retry", TestStatus.PASS, f"Loops back to {route}", duration)
            else:
                self.add_result("Workflow: Validation retry", TestStatus.FAIL, f"Got {route}, expected retriever", duration)
                all_pass = False

            # Test max iterations
            start = time.time()
            state_max = AgentState(original_query="test", is_valid=False, iteration_count=3)
            route = workflow._route_after_validation(state_max)
            duration = int((time.time() - start) * 1000)

            if route == "end":
                self.add_result("Workflow: Max iterations", TestStatus.PASS, "Ends at max iterations", duration)
            else:
                self.add_result("Workflow: Max iterations", TestStatus.FAIL, f"Got {route}, expected end", duration)
                all_pass = False

        except Exception as e:
            self.add_result("Workflow: Import/Init", TestStatus.FAIL, str(e))
            all_pass = False

        return all_pass

    async def test_error_recovery(self) -> bool:
        """Test error recovery agent."""
        print("\n=== Error Recovery Tests ===")
        all_pass = True

        try:
            from app.agents.error_recovery import ErrorRecoveryAgent, ErrorType

            agent = ErrorRecoveryAgent()

            # Test error classification
            test_cases = [
                ("Connection timeout", ErrorType.TIMEOUT),
                ("Rate limit exceeded", ErrorType.RATE_LIMIT),
                ("Server error 500", ErrorType.SERVER_ERROR),
            ]

            for error_msg, expected_type in test_cases:
                start = time.time()
                result = agent._classify_error(error_msg)
                duration = int((time.time() - start) * 1000)

                if result == expected_type:
                    self.add_result(f"ErrorRecovery: Classify '{error_msg}'", TestStatus.PASS, f"Got {result.value}", duration)
                else:
                    self.add_result(f"ErrorRecovery: Classify '{error_msg}'", TestStatus.FAIL, f"Got {result}, expected {expected_type}", duration)
                    all_pass = False

            # Test circuit breaker
            start = time.time()
            # Initial state should be closed
            is_open = agent._is_circuit_open(ErrorType.TIMEOUT)
            duration = int((time.time() - start) * 1000)

            if not is_open:
                self.add_result("ErrorRecovery: Circuit breaker (initial)", TestStatus.PASS, "Circuit closed", duration)
            else:
                self.add_result("ErrorRecovery: Circuit breaker (initial)", TestStatus.FAIL, "Circuit should be closed", duration)
                all_pass = False

            # Test health status
            start = time.time()
            status = agent.get_health_status()
            duration = int((time.time() - start) * 1000)

            if status["status"] == "healthy":
                self.add_result("ErrorRecovery: Health status", TestStatus.PASS, "Healthy", duration)
            else:
                self.add_result("ErrorRecovery: Health status", TestStatus.WARN, f"Status: {status['status']}", duration)

        except Exception as e:
            self.add_result("ErrorRecovery: Import/Init", TestStatus.FAIL, str(e))
            all_pass = False

        return all_pass

    # =========================================================================
    # Live Server Tests
    # =========================================================================

    async def test_live_server(self) -> bool:
        """Test against live server."""
        print("\n=== Live Server Tests ===")
        all_pass = True

        import aiohttp

        base_url = "http://localhost:8010"

        # Test ping
        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{base_url}/ping", timeout=5) as response:
                    duration = int((time.time() - start) * 1000)
                    if response.status == 200:
                        data = await response.json()
                        self.add_result("Live: /ping", TestStatus.PASS, f"Status: {data.get('status')}", duration)
                    else:
                        self.add_result("Live: /ping", TestStatus.FAIL, f"Status code: {response.status}", duration)
                        all_pass = False
        except Exception as e:
            duration = int((time.time() - start) * 1000)
            self.add_result("Live: /ping", TestStatus.FAIL, str(e), duration)
            all_pass = False

        # Test health
        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{base_url}/health", timeout=5) as response:
                    duration = int((time.time() - start) * 1000)
                    if response.status == 200:
                        data = await response.json()
                        self.add_result("Live: /health", TestStatus.PASS, f"Status: {data.get('status')}", duration)
                    else:
                        self.add_result("Live: /health", TestStatus.FAIL, f"Status code: {response.status}", duration)
                        all_pass = False
        except Exception as e:
            duration = int((time.time() - start) * 1000)
            self.add_result("Live: /health", TestStatus.FAIL, str(e), duration)
            all_pass = False

        # Test query endpoint
        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                payload = {"query": "What is Apple's revenue?"}
                async with session.post(f"{base_url}/query", json=payload, timeout=60) as response:
                    duration = int((time.time() - start) * 1000)
                    if response.status == 200:
                        data = await response.json()
                        answer_len = len(data.get("answer", ""))
                        self.add_result("Live: /query", TestStatus.PASS, f"Response: {answer_len} chars", duration)
                    else:
                        text = await response.text()
                        self.add_result("Live: /query", TestStatus.FAIL, f"Status: {response.status}, {text[:100]}", duration)
                        all_pass = False
        except asyncio.TimeoutError:
            duration = int((time.time() - start) * 1000)
            self.add_result("Live: /query", TestStatus.WARN, "Timeout (may be normal for first query)", duration)
        except Exception as e:
            duration = int((time.time() - start) * 1000)
            self.add_result("Live: /query", TestStatus.FAIL, str(e), duration)
            all_pass = False

        return all_pass

    # =========================================================================
    # Summary
    # =========================================================================

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        passed = sum(1 for r in self.results if r.status == TestStatus.PASS)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAIL)
        warned = sum(1 for r in self.results if r.status == TestStatus.WARN)
        skipped = sum(1 for r in self.results if r.status == TestStatus.SKIP)

        total = len(self.results)

        print(f"Total: {total} | Passed: {passed} | Failed: {failed} | Warnings: {warned} | Skipped: {skipped}")

        if failed > 0:
            print("\nFailed Tests:")
            for r in self.results:
                if r.status == TestStatus.FAIL:
                    print(f"  - {r.name}: {r.message}")

        if warned > 0:
            print("\nWarnings:")
            for r in self.results:
                if r.status == TestStatus.WARN:
                    print(f"  - {r.name}: {r.message}")

        print("=" * 60)

        return failed == 0


async def main():
    parser = argparse.ArgumentParser(description="FinAgent Flow Verification")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--live", action="store_true", help="Include live server tests")
    parser.add_argument("--component", type=str, help="Test specific component (router, retriever, validator, workflow, error_recovery)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    verifier = FlowVerifier(verbose=args.verbose)

    print("=" * 60)
    print("FinAgent Flow Verification")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Run environment tests
    verifier.test_environment()

    # Run component tests
    if args.component:
        component_tests = {
            "router": verifier.test_router,
            "retriever": verifier.test_retriever,
            "validator": verifier.test_validator,
            "workflow": verifier.test_workflow,
            "error_recovery": verifier.test_error_recovery,
        }

        if args.component in component_tests:
            await component_tests[args.component]()
        else:
            print(f"Unknown component: {args.component}")
            print(f"Available: {list(component_tests.keys())}")
    else:
        await verifier.test_router()
        await verifier.test_retriever()
        await verifier.test_validator()
        await verifier.test_workflow()
        await verifier.test_error_recovery()

    # Run live tests if requested
    if args.live:
        await verifier.test_live_server()

    # Print summary
    success = verifier.print_summary()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
