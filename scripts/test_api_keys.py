#!/usr/bin/env python3
"""
Test API Keys and Basic Components

Tests that your OpenAI and Cohere API keys work correctly
without requiring the full infrastructure.
"""

import sys
import asyncio
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.config import settings
from app.agents.router import QueryRouter
from app.agents.validator import Validator


async def test_openai_connection():
    """Test OpenAI API connection."""
    print("\n" + "="*60)
    print("TEST 1: OpenAI API Connection")
    print("="*60)
    
    try:
        import openai
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Test simple completion
        response = await client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": "Say 'API test successful'"}],
            max_tokens=10
        )
        
        print(f"✅ OpenAI API working!")
        print(f"   Response: {response.choices[0].message.content}")
        
        # Test embeddings
        embedding_response = await client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input="Apple's revenue was $387.8 billion"
        )
        
        print(f"✅ Embeddings working!")
        print(f"   Dimensions: {len(embedding_response.data[0].embedding)}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API failed: {e}")
        return False


async def test_cohere_connection():
    """Test Cohere API connection."""
    print("\n" + "="*60)
    print("TEST 2: Cohere API Connection")
    print("="*60)
    
    try:
        import cohere
        
        client = cohere.AsyncClient(api_key=settings.COHERE_API_KEY)
        
        # Test reranking
        response = await client.rerank(
            model="rerank-english-v3.0",
            query="What is Apple's revenue?",
            documents=[
                "Apple reported revenue of $387.8 billion in 2024.",
                "Microsoft's revenue was $211.9 billion.",
                "Google's revenue increased by 13%."
            ],
            top_n=3
        )
        
        print(f"✅ Cohere API working!")
        print(f"   Reranked {len(response.results)} documents")
        for i, result in enumerate(response.results):
            print(f"   {i+1}. Score: {result.relevance_score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cohere API failed: {e}")
        return False


async def test_router_with_llm():
    """Test Router agent with actual LLM."""
    print("\n" + "="*60)
    print("TEST 3: Router Agent with LLM")
    print("="*60)
    
    try:
        router = QueryRouter(use_llm=True)  # Use LLM this time
        
        test_queries = [
            "What is Apple's revenue?",
            "Compare Apple and Microsoft margins",
            "Analyze the trend in Apple's revenue over the past 3 years"
        ]
        
        print("\nClassifying with LLM:")
        for query in test_queries:
            complexity = await router.classify(query)
            print(f"  '{query[:40]}...' → {complexity.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Router LLM test failed: {e}")
        return False


async def test_validator_with_real_data():
    """Test Validator with more realistic data."""
    print("\n" + "="*60)
    print("TEST 4: Validator with Realistic Data")
    print("="*60)
    
    try:
        from app.models import DocumentChunk, DocumentMetadata, DocumentType, RetrievedDocument, Citation
        from datetime import datetime
        
        validator = Validator()
        
        # Create realistic test data
        chunk = DocumentChunk(
            chunk_id="test_1",
            document_id="aapl_10k_2024",
            content="Apple Inc. reported total net sales of $387.8 billion for fiscal year 2024, representing a 2% increase year-over-year. The company's gross margin was 48.3%, up from 45.7% in the prior year. Operating margin expanded to 33.5%.",
            metadata=DocumentMetadata(
                ticker="AAPL",
                company_name="Apple Inc.",
                document_type=DocumentType.SEC_10K,
                filing_date=datetime(2024, 11, 1),
                source_url="https://sec.gov/test"
            ),
            section="item_7",
            chunk_index=0,
            embedding=None
        )
        
        # Test 1: Perfect response
        print("\n1. Testing PERFECT response:")
        perfect_response = "Apple reported total net sales of $387.8 billion for fiscal year 2024 [1]. The company's gross margin was 48.3% [1]."
        
        perfect_citations = [
            Citation(
                citation_id="1",
                claim="Apple reported total net sales of $387.8 billion for fiscal year 2024",
                source_chunk_id="test_1",
                source_text=chunk.content,
                confidence=0.95,
                page_reference="Item 7"
            )
        ]
        
        result = await validator.validate(
            query="What is Apple's revenue and margin?",
            response=perfect_response,
            citations=perfect_citations,
            documents=[RetrievedDocument(chunk=chunk, score=0.95, retrieval_method="hybrid")]
        )
        
        print(f"   Score: {result.score:.1f}/100")
        print(f"   Valid: {result.is_valid}")
        print(f"   Issues: {len(result.issues)}")
        if result.issues:
            for issue in result.issues[:3]:
                print(f"     - {issue}")
        
        # Test 2: Hallucinated numbers
        print("\n2. Testing HALLUCINATED numbers:")
        hallucinated_response = "Apple reported revenue of $500 billion in 2024 [1]."
        
        result_hallucinated = await validator.validate(
            query="What is Apple's revenue?",
            response=hallucinated_response,
            citations=perfect_citations,
            documents=[RetrievedDocument(chunk=chunk, score=0.95, retrieval_method="hybrid")]
        )
        
        print(f"   Score: {result_hallucinated.score:.1f}/100")
        print(f"   Valid: {result_hallucinated.is_valid}")
        print(f"   ✓ Hallucination caught: {'YES' if not result_hallucinated.is_valid else 'NO'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Validator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_environment_variables():
    """Test that all required environment variables are set."""
    print("\n" + "="*60)
    print("TEST 5: Environment Variables Check")
    print("="*60)
    
    required_vars = [
        "OPENAI_API_KEY",
        "COHERE_API_KEY",
        "QDRANT_HOST",
        "LLM_MODEL",
        "EMBEDDING_MODEL",
    ]
    
    missing_vars = []
    for var in required_vars:
        value = getattr(settings, var, None)
        if not value or value == "":
            if var == "QDRANT_HOST":  # This can be localhost
                if value == "localhost":
                    print(f"  ✓ {var}: {value}")
                    continue
            missing_vars.append(var)
        else:
            # Mask sensitive values
            display_value = value[:10] + "..." if "KEY" in var else value
            print(f"  ✓ {var}: {display_value}")
    
    if missing_vars:
        print(f"\n❌ Missing environment variables: {missing_vars}")
        return False
    
    print(f"\n✅ All required environment variables set!")
    return True


async def main():
    """Run all API tests."""
    print("\n" + "="*60)
    print("🧪 FinAgent API Keys & Components Test")
    print("="*60)
    print("\nTesting your API keys and basic components...")
    
    results = {}
    
    # Run tests
    results["env_vars"] = await test_environment_variables()
    results["openai"] = await test_openai_connection()
    results["cohere"] = await test_cohere_connection()
    results["router"] = await test_router_with_llm()
    results["validator"] = await test_validator_with_real_data()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    print("\nComponent Status:")
    for component, passed in results.items():
        icon = "✅" if passed else "❌"
        name = component.replace("_", " ").title()
        print(f"  {icon} {name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All API tests passed!")
        print("\n✅ Your API keys are working correctly!")
        print("✅ Core components are functional!")
        print("\n🚀 Ready for full workflow testing!")
        print("\nNext steps:")
        print("  1. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant")
        print("  2. Ingest data: python scripts/ingest_filings.py")
        print("  3. Run workflow: python scripts/test_full_workflow.py")
    else:
        print("\n⚠️  Some tests failed")
        print("Check the errors above and fix your configuration")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
