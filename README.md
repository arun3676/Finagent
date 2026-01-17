# FinAgent

**Enterprise-Grade Agentic RAG System for Financial Research**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.40+-purple.svg)](https://github.com/langchain-ai/langgraph)
[![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org/)
[![Tests](https://img.shields.io/badge/Tests-75+-brightgreen.svg)](tests/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue.svg)](.github/workflows/test.yml)

FinAgent is a production-ready multi-agent system that answers complex financial research queries using SEC filings with **compliance-grade citations**. It combines **Hybrid Retrieval (BM25 + Dense Embeddings) with Reciprocal Rank Fusion (RRF)**, LLM-powered analysis, and a self-correcting validator agent to prevent hallucinations.

---

## Key Differentiators

| Feature | Implementation |
|---------|---------------|
| **Hybrid Retrieval** | BM25 sparse + OpenAI dense embeddings combined via **Reciprocal Rank Fusion (RRF)** |
| **Self-Correcting Agents** | Validator detects hallucinations and loops back to Retriever (up to 3 attempts) |
| **Tiered Model Selection** | Cost-optimized routing: Gemini Flash for simple tasks, GPT-4o for complex reasoning |
| **Compliance-Grade Citations** | Every claim traces to exact source paragraph with confidence scores |
| **Observability** | LangSmith tracing for production debugging and performance monitoring |
| **Evaluation** | DeepEval integration for Faithfulness, Answer Relevancy, and Contextual Precision metrics |

---

## Architecture

### Multi-Agent Workflow (LangGraph)

```
Query → Router → [Planner] → Retriever → [Analyst] → Synthesizer → Validator → Response
           │          │           │            │            │            │
      Complexity   (COMPLEX    Hybrid      (Tools:       Citations    Self-correction
      Detection     only)     Search +    Calculator,    + Draft      Loop (max 3)
                              RRF         PriceLookup)
```

### Hybrid Retrieval with Reciprocal Rank Fusion (RRF)

```
Query → Embed → [Dense Search (Qdrant)] + [Sparse Search (BM25)]
                       ↓                          ↓
               [Top-K results]          [Top-K results]
                       ↓                          ↓
                   [Reciprocal Rank Fusion (k=60)]
                              ↓
                       [Merged Top-K]
                              ↓
                       [Cohere Reranking]
                              ↓
                       [Final Results]
```

**RRF Formula**: `score = Σ 1/(k + rank)` where k=60

This approach combines the precision of semantic search with the recall of keyword matching, outperforming either method alone.

---

## Self-Correcting Agent Behavior (Reflection Pattern)

The Validator Agent implements a **reflection loop** that catches hallucinations and triggers self-correction:

```json
{
  "trace": [
    {
      "agent": "synthesizer",
      "action": "generate_response",
      "output": "Apple's revenue was $394B in 2023, a 15% increase...",
      "citations": ["cite_1", "cite_2"]
    },
    {
      "agent": "validator",
      "action": "validate_response",
      "result": "FAILED",
      "confidence": 0.42,
      "reason": "Claim 'a 15% increase' not supported by sources. Sources show 2% decline.",
      "feedback": "Revenue growth claim lacks citation support"
    },
    {
      "agent": "router",
      "action": "loop_back",
      "decision": "Returning to Retriever (Attempt 2/3)",
      "reason": "Validation failed, fetching additional sources"
    },
    {
      "agent": "retriever",
      "action": "fetch_additional",
      "documents_added": 3
    },
    {
      "agent": "synthesizer",
      "action": "regenerate_response",
      "output": "Apple's revenue was $383B in 2023, a 2% decrease from $394B in 2022...",
      "citations": ["cite_1", "cite_2", "cite_3", "cite_4"]
    },
    {
      "agent": "validator",
      "action": "validate_response",
      "result": "PASSED",
      "confidence": 0.89,
      "breakdown": {
        "factual_accuracy": 0.92,
        "numerical_accuracy": 0.95,
        "citation_coverage": 0.88,
        "completeness": 0.81
      }
    }
  ]
}
```

**Key Insight**: The system caught a factual error (15% increase vs 2% decline), fetched additional sources, and regenerated a correct response with proper citations.

---

## Cost vs. Performance Trade-offs

FinAgent uses **tiered model selection** to balance cost and accuracy:

| Path | Model | Cost/Query | Latency | Use Case |
|------|-------|------------|---------|----------|
| **Fast Path** | Gemini 2.0 Flash Lite | ~$0.0001 | <1s | Simple queries, classification |
| **Standard Path** | GPT-4o-mini | ~$0.002 | 2-3s | Moderate analysis, most queries |
| **Reasoning Path** | GPT-4o | ~$0.01 | 3-5s | Complex multi-step reasoning |

**Routing Logic**:
- SIMPLE queries (single entity, direct lookup) → Fast Path
- MODERATE queries (analysis, comparisons) → Standard Path
- COMPLEX queries (multi-year trends, cross-company analysis) → Reasoning Path

---

## Key Features

### Multi-Agent Workflow
- **Router Agent**: Classifies query complexity (simple/moderate/complex)
- **Planner Agent**: Decomposes complex queries into sub-tasks
- **Retriever Agent**: Hybrid search (BM25 + dense embeddings + RRF + reranking)
- **Analyst Agent**: Extracts data and performs calculations
- **Synthesizer Agent**: Generates responses with citations
- **Validator Agent**: Detects hallucinations and triggers self-correction

### Interactive Chat Features
- **Follow-up Questions**: Automatically generates 3 contextual questions after each response
  - Temporal: How has X changed over time?
  - Deeper: What factors/reasons/details about X?
  - Comparative: How does X compare to peers?
- **Response Length Control**: Choose between SHORT, NORMAL, or DETAILED responses
- **Real-Time Streaming**: Watch agents work with live step indicators

### Hallucination Prevention
- **Validator Agent**: Self-correcting with up to 3 validation attempts
- **Factual Accuracy**: Claim-source similarity >0.8 required
- **Numerical Accuracy**: All numbers extracted from sources, not generated
- **Citation Coverage**: >95% of claims must have supporting evidence

### Production Ready
- **Compliance-Grade Citations**: Every claim traces to exact source paragraph
- **Real-Time Streaming**: Server-Sent Events (SSE) for progressive responses
- **LangSmith Observability**: Production tracing and debugging
- **DeepEval Metrics**: Industry-standard RAG evaluation
- **CI/CD Pipeline**: GitHub Actions with automated testing
- **Type-Safe**: Full type hints with Pydantic v2
- **Comprehensive Testing**: 75+ tests with 100% pass rate

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (for Qdrant)
- API Keys: OpenAI, Cohere (optional: Google, LangSmith)

### Installation

```bash
# Clone the repository
git clone https://github.com/arun3676/finagent.git
cd finagent

# Start Qdrant vector database
docker run -p 6333:6333 qdrant/qdrant

# Setup backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run the backend server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8010

# In a new terminal, setup and run frontend
cd ../frontend
npm install
npm run dev
```

### Try It Out

```bash
# Backend health check
curl http://localhost:8010/health

# Frontend access
# Open http://localhost:3000 in your browser

# Run tests
cd backend && pytest tests/ -v
```

---

## Evaluation (DeepEval)

FinAgent uses **DeepEval** for industry-standard RAG evaluation metrics:

```bash
# Run evaluation suite
cd backend && pytest tests/evaluation/ -v

# Or use DeepEval CLI
deepeval test run tests/evaluation/
```

### Metrics

| Metric | Description | Threshold |
|--------|-------------|-----------|
| **Faithfulness** | Are claims supported by retrieved contexts? | 0.7 |
| **Answer Relevancy** | Does the answer address the question? | 0.7 |
| **Contextual Precision** | Are retrieved docs relevant and well-ranked? | 0.7 |
| **Contextual Recall** | Do contexts contain all needed information? | 0.7 |
| **Hallucination** | Percentage of unsupported claims (lower is better) | <0.3 |

### Running Evaluations

```python
from app.evaluation.metrics import GenerationMetrics

metrics = GenerationMetrics(model="gpt-4o-mini", threshold=0.7)

result = await metrics.evaluate_response(
    question="What was Apple's revenue in 2023?",
    answer="Apple's revenue was $383B in 2023...",
    contexts=["Apple reported revenue of $383 billion..."],
    expected_output="Apple's 2023 revenue was $383 billion"
)

print(result)
# {
#   "metrics": {
#     "faithfulness": {"score": 0.95, "passed": True},
#     "answer_relevance": {"score": 0.88, "passed": True},
#     "contextual_precision": {"score": 0.82, "passed": True}
#   },
#   "overall_score": 0.88,
#   "all_passed": True
# }
```

---

## Observability (LangSmith)

Enable LangSmith tracing for production debugging:

```bash
# In .env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=finagent
```

Features:
- **Trace Visualization**: See complete agent execution flow
- **Latency Analysis**: Identify slow components
- **Error Tracking**: Debug failures with full context
- **Cost Monitoring**: Track LLM API spend per query

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| API Framework | FastAPI |
| Agent Orchestration | LangGraph |
| Vector Database | Qdrant |
| Embeddings | OpenAI text-embedding-3-large |
| LLM (Tiered) | Gemini Flash / GPT-4o-mini / GPT-4o |
| Reranking | Cohere |
| Retrieval | Hybrid (BM25 + Dense + RRF) |
| Observability | LangSmith |
| Evaluation | DeepEval |
| Frontend | Next.js 14, TypeScript, Tailwind CSS |
| Streaming | Server-Sent Events (SSE) |
| CI/CD | GitHub Actions |
| Testing | Pytest, Jest |

---

## Configuration

Key environment variables:

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings and LLM | Yes |
| `GOOGLE_API_KEY` | Google API key for Gemini models | Yes (for fast path) |
| `COHERE_API_KEY` | Cohere API key for reranking | Optional |
| `QDRANT_HOST` | Qdrant server host | Yes |
| `LANGCHAIN_API_KEY` | LangSmith API key for tracing | Optional |
| `DEEPEVAL_API_KEY` | DeepEval API key for evaluation dashboard | Optional |

See `backend/.env.example` for all options.

---

## Limitations & Roadmap

### Current Limitations

| Limitation | Current State | Planned Solution |
|------------|---------------|------------------|
| **PDF Processing** | Digital-native 10-Ks only | Multimodal LLM (GPT-4o Vision) for scanned PDFs |
| **Real-time Data** | SEC filings (quarterly lag) | Integration with live market data APIs |
| **Multi-language** | English only | Multilingual embeddings and prompts |
| **Document Types** | 10-K, 10-Q filings | Expand to 8-K, proxy statements, earnings calls |

### Honest Notes

> **Multimodal vs OCR**: The current pipeline handles **digital-native 10-Ks** (searchable PDFs with embedded text). For scanned PDFs or image-heavy documents, we're planning a **Multimodal LLM pipeline** (e.g., GPT-4o Vision) that can directly interpret images and tables, with **Tesseract OCR** as a fallback for high-volume simple forms. This hybrid approach balances accuracy with cost.

---

## Project Structure

```
finagent/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point with SSE streaming
│   │   ├── config.py            # Configuration with LangSmith settings
│   │   ├── models.py            # Pydantic models
│   │   ├── agents/              # Multi-agent system (LangGraph)
│   │   ├── retrieval/           # Hybrid search (BM25 + Dense + RRF)
│   │   ├── followup/            # Follow-up question generation
│   │   ├── evaluation/          # DeepEval metrics integration
│   │   └── llm/                 # Tiered model selection
│   ├── tests/                   # Test suite (75+ tests)
│   └── requirements.txt
├── frontend/
│   ├── app/                     # Next.js app router
│   ├── components/chat/         # Chat interface with streaming
│   └── lib/                     # API client and utilities
├── .github/workflows/
│   └── test.yml                 # CI/CD pipeline
└── docker-compose.yml
```

---

## Testing

```bash
# Run all backend tests
cd backend && pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run specific test suites
pytest tests/test_e2e_flow.py -v        # E2E tests (32)
pytest tests/test_followup.py -v        # Follow-up tests (19)
pytest tests/evaluation/ -v              # DeepEval metrics
```

### Test Coverage Summary

| Suite | Tests | Coverage |
|-------|-------|----------|
| E2E Flow | 32 | Router, Retriever, Validator, Workflow |
| Comprehensive | 24 | Models, SSE Events, API Contracts |
| Follow-up | 19 | Cache, Generator, Executor |
| Evaluation | 10+ | DeepEval metrics integration |

---

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Tools

```bash
# Format code
black app/ tests/
isort app/ tests/

# Type checking
mypy app/

# Lint
ruff check app/
```

---

## CI/CD

The project uses GitHub Actions for continuous integration:

- **On Push to main**: Run all tests, linting, type checking
- **On Pull Request**: Run tests + build validation
- **Evaluation Tests**: Run DeepEval metrics (if API keys present)

See [.github/workflows/test.yml](.github/workflows/test.yml) for details.
