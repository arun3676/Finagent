# FinAgent 🤖💰

**Enterprise-Grade Agentic RAG System for Financial Research**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.40+-purple.svg)](https://github.com/langchain-ai/langgraph)
[![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org/)
[![Tests](https://img.shields.io/badge/Tests-56+-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

FinAgent is a production-ready multi-agent system that answers complex financial research queries using SEC filings with **compliance-grade citations**. It combines hybrid retrieval (BM25 + dense embeddings), LLM-powered analysis, and a self-correcting validator agent to prevent hallucinations.

## 🎯 Key Features

### 🤖 Multi-Agent Workflow
- **Router Agent**: Classifies query complexity (simple/moderate/complex)
- **Planner Agent**: Decomposes complex queries into sub-tasks
- **Retriever Agent**: Hybrid search (BM25 + dense embeddings + reranking)
- **Analyst Agent**: Extracts data and performs calculations
- **Synthesizer Agent**: Generates responses with citations
- **Validator Agent**: Detects hallucinations and ensures factual accuracy

### 🔍 Advanced Retrieval
- **Hybrid Search**: BM25 sparse + OpenAI dense embeddings (text-embedding-3-large)
- **Reciprocal Rank Fusion**: Optimal combination of retrieval methods
- **Cohere Reranking**: Cross-encoder for precision optimization
- **Document-Aware Chunking**: Preserves SEC 10-K section boundaries

### 🛡️ Hallucination Prevention
- **Validator Agent**: Self-correcting with up to 3 validation attempts
- **Factual Accuracy**: Claim-source similarity >0.8 required
- **Numerical Accuracy**: All numbers extracted from sources, not generated
- **Citation Coverage**: >95% of claims must have supporting evidence

### 📊 Production Ready
- **Compliance-Grade Citations**: Every claim traces to exact source paragraph
- **Real-Time Streaming**: Server-Sent Events (SSE) for progressive responses
- **Modern Frontend**: Next.js 14 with TypeScript and Tailwind CSS
- **Cross-Platform**: Works on Windows, Linux, and macOS
- **Type-Safe**: Full type hints with Pydantic v2
- **Tiered LLM Routing**: FAST/STANDARD/COMPLEX model selection for cost and latency
- **Multi-Provider LLMs**: OpenAI, Gemini, and Anthropic support behind a unified client
- **Comprehensive Testing**: 56+ tests with 100% pass rate
- **Error Recovery**: Circuit breaker pattern with automatic retry

## 🏗️ Architecture

```
User Query → Router → [Planner] → Retriever → Analyst → Synthesizer → Validator → Response
                         │            │
                    (if complex)  Hybrid Search
                                 + Reranking
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (for Qdrant)
- API Keys: OpenAI, Cohere, Google (Gemini), Anthropic

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
cp .env.local.example .env.local
# Edit .env.local if needed (default: http://localhost:8010)
npm run dev
```

### Try It Out

```bash
# Backend health check
curl http://localhost:8010/health

# Frontend access
# Open http://localhost:3000 in your browser

# Run backend tests
cd backend && python -m pytest tests/ -v
```

## 📁 Project Structure

```
finagent/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point with SSE streaming
│   │   ├── config.py            # Configuration
│   │   ├── models.py            # Pydantic models
│   │   ├── ingestion/           # Document loaders
│   │   ├── chunking/            # Document chunkers
│   │   ├── retrieval/           # Search components
│   │   ├── agents/              # Multi-agent system
│   │   ├── tools/               # Agent tools
│   │   ├── citations/           # Citation system
│   │   └── evaluation/          # Metrics & benchmarks
│   ├── tests/                   # Test suite (56+ tests)
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app/                     # Next.js app router
│   ├── components/              # React components
│   │   └── chat/                # Chat interface with streaming
│   ├── lib/                     # API client and utilities
│   ├── types/                   # TypeScript types
│   ├── package.json
│   └── Dockerfile
├── scripts/
│   ├── ingest_filings.py        # Batch ingestion
│   ├── run_evaluation.py        # Run benchmarks
│   └── qdrant_smoke_check.py    # Verify ingestion
├── docs/
│   ├── architecture.md
│   ├── api.md
│   └── deployment.md
├── .claude/                     # Claude skills
│   └── skills/                  # Development, testing, debugging
└── docker-compose.yml
```

## 🔧 Configuration

Key environment variables:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings and LLM |
| `COHERE_API_KEY` | Cohere API key for reranking |
| `GOOGLE_API_KEY` | Google API key for Gemini models |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude models |
| `QDRANT_HOST` | Qdrant server host |
| `LLM_MODEL` | LLM model (default: gpt-4-turbo-preview) |
| `LLM_MODEL_FAST` | FAST tier model (default: gemini-1.5-flash) |
| `LLM_MODEL_STANDARD` | STANDARD tier model (default: gpt-4o-mini) |
| `LLM_MODEL_COMPLEX` | COMPLEX tier model (default: claude-3-5-sonnet-20241022) |

See `backend/.env.example` for all options.

## 📊 Evaluation

Run benchmarks:

```bash
# Run full evaluation suite
cd backend && python scripts/run_evaluation.py

# Run sample evaluation
cd backend && python scripts/run_evaluation.py --sample
```

Metrics tracked:
- **Retrieval**: Recall@K, Precision@K, MRR, NDCG
- **Generation**: Answer similarity, Faithfulness, Citation precision
- **End-to-End**: Query latency, Agent success rates, Validation passes

### Test Coverage (2026-01-15)
- **Total Tests**: 56+ with 100% pass rate
- **E2E Flow Tests**: 32 tests covering router, retriever, validator, workflow
- **Comprehensive Tests**: 24 tests covering models, SSE events, contracts
- **Coverage Areas**: Query classification, ticker extraction, citation validation, workflow routing, error recovery

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| API Framework | FastAPI |
| Agent Orchestration | LangGraph |
| Vector Database | Qdrant |
| Embeddings | OpenAI text-embedding-3-large |
| LLM | OpenAI, Gemini, Anthropic |
| Reranking | Cohere |
| Frontend | Next.js 14, TypeScript, Tailwind CSS |
| Streaming | Server-Sent Events (SSE) |
| Testing | Pytest, Jest, React Testing Library |

## 📈 Implementation Status

- [x] **Core Infrastructure**: SEC EDGAR loader, document chunking, data models
- [x] **Retrieval System**: Hybrid search (BM25 + dense), Qdrant vector store, Cohere reranking
- [x] **Multi-Agent Workflow**: Complete LangGraph implementation with all 6 agents
- [x] **Validator Agent**: Hallucination detection with factual accuracy checks
- [x] **Citations Engine**: Automatic claim extraction and source linking
- [x] **Evaluation Framework**: Comprehensive metrics and test datasets
- [x] **Testing Suite**: 56+ E2E tests with 100% pass rate
- [x] **Error Recovery**: Circuit breaker pattern, automatic retry, graceful degradation
- [x] **Frontend Integration**: Next.js UI with real-time streaming responses
- [x] **SSE Streaming**: Progressive query responses with step indicators
- [ ] **Production Deployment**: Docker Compose, CI/CD, monitoring (In Progress)

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 🔍 Development Tools

### Claude Skills
The project includes Claude skills for enhanced development experience:
- **finagent-dev**: Development workflow and architecture guidance
- **finagent-test**: Test runner and validation commands
- **finagent-debug**: Troubleshooting and diagnostic tools

### Quick Commands
```bash
# Run all tests
cd backend && python -m pytest tests/ -v

# Debug specific components
cd backend && python -c "from app.agents.router import QueryRouter; print('Router OK')"

# Verify ingestion
python scripts/qdrant_smoke_check.py --require AAPL
```

## 📄 License

MIT License - see LICENSE file for details.


---

