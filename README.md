# FinAgent 🤖💰

**Enterprise-Grade Agentic RAG System for Financial Research**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.40+-purple.svg)](https://github.com/langchain-ai/langgraph)
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
- **Cross-Platform**: Works on Windows, Linux, and macOS
- **Type-Safe**: Full type hints with Pydantic v2
- **Comprehensive Testing**: Unit tests, integration tests, and validation tools

## 🏗️ Architecture

```
User Query → Router → [Planner] → Retriever → Analyst → Synthesizer → Validator → Response
                         │            │
                    (if complex)  Hybrid Search
                                 + Reranking
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker (for Qdrant)
- API Keys: OpenAI, Cohere

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/finagent.git
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

# Run the server
uvicorn app.main:app --reload
```

### Try It Out

```bash
# Health check
curl http://localhost:8000/health

# Run demo
python scripts/demo.py
```

## 📁 Project Structure

```
finagent/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── config.py            # Configuration
│   │   ├── models.py            # Pydantic models
│   │   ├── ingestion/           # Document loaders
│   │   ├── chunking/            # Document chunkers
│   │   ├── retrieval/           # Search components
│   │   ├── agents/              # Multi-agent system
│   │   ├── tools/               # Agent tools
│   │   ├── citations/           # Citation system
│   │   └── evaluation/          # Metrics & benchmarks
│   ├── tests/                   # Test suite
│   ├── requirements.txt
│   └── Dockerfile
├── scripts/
│   ├── ingest_filings.py        # Batch ingestion
│   ├── run_evaluation.py        # Run benchmarks
│   └── demo.py                  # Interactive demo
├── docs/
│   ├── architecture.md
│   ├── api.md
│   └── deployment.md
└── docker-compose.yml
```

## 🔧 Configuration

Key environment variables:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings and LLM |
| `COHERE_API_KEY` | Cohere API key for reranking |
| `QDRANT_HOST` | Qdrant server host |
| `LLM_MODEL` | LLM model (default: gpt-4-turbo-preview) |

See `backend/.env.example` for all options.

## 📊 Evaluation

Run benchmarks:

```bash
python scripts/run_evaluation.py --sample
```

Metrics tracked:
- **Retrieval**: Recall@K, Precision@K, MRR, NDCG
- **Generation**: Answer similarity, Faithfulness, Citation precision

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| API Framework | FastAPI |
| Agent Orchestration | LangGraph |
| Vector Database | Qdrant |
| Embeddings | OpenAI text-embedding-3-small |
| LLM | GPT-4 Turbo |
| Reranking | Cohere |
| Frontend | Next.js (Week 4) |

## 📈 Implementation Status

- [x] **Core Infrastructure**: SEC EDGAR loader, document chunking, data models
- [x] **Retrieval System**: Hybrid search (BM25 + dense), Qdrant vector store, Cohere reranking
- [x] **Multi-Agent Workflow**: Complete LangGraph implementation with all 6 agents
- [x] **Validator Agent**: Hallucination detection with factual accuracy checks
- [x] **Citations Engine**: Automatic claim extraction and source linking
- [x] **Evaluation Framework**: Comprehensive metrics and test datasets
- [x] **Testing Suite**: API validation, workflow tests, code validation tools
- [ ] **Frontend Integration**: Next.js UI (in progress)
- [ ] **Production Deployment**: Docker, CI/CD, monitoring

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see LICENSE file for details.

## 👤 Author

**Arun K**
- Portfolio project demonstrating AI engineering skills
- Built for $150K+ AI/ML engineering roles

---

*Built with ❤️ and lots of ☕*
