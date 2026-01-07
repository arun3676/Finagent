# FinAgent 🤖💰

**Enterprise-grade Agentic RAG System for Financial Research**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

FinAgent is a multi-agent system that answers complex financial research queries using SEC filings and earnings call transcripts. It combines hybrid retrieval (dense + sparse), LLM-powered analysis, and automatic citation generation.

## 🎯 Features

- **Multi-Agent Architecture**: Specialized agents for routing, planning, retrieval, analysis, synthesis, and validation
- **Hybrid Search**: Combines dense embeddings (OpenAI) with sparse retrieval (BM25) using Reciprocal Rank Fusion
- **Cross-Encoder Reranking**: Cohere reranker for precision optimization
- **Automatic Citations**: Every claim linked to source documents
- **SEC Filing Support**: 10-K, 10-Q, 8-K with section-aware chunking
- **Earnings Call Processing**: Q&A pair preservation and speaker identification
- **Built-in Evaluation**: Retrieval and generation metrics with benchmarking

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

## 📈 Roadmap

- [x] Week 1: Core infrastructure & ingestion
- [ ] Week 2: Retrieval system & agents
- [ ] Week 3: Citations & evaluation
- [ ] Week 4: Frontend & deployment

## 🤝 Contributing

Contributions welcome! Please read the contributing guidelines first.

## 📄 License

MIT License - see LICENSE file for details.

## 👤 Author

**Arun K**
- Portfolio project demonstrating AI engineering skills
- Built for $150K+ AI/ML engineering roles

---

*Built with ❤️ and lots of ☕*
