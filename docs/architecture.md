# FinAgent Architecture

## Overview

FinAgent is an enterprise-grade agentic RAG (Retrieval-Augmented Generation) system designed for financial research. It processes SEC filings and earnings call transcripts to answer complex financial queries with citations.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js)                        │
│                    (Week 4 Implementation)                       │
└─────────────────────────────┬───────────────────────────────────┘
                              │ REST API
┌─────────────────────────────▼───────────────────────────────────┐
│                     FastAPI Backend                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Router    │  │   Planner   │  │      Retriever          │  │
│  │   Agent     │──│   Agent     │──│       Agent             │  │
│  └─────────────┘  └─────────────┘  └───────────┬─────────────┘  │
│                                                 │                │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────▼─────────────┐  │
│  │  Validator  │──│ Synthesizer │──│       Analyst           │  │
│  │   Agent     │  │   Agent     │  │        Agent            │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    LangGraph Workflow Engine                     │
└─────────────────────────────┬───────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼───────┐    ┌────────▼────────┐    ┌──────▼──────┐
│    Qdrant     │    │     OpenAI      │    │   Cohere    │
│ Vector Store  │    │   Embeddings    │    │  Reranker   │
└───────────────┘    │      + LLM      │    └─────────────┘
                     └─────────────────┘
```

## Component Details

### 1. Ingestion Pipeline

**Purpose**: Load and preprocess financial documents

```
SEC EDGAR API → Document Loader → Section Parser → Chunker → Embedder → Vector Store
```

Components:
- **SECEdgarLoader**: Fetches 10-K, 10-Q, 8-K filings from SEC EDGAR
- **EarningsCallLoader**: Processes earnings call transcripts
- **XBRLParser**: Extracts structured financial data

### 2. Chunking Strategy

**Purpose**: Split documents while preserving semantic meaning

| Document Type | Strategy | Key Features |
|--------------|----------|--------------|
| SEC 10-K | Section-aware | Respects Item boundaries |
| SEC 10-Q | Section-aware | Quarterly section mapping |
| Earnings Calls | Q&A preservation | Keeps Q&A pairs together |
| Financial Tables | Table-intact | Never splits tables |

### 3. Retrieval System

**Purpose**: Find relevant documents for queries

```
Query → Hybrid Search → Reranking → Top-K Documents
         ├── Dense (OpenAI Embeddings)
         └── Sparse (BM25)
              └── RRF Fusion
```

Components:
- **EmbeddingService**: OpenAI text-embedding-3-small
- **VectorStore**: Qdrant with metadata filtering
- **BM25Index**: Keyword-based retrieval
- **HybridSearcher**: Combines dense + sparse with RRF
- **Reranker**: Cohere cross-encoder reranking

### 4. Multi-Agent System

**Purpose**: Process queries through specialized agents

```
┌─────────┐     ┌─────────┐     ┌───────────┐
│ Router  │────▶│ Planner │────▶│ Retriever │
└─────────┘     └─────────┘     └─────┬─────┘
                                      │
┌─────────┐     ┌───────────┐   ┌─────▼─────┐
│Validator│◀────│Synthesizer│◀──│  Analyst  │
└─────────┘     └───────────┘   └───────────┘
```

| Agent | Role | Input | Output |
|-------|------|-------|--------|
| Router | Classify complexity | Query | SIMPLE/MODERATE/COMPLEX |
| Planner | Decompose query | Complex query | Sub-queries |
| Retriever | Fetch documents | Query/Sub-queries | Relevant chunks |
| Analyst | Extract & calculate | Chunks + Query | Structured data |
| Synthesizer | Generate response | Data + Chunks | Cited response |
| Validator | Quality check | Response | Pass/Fail + Feedback |

### 5. Citation System

**Purpose**: Link claims to source documents

```
Response → Claim Extractor → Citation Linker → Formatter
                                    │
                              Source Chunks
```

Citation types:
- Inline: `[1]`, `[2]` markers in text
- Footnote: Superscript with footnotes
- Bibliography: Full source list

## Data Flow

### Query Processing Flow

```
1. User submits query
2. Router classifies complexity
3. If complex: Planner decomposes into sub-queries
4. Retriever fetches relevant documents
5. Analyst extracts data and performs calculations
6. Synthesizer generates response with citations
7. Validator checks quality
8. If invalid: Loop back to Synthesizer
9. Return cited response to user
```

### Ingestion Flow

```
1. Fetch document from SEC EDGAR
2. Parse into sections (10-K items, etc.)
3. Chunk sections with overlap
4. Generate embeddings for each chunk
5. Store in Qdrant with metadata
6. Build BM25 index for sparse retrieval
```

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| API | FastAPI | REST API server |
| Orchestration | LangGraph | Agent workflow |
| Vector DB | Qdrant | Semantic search |
| Embeddings | OpenAI | text-embedding-3-small |
| LLM | OpenAI | GPT-4 Turbo |
| Reranking | Cohere | Cross-encoder |
| Frontend | Next.js | User interface |

## Scalability Considerations

1. **Horizontal Scaling**: Stateless API servers behind load balancer
2. **Vector Store**: Qdrant cluster for high availability
3. **Caching**: Redis for embedding cache and query results
4. **Rate Limiting**: Per-user and global limits
5. **Async Processing**: Background jobs for ingestion

## Security

1. **API Keys**: Environment variables, never in code
2. **RBAC**: Role-based access to document collections
3. **Input Validation**: Pydantic models for all inputs
4. **Rate Limiting**: Prevent abuse
5. **Audit Logging**: Track all queries and responses
