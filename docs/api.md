# FinAgent API Documentation

## Base URL

```
Development: http://localhost:8000
Production: https://api.finagent.example.com
```

## Authentication

All API requests require an API key in the header:

```
Authorization: Bearer <your-api-key>
```

## Endpoints

### Health Check

#### GET /health

Check system health and component status.

**Response**
```json
{
  "status": "healthy",
  "components": {
    "api": "operational",
    "vector_store": "operational",
    "llm": "operational",
    "embeddings": "operational"
  }
}
```

---

### Query

#### POST /query

Submit a financial research query.

**Request Body**
```json
{
  "query": "What was Apple's revenue in fiscal year 2023?",
  "filters": {
    "ticker": "AAPL",
    "document_type": "10-K",
    "date_range": {
      "start": "2023-01-01",
      "end": "2023-12-31"
    }
  },
  "max_sources": 5,
  "include_reasoning": false
}
```

**Parameters**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| query | string | Yes | Natural language financial query |
| filters | object | No | Metadata filters |
| filters.ticker | string | No | Stock ticker symbol |
| filters.document_type | string | No | 10-K, 10-Q, 8-K, earnings_call |
| filters.date_range | object | No | Date range filter |
| max_sources | integer | No | Maximum sources to cite (default: 5) |
| include_reasoning | boolean | No | Include agent reasoning trace |

**Response**
```json
{
  "query": "What was Apple's revenue in fiscal year 2023?",
  "answer": "Apple's total revenue in fiscal year 2023 was $383.3 billion [1], representing a 2.8% decrease from the prior year's $394.3 billion [2].",
  "citations": [
    {
      "citation_id": "cite_1",
      "claim": "Apple's total revenue in fiscal year 2023 was $383.3 billion",
      "source_chunk_id": "aapl_10k_2023_chunk_42",
      "source_text": "Total net sales were $383,285 million...",
      "confidence": 0.95,
      "page_reference": "10-K FY2023, Item 7"
    }
  ],
  "sources": [
    {
      "ticker": "AAPL",
      "company_name": "Apple Inc.",
      "document_type": "10-K",
      "filing_date": "2023-11-03",
      "source_url": "https://sec.gov/..."
    }
  ],
  "confidence": 0.92,
  "reasoning_trace": null,
  "processing_time_ms": 2340
}
```

**Error Responses**

| Status | Description |
|--------|-------------|
| 400 | Invalid query format |
| 401 | Unauthorized |
| 429 | Rate limit exceeded |
| 500 | Internal server error |

---

### Ingestion

#### POST /ingest

Ingest a new document into the system.

**Request Body**
```json
{
  "ticker": "AAPL",
  "document_type": "10-K",
  "fiscal_year": 2023,
  "source_url": "https://sec.gov/...",
  "content": "..."
}
```

**Response**
```json
{
  "document_id": "aapl_10k_2023",
  "chunks_created": 156,
  "status": "success"
}
```

#### GET /ingest/status/{job_id}

Check ingestion job status.

**Response**
```json
{
  "job_id": "job_123",
  "status": "processing",
  "progress": 0.75,
  "chunks_processed": 120,
  "total_chunks": 160
}
```

---

### Search

#### POST /search

Direct search without agent processing.

**Request Body**
```json
{
  "query": "risk factors cybersecurity",
  "filters": {
    "ticker": "MSFT"
  },
  "top_k": 10,
  "search_type": "hybrid"
}
```

**Parameters**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| query | string | Yes | Search query |
| filters | object | No | Metadata filters |
| top_k | integer | No | Number of results (default: 10) |
| search_type | string | No | dense, sparse, or hybrid |

**Response**
```json
{
  "results": [
    {
      "chunk_id": "msft_10k_2023_chunk_89",
      "content": "Cybersecurity risks include...",
      "score": 0.89,
      "metadata": {
        "ticker": "MSFT",
        "document_type": "10-K",
        "section": "Item 1A - Risk Factors"
      }
    }
  ],
  "total_results": 10,
  "search_type": "hybrid"
}
```

---

### Evaluation

#### POST /evaluate

Run evaluation on a test query.

**Request Body**
```json
{
  "query": "What was Apple's revenue in 2023?",
  "expected_answer": "Apple's revenue was $383.3 billion",
  "relevant_chunk_ids": ["aapl_10k_2023_chunk_42"]
}
```

**Response**
```json
{
  "query": "What was Apple's revenue in 2023?",
  "generated_answer": "...",
  "metrics": {
    "recall_at_5": 0.8,
    "precision_at_5": 0.6,
    "answer_similarity": 0.85,
    "faithfulness": 0.9
  },
  "passed": true
}
```

---

## Rate Limits

| Tier | Requests/min | Queries/day |
|------|-------------|-------------|
| Free | 10 | 100 |
| Pro | 60 | 1000 |
| Enterprise | 300 | Unlimited |

## Error Codes

| Code | Description |
|------|-------------|
| INVALID_QUERY | Query format is invalid |
| TICKER_NOT_FOUND | Specified ticker not in database |
| NO_RESULTS | No relevant documents found |
| RATE_LIMITED | Too many requests |
| LLM_ERROR | Error from LLM provider |
| VECTOR_STORE_ERROR | Vector database error |

## SDKs

### Python

```python
from finagent import FinAgentClient

client = FinAgentClient(api_key="your-key")
response = client.query("What was Apple's revenue in 2023?")
print(response.answer)
```

### JavaScript

```javascript
import { FinAgentClient } from 'finagent-js';

const client = new FinAgentClient({ apiKey: 'your-key' });
const response = await client.query("What was Apple's revenue in 2023?");
console.log(response.answer);
```
