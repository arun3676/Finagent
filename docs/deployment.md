# FinAgent Deployment Guide

## Prerequisites

- Docker & Docker Compose
- Python 3.10+
- Node.js 18+ (for frontend)
- API Keys: OpenAI, Cohere

## Local Development

### 1. Clone and Setup

```bash
cd finagent
cp backend/.env.example backend/.env
# Edit .env with your API keys
```

### 2. Start Services

```bash
# Start Qdrant vector database
docker-compose up -d qdrant

# Install Python dependencies
cd backend
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# Run the API server
uvicorn app.main:app --reload --port 8000
```

### 3. Verify Installation

```bash
curl http://localhost:8000/health
```

## Docker Deployment

### Build Images

```bash
# Build backend
docker build -t finagent-backend:latest ./backend

# Build frontend (when ready)
docker build -t finagent-frontend:latest ./frontend
```

### Run with Docker Compose

```bash
docker-compose up -d
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - COHERE_API_KEY=${COHERE_API_KEY}
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
    depends_on:
      - qdrant

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8000
    depends_on:
      - backend

volumes:
  qdrant_data:
```

## Cloud Deployment

### Option 1: Render

**Backend (Web Service)**

1. Connect GitHub repository
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables

**Qdrant (Qdrant Cloud)**

1. Create cluster at cloud.qdrant.io
2. Get API key and endpoint
3. Update `QDRANT_HOST` and `QDRANT_API_KEY`

### Option 2: AWS

**Architecture**

```
Route 53 → CloudFront → ALB → ECS Fargate
                              ├── Backend Service
                              └── Frontend Service
                                      │
                              Amazon OpenSearch (Vector)
```

**Terraform Example**

```hcl
resource "aws_ecs_service" "finagent_backend" {
  name            = "finagent-backend"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.backend.arn
  desired_count   = 2
  
  load_balancer {
    target_group_arn = aws_lb_target_group.backend.arn
    container_name   = "backend"
    container_port   = 8000
  }
}
```

### Option 3: GCP

**Architecture**

```
Cloud DNS → Cloud Load Balancer → Cloud Run
                                  ├── Backend
                                  └── Frontend
                                        │
                                  Vertex AI Vector Search
```

## Environment Variables

### Required

| Variable | Description |
|----------|-------------|
| OPENAI_API_KEY | OpenAI API key |
| COHERE_API_KEY | Cohere API key |
| QDRANT_HOST | Qdrant server host |
| QDRANT_PORT | Qdrant server port |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| APP_ENV | development | Environment name |
| DEBUG | true | Enable debug mode |
| LLM_MODEL | gpt-4-turbo-preview | LLM model |
| EMBEDDING_MODEL | text-embedding-3-small | Embedding model |

## Monitoring

### Health Checks

```bash
# API health
curl https://api.finagent.example.com/health

# Qdrant health
curl http://qdrant:6333/health
```

### Logging

Configure structured logging:

```python
import structlog

structlog.configure(
    processors=[
        structlog.processors.JSONRenderer()
    ]
)
```

### Metrics

Key metrics to monitor:
- Request latency (p50, p95, p99)
- Query success rate
- LLM token usage
- Vector store query time
- Error rates by type

## Scaling

### Horizontal Scaling

```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: finagent-backend
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: finagent-backend
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Database Scaling

**Qdrant Cluster**
- Use Qdrant Cloud for managed scaling
- Or deploy Qdrant cluster with replication

## Security Checklist

- [ ] API keys in environment variables
- [ ] HTTPS enabled
- [ ] Rate limiting configured
- [ ] CORS properly configured
- [ ] Input validation on all endpoints
- [ ] Audit logging enabled
- [ ] Regular dependency updates

## Troubleshooting

### Common Issues

**1. Qdrant Connection Failed**
```
Check QDRANT_HOST and QDRANT_PORT
Verify Qdrant container is running
Check network connectivity
```

**2. OpenAI Rate Limits**
```
Implement exponential backoff
Use embedding cache
Consider batch processing
```

**3. High Latency**
```
Check LLM response times
Optimize retrieval top_k
Enable response caching
```

## Backup & Recovery

### Qdrant Snapshots

```bash
# Create snapshot
curl -X POST 'http://localhost:6333/collections/finagent_docs/snapshots'

# List snapshots
curl 'http://localhost:6333/collections/finagent_docs/snapshots'

# Restore from snapshot
curl -X PUT 'http://localhost:6333/collections/finagent_docs/snapshots/recover' \
  -H 'Content-Type: application/json' \
  -d '{"location": "snapshot_name"}'
```
