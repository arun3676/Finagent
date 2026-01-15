"""
FastAPI Application Entry Point

This module initializes the FastAPI application and configures:
- CORS middleware for frontend communication
- API routers for different endpoints
- Health check and status endpoints
- OpenAPI documentation customization

Usage:
    uvicorn app.main:app --reload --port 8010
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import json
import asyncio
import time
import logging

from langserve import add_routes
from langchain_core.runnables import RunnableLambda

from app.config import settings
from app.agents.workflow import FinAgentWorkflow
from app.models import AgentState

logger = logging.getLogger(__name__)

# Input model matching frontend request
class ChatInput(BaseModel):
    input: str
    chat_history: Optional[List[Any]] = None
    file: Optional[Dict[str, Any]] = None

# Query endpoint models
class QueryOptions(BaseModel):
    include_reasoning: Optional[bool] = False
    include_citations: Optional[bool] = True

class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    options: Optional[QueryOptions] = None

# Ingest endpoint models
class IngestRequest(BaseModel):
    ticker: str
    document_types: List[str] = ["10-K", "10-Q"]
    years: Optional[List[int]] = None

class IngestResponse(BaseModel):
    job_id: str
    ticker: str
    status: str
    message: str

# Store for tracking ingestion jobs (in-memory for now)
ingest_jobs = {}

# Store for caching citations from recent queries (in-memory for now)
# Maps citation_id -> full citation data with extended context
citation_cache = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    """
    print("FinAgent starting up...")
    # Initialize workflow and agents
    await workflow.initialize()
    print("FinAgent workflow initialized successfully")
    yield
    print("FinAgent shutting down...")


app = FastAPI(
    title="FinAgent API",
    description="Enterprise-grade agentic RAG system for financial research",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Workflow
workflow = FinAgentWorkflow()

def input_adapter(input_data: Dict[str, Any]) -> AgentState:
    """Adapt frontend input to AgentState."""
    # Handle both dict (from LangServe) and Pydantic model
    query = input_data.get("input", "")
    return AgentState(original_query=query)

# Create the runnable chain
# We wrap the graph execution to handle input adaptation
chain = RunnableLambda(input_adapter) | workflow.graph

# Add LangServe routes
add_routes(
    app,
    chain,
    path="/chat",
    input_type=ChatInput,
)

@app.get("/ping", tags=["Health"])
async def ping():
    """Simple ping endpoint for debugging."""
    return {"status": "pong", "timestamp": time.time()}


@app.post("/query/test", tags=["Query"])
async def query_test(request: QueryRequest):
    """Test endpoint - returns hardcoded response without workflow."""
    async def generate():
        yield f"data: {json.dumps({'type': 'step', 'step': 'test', 'status': 'started'})}\n\n"
        await asyncio.sleep(0.1)

        # Simulate streaming response
        test_response = f"This is a test response for your query: {request.query}"
        words = test_response.split()
        for word in words:
            yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"
            await asyncio.sleep(0.05)

        yield f"data: {json.dumps({'type': 'citations', 'citations': []})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'metadata': {'query_time_ms': 100, 'model_used': 'test', 'sources_consulted': 0}})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.get("/", tags=["Health"])
async def root() -> Dict[str, str]:
    """Root endpoint - API welcome message."""
    return {
        "message": "Welcome to FinAgent API",
        "version": "0.1.0",
        "docs": "/docs",
        "chat": "/chat/playground"
    }


@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for monitoring and load balancers.
    """
    return {
        "status": "healthy",
        "components": {
            "api": "operational",
            # Add real checks here
        }
    }


@app.post("/query", tags=["Query"])
async def query(request: QueryRequest) -> Dict[str, Any]:
    """
    Process a query and return the response.
    """
    start_time = time.time()

    try:
        state = AgentState(original_query=request.query)
        result = await workflow.graph.ainvoke(state)

        processing_time = int((time.time() - start_time) * 1000)

        # Handle state being a dict or object
        if isinstance(result, dict):
            answer = result.get("draft_response") or result.get("final_answer", "")
            citations = result.get("citations", [])
            retrieved_docs = result.get("retrieved_docs", [])
            validation_result = result.get("validation_result")
            analyst_notebook = result.get("analyst_notebook")
        else:
            answer = getattr(result, "draft_response", "") or getattr(result, "final_answer", "")
            citations = getattr(result, "citations", [])
            retrieved_docs = getattr(result, "retrieved_docs", [])
            validation_result = getattr(result, "validation_result", None)
            analyst_notebook = getattr(result, "analyst_notebook", None)

        # Cache citations for later retrieval
        for c in citations:
            if hasattr(c, 'citation_id') and hasattr(c, 'citation_number'):
                citation_cache[c.citation_id] = {
                    "citation": c.model_dump() if hasattr(c, 'model_dump') else c,
                    "full_chunk": retrieved_docs[c.citation_number - 1].chunk if c.citation_number <= len(retrieved_docs) else None
                }

        # Serialize validation_result if present
        validation_data = None
        if validation_result:
            if hasattr(validation_result, 'model_dump'):
                validation_data = validation_result.model_dump()
            elif isinstance(validation_result, dict):
                validation_data = validation_result

        # Serialize analyst_notebook if present
        notebook_data = None
        if analyst_notebook:
            if hasattr(analyst_notebook, 'model_dump'):
                notebook_data = analyst_notebook.model_dump()
            elif isinstance(analyst_notebook, dict):
                notebook_data = analyst_notebook

        return {
            "answer": answer,
            "citations": citations,
            "reasoning_trace": result.get("reasoning_trace") if isinstance(result, dict) and request.options and request.options.include_reasoning else None,
            "processing_time_ms": processing_time,
            "confidence": result.get("confidence", 0.0) if isinstance(result, dict) else 0.0,
            "sources": retrieved_docs,
            "validation": validation_data,
            "analyst_notebook": notebook_data,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream", tags=["Query"])
async def query_stream(request: QueryRequest):
    """
    Process a query and stream the response.
    Uses non-streaming workflow execution for reliability.
    """
    logger.info(f"[query/stream] Received query: {request.query[:50]}...")

    async def generate():
        start_time = time.time()

        try:
            # Send initial status
            yield f"data: {json.dumps({'type': 'step', 'step': 'router', 'status': 'started'})}\n\n"

            # Run workflow synchronously (more reliable than streaming events)
            state = AgentState(original_query=request.query)
            logger.info("[query/stream] Running workflow...")

            result = await workflow.graph.ainvoke(state)
            logger.info("[query/stream] Workflow completed")

            # Extract results
            if isinstance(result, dict):
                answer = result.get("draft_response") or result.get("final_answer") or ""
                citations = result.get("citations", [])
                docs = result.get("retrieved_docs", [])
                complexity_info = result.get("complexity_info")
                validation_result = result.get("validation_result")
                analyst_notebook = result.get("analyst_notebook")
                step_events = result.get("step_events", [])
            else:
                answer = getattr(result, "draft_response", "") or getattr(result, "final_answer", "")
                citations = getattr(result, "citations", [])
                docs = getattr(result, "retrieved_docs", [])
                complexity_info = getattr(result, "complexity_info", None)
                validation_result = getattr(result, "validation_result", None)
                analyst_notebook = getattr(result, "analyst_notebook", None)
                step_events = getattr(result, "step_events", [])

            # Emit step events from agents
            if step_events:
                for step_event in step_events:
                    # Serialize step event
                    if hasattr(step_event, 'model_dump'):
                        event_data = step_event.model_dump()
                    elif isinstance(step_event, dict):
                        event_data = step_event
                    else:
                        continue

                    # Emit based on event type
                    event_type = event_data.get('event_type')
                    agent = event_data.get('agent')
                    data = event_data.get('data', {})

                    # Create appropriate SSE event
                    sse_event = {
                        'type': event_type,
                        'agent': agent,
                        'data': data
                    }

                    yield f"data: {json.dumps(sse_event)}\n\n"

            # Emit complexity info immediately after workflow completes
            if complexity_info:
                # Serialize complexity_info
                if hasattr(complexity_info, 'model_dump'):
                    complexity_data = complexity_info.model_dump()
                elif isinstance(complexity_info, dict):
                    complexity_data = complexity_info
                else:
                    complexity_data = None

                if complexity_data:
                    yield f"data: {json.dumps({'type': 'complexity', 'data': complexity_data})}\n\n"

            # Emit validation results if available
            if validation_result:
                # Serialize validation_result
                if hasattr(validation_result, 'model_dump'):
                    validation_data = validation_result.model_dump()
                elif isinstance(validation_result, dict):
                    validation_data = validation_result
                else:
                    validation_data = None

                if validation_data:
                    yield f"data: {json.dumps({'type': 'validation', 'data': validation_data})}\n\n"

            # Emit analyst notebook if available
            if analyst_notebook:
                # Serialize analyst_notebook
                if hasattr(analyst_notebook, 'model_dump'):
                    notebook_data = analyst_notebook.model_dump()
                elif isinstance(analyst_notebook, dict):
                    notebook_data = analyst_notebook
                else:
                    notebook_data = None

                if notebook_data:
                    yield f"data: {json.dumps({'type': 'analyst_notebook', 'data': notebook_data})}\n\n"

            # Send the answer as tokens
            if answer:
                # Send in chunks to simulate streaming
                words = answer.split()
                for i in range(0, len(words), 5):
                    chunk = " ".join(words[i:i+5]) + " "
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

            processing_time = int((time.time() - start_time) * 1000)

            # Serialize citations and cache them
            serialized_citations = []
            for c in citations:
                if hasattr(c, 'model_dump'):
                    citation_data = c.model_dump()
                    serialized_citations.append(citation_data)
                    # Cache citation for later retrieval
                    citation_cache[c.citation_id] = {
                        "citation": citation_data,
                        "full_chunk": docs[c.citation_number - 1].chunk if c.citation_number <= len(docs) else None
                    }
                elif isinstance(c, dict):
                    serialized_citations.append(c)
                    if "citation_id" in c:
                        citation_cache[c["citation_id"]] = {
                            "citation": c,
                            "full_chunk": None
                        }

            # Send final metadata
            yield f"data: {json.dumps({'type': 'citations', 'citations': serialized_citations})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'metadata': {'query_time_ms': processing_time, 'model_used': 'gpt-4', 'sources_consulted': len(docs)}})}\n\n"

        except Exception as e:
            logger.error(f"[query/stream] Error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@app.post("/query/stream-old", tags=["Query"])
async def query_stream_old(request: QueryRequest):
    """
    Old streaming implementation (backup).
    """
    async def generate():
        start_time = time.time()
        final_citations = []
        final_docs = []
        final_answer = ""
        tokens_emitted = False

        try:
            state = AgentState(original_query=request.query)

            # Stream events and capture final state in ONE pass
            async for event in workflow.graph.astream_events(state, version="v2"):
                event_type = event.get("event", "")
                event_name = event.get("name", "")

                # Stream LLM tokens as they're generated
                if event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        token_content = chunk.content
                        if isinstance(token_content, list):
                            token_content = "".join(str(c) for c in token_content)
                        if token_content:
                            yield f"data: {json.dumps({'type': 'token', 'content': token_content})}\n\n"
                            tokens_emitted = True

                # Track agent progress
                elif event_type == "on_chain_start":
                    if event_name in ["router", "retriever", "analyst", "synthesizer"]:
                        yield f"data: {json.dumps({'type': 'step', 'step': event_name, 'status': 'started'})}\n\n"

                elif event_type == "on_chain_end":
                    if event_name in ["router", "retriever", "analyst", "synthesizer"]:
                        yield f"data: {json.dumps({'type': 'step', 'step': event_name, 'status': 'completed'})}\n\n"

                    # Capture final state from the graph's end event (name may vary)
                    output = event.get("data", {}).get("output", None)
                    is_graph_end = event_name in ["LangGraph", "graph"]
                    if output and (is_graph_end or hasattr(output, "draft_response") or (isinstance(output, dict) and ("draft_response" in output or "final_answer" in output))):
                        # Handle both dict output and AgentState/Pydantic objects
                        if isinstance(output, dict):
                            final_citations = output.get("citations", []) or final_citations
                            final_docs = output.get("retrieved_docs", []) or final_docs
                            final_answer = (
                                output.get("draft_response")
                                or output.get("final_answer")
                                or final_answer
                            )
                        else:
                            final_citations = getattr(output, "citations", []) or final_citations
                            final_docs = getattr(output, "retrieved_docs", []) or final_docs
                            final_answer = (
                                getattr(output, "draft_response", "")
                                or getattr(output, "final_answer", "")
                                or final_answer
                            )

            processing_time = int((time.time() - start_time) * 1000)

            # If no streaming tokens were emitted (common in some tool-only paths),
            # send the final answer once so the UI has text to render.
            if final_answer and not tokens_emitted:
                yield f"data: {json.dumps({'type': 'token', 'content': final_answer})}\n\n"

            # Serialize citations properly (handle Pydantic models)
            serialized_citations = []
            for c in final_citations:
                if hasattr(c, "model_dump"):
                    serialized_citations.append(c.model_dump())
                elif isinstance(c, dict):
                    serialized_citations.append(c)
                else:
                    # Fallback to best-effort serialization
                    try:
                        serialized_citations.append(json.loads(json.dumps(c, default=str)))
                    except Exception:
                        serialized_citations.append({"value": str(c)})

            # Send final metadata
            yield f"data: {json.dumps({'type': 'citations', 'citations': serialized_citations})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'metadata': {'query_time_ms': processing_time, 'model_used': 'gpt-4', 'sources_consulted': len(final_docs)}})}\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/ingest", tags=["Ingestion"], response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """
    Trigger document ingestion for a company ticker.
    """
    import uuid

    job_id = str(uuid.uuid4())

    # Initialize ingestion job status
    ingest_jobs[job_id] = {
        "ticker": request.ticker,
        "status": "processing",
        "progress_percent": 0,
        "current_step": "Fetching SEC filings",
        "documents_processed": 0,
        "documents_total": 0,
        "chunks_created": 0,
        "error_message": None
    }

    # Run ingestion in background
    asyncio.create_task(run_ingestion(job_id, request))

    return IngestResponse(
        job_id=job_id,
        ticker=request.ticker,
        status="queued",
        message=f"Ingestion started for {request.ticker}"
    )


async def run_ingestion(job_id: str, request: IngestRequest):
    """Background task to run the ingestion process."""
    try:
        from app.ingestion.sec_edgar_loader import SECEdgarLoader
        from app.chunking.sec_chunker import SECChunker
        from app.retrieval.embeddings import EmbeddingService
        from app.retrieval.vector_store import VectorStore
        from app.retrieval.bm25_index import BM25Index
        import os

        # Update status
        ingest_jobs[job_id]["current_step"] = "Initializing"
        ingest_jobs[job_id]["progress_percent"] = 10

        logger.info(f"Starting ingestion for job {job_id}, ticker {request.ticker}")

        # Initialize services
        try:
            logger.info("Initializing SEC Edgar loader...")
            loader = SECEdgarLoader()
            
            logger.info("Initializing chunker...")
            chunker = SECChunker()
            
            logger.info("Initializing embedding service...")
            embedder = EmbeddingService()
            
            logger.info("Initializing vector store...")
            vector_store = VectorStore()
            
            logger.info("Initializing BM25 index...")
            bm25_index = BM25Index()
            index_path = os.path.join(os.getcwd(), "data", "indexes", "bm25.pkl")
            
            if os.path.exists(index_path):
                try:
                    bm25_index.load_index(index_path)
                    logger.info("Loaded existing BM25 index")
                except Exception as e:
                    logger.warning(f"Failed to load BM25 index: {e}")
            
            logger.info("Creating collection...")
            vector_store.create_collection()
            logger.info("Collection created or already exists")

        except Exception as e:
            logger.error(f"Service initialization failed: {str(e)}", exc_info=True)
            raise

        # Fetch filings
        ingest_jobs[job_id]["current_step"] = "Fetching SEC filings"
        ingest_jobs[job_id]["progress_percent"] = 20

        all_filings = []
        for doc_type in request.document_types:
            filings = await loader.get_filings(
                ticker=request.ticker,
                filing_type=doc_type,
                limit=3 if request.years is None else len(request.years)
            )
            all_filings.extend(filings)

        ingest_jobs[job_id]["documents_total"] = len(all_filings)
        ingest_jobs[job_id]["progress_percent"] = 30

        # Process each filing
        total_chunks = 0
        for i, filing in enumerate(all_filings):
            # filing is a dict, not an object
            filing_type = filing.get("filing_type", "Unknown")
            filing_date = filing.get("filing_date", "Unknown")

            ingest_jobs[job_id]["current_step"] = f"Processing {filing_type} - {filing_date}"
            ingest_jobs[job_id]["documents_processed"] = i + 1
            ingest_jobs[job_id]["progress_percent"] = 30 + int((i + 1) / len(all_filings) * 50)

            logger.info(f"Processing filing {i+1}/{len(all_filings)}: {filing_type} from {filing_date}")

            # Download the filing content
            try:
                content = await loader.download_filing(
                    accession_number=filing["accession_number"],
                    cik=filing["cik"],
                    primary_document=filing.get("primary_document")
                )
                logger.info(f"Downloaded {len(content)} characters")

                # Skip empty or very small filings
                if not content or len(content) < 100:
                    logger.warning(f"Skipping filing {filing_type} - content too small")
                    continue

            except Exception as e:
                logger.warning(f"Failed to download filing {filing_type}: {str(e)}")
                continue

            # Create metadata for chunking
            from app.models import DocumentMetadata, DocumentType
            from app.utils.temporal import derive_fiscal_metadata
            from datetime import datetime

            # Parse filing date
            try:
                filing_date_obj = datetime.strptime(filing_date, "%Y-%m-%d")
            except:
                filing_date_obj = datetime.now()

            report_date = filing.get("report_date")
            if isinstance(report_date, str):
                try:
                    report_date = datetime.strptime(report_date, "%Y-%m-%d")
                except ValueError:
                    report_date = None

            # Map filing type to DocumentType enum
            doc_type_map = {
                "10-K": DocumentType.SEC_10K,
                "10-Q": DocumentType.SEC_10Q,
                "8-K": DocumentType.SEC_8K,
            }
            doc_type = doc_type_map.get(filing_type, DocumentType.SEC_10K)

            derived = derive_fiscal_metadata(
                report_date=report_date,
                fiscal_year_end_mmdd=filing.get("fiscal_year_end"),
                document_type=doc_type
            )

            metadata = DocumentMetadata(
                ticker=request.ticker,
                company_name=filing.get("company_name", request.ticker),
                document_type=doc_type,
                filing_date=filing_date_obj,
                fiscal_year=derived.fiscal_year,
                fiscal_quarter=derived.fiscal_quarter,
                fiscal_period=derived.fiscal_period,
                period_end_date=derived.period_end_date,
                source_url=filing.get("url", ""),
                accession_number=filing.get("accession_number")
            )

            # Chunk the document
            chunks = chunker.chunk_document(content, metadata)
            logger.info(f"Created {len(chunks)} chunks")

            # Embed chunks
            texts = [chunk.content for chunk in chunks]
            embeddings = await embedder.embed_texts(texts)
            logger.info(f"Generated {len(embeddings)} embeddings")

            # Add embeddings to chunks and store
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding

            await vector_store.upsert_chunks(chunks)
            
            # Update and save BM25 index
            try:
                bm25_index.add_documents(chunks)
                bm25_index.save_index(index_path)
                logger.info(f"Updated BM25 index with {len(chunks)} chunks")
            except Exception as e:
                logger.warning(f"Failed to update BM25 index: {e}")
                
            total_chunks += len(chunks)
            logger.info(f"Stored {len(chunks)} chunks. Total so far: {total_chunks}")

        ingest_jobs[job_id]["chunks_created"] = total_chunks
        ingest_jobs[job_id]["current_step"] = "Completed"
        ingest_jobs[job_id]["progress_percent"] = 100
        ingest_jobs[job_id]["status"] = "completed"

    except Exception as e:
        logger.error(f"Ingestion failed for job {job_id}: {str(e)}")
        ingest_jobs[job_id]["status"] = "failed"
        ingest_jobs[job_id]["error_message"] = str(e)
        ingest_jobs[job_id]["progress_percent"] = 0


@app.get("/ingest/{job_id}/progress", tags=["Ingestion"])
async def get_ingest_progress(job_id: str):
    """
    Get the progress of an ingestion job.
    """
    if job_id not in ingest_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return ingest_jobs[job_id]


@app.get("/companies", tags=["Companies"])
async def get_companies():
    """
    Get list of indexed companies.
    """
    try:
        from app.config import settings
        from qdrant_client import QdrantClient

        # Connect to Qdrant
        client = QdrantClient(
            url=settings.QDRANT_URL,
            timeout=settings.QDRANT_TIMEOUT
        )

        # Get collection info
        try:
            collection_info = client.get_collection(settings.QDRANT_COLLECTION_NAME)

            # Get unique tickers from metadata (this is a simplified version)
            # In production, you'd want to maintain a separate index of companies
            companies = []

            # For now, return an empty list if no data or basic info
            return {
                "companies": companies,
                "total": len(companies),
                "collection_size": collection_info.points_count if collection_info else 0
            }
        except Exception:
            # Collection doesn't exist yet
            return {
                "companies": [],
                "total": 0,
                "collection_size": 0
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/citation/{citation_id}/source", tags=["Citations"])
async def get_citation_source(citation_id: str):
    """
    Get full source document context for a citation.

    This endpoint provides extended context for interactive source viewing,
    including the full chunk, surrounding context, and document metadata.

    Args:
        citation_id: The citation ID (e.g., "cite_1", "cite_2")

    Returns:
        Full source context including chunk data and metadata
    """
    # Look up citation in cache
    if citation_id not in citation_cache:
        raise HTTPException(
            status_code=404,
            detail=f"Citation {citation_id} not found. It may have expired from cache."
        )

    cached = citation_cache[citation_id]
    citation_data = cached["citation"]
    full_chunk = cached.get("full_chunk")

    # Get surrounding context (if we have the chunk)
    context_before = ""
    context_after = ""

    if full_chunk:
        # Try to get surrounding chunks (would need chunk index lookup)
        # For now, we'll just return the current chunk's full content
        chunk_content = full_chunk.content if hasattr(full_chunk, 'content') else full_chunk.get('content', '')

        # Split content into before/after based on highlight position
        if citation_data.get("highlight_start", 0) > 0:
            context_before = chunk_content[:citation_data.get("highlight_start", 0)]

        if citation_data.get("highlight_end", 0) > 0:
            context_after = chunk_content[citation_data.get("highlight_end", 0):]

    # Format response
    return {
        "citation_id": citation_id,
        "citation": citation_data,
        "source_chunk": {
            "chunk_id": citation_data.get("source_chunk_id"),
            "document_id": citation_data.get("source_document_id"),
            "content": full_chunk.content if full_chunk and hasattr(full_chunk, 'content') else citation_data.get("source_context", ""),
            "section": citation_data.get("source_metadata", {}).get("section"),
            "page_number": citation_data.get("source_metadata", {}).get("page_number"),
        },
        "context_before": context_before[:500] if context_before else "",  # Limit size
        "context_after": context_after[:500] if context_after else "",
        "document_metadata": citation_data.get("source_metadata", {}),
        "highlight": {
            "start": citation_data.get("highlight_start", 0),
            "end": citation_data.get("highlight_end", 0),
            "text": citation_data.get("source_text", "")
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app", 
        host=settings.APP_HOST, 
        port=settings.APP_PORT, 
        reload=settings.APP_RELOAD
    )
