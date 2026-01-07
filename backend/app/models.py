"""
Pydantic Data Models

Defines all data structures used throughout the application:
- Request/Response models for API endpoints
- Document and chunk representations
- Agent state models for LangGraph
- Citation and source tracking models
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class DocumentType(str, Enum):
    """Types of financial documents supported."""
    SEC_10K = "10-K"
    SEC_10Q = "10-Q"
    SEC_8K = "8-K"
    EARNINGS_CALL = "earnings_call"
    PRESS_RELEASE = "press_release"


class QueryComplexity(str, Enum):
    """Query complexity levels for routing."""
    SIMPLE = "simple"          # Single fact lookup
    MODERATE = "moderate"      # Multi-step reasoning
    COMPLEX = "complex"        # Multi-document analysis


class AgentRole(str, Enum):
    """Roles in the multi-agent system."""
    ROUTER = "router"
    PLANNER = "planner"
    RETRIEVER = "retriever"
    ANALYST = "analyst"
    SYNTHESIZER = "synthesizer"
    VALIDATOR = "validator"


# ============================================================================
# Document Models
# ============================================================================

class DocumentMetadata(BaseModel):
    """Metadata for a financial document."""
    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: str = Field(..., description="Company name")
    document_type: DocumentType = Field(..., description="Type of document")
    filing_date: datetime = Field(..., description="Date of filing/publication")
    fiscal_year: Optional[int] = Field(None, description="Fiscal year covered")
    fiscal_quarter: Optional[int] = Field(None, description="Fiscal quarter (1-4)")
    source_url: str = Field(..., description="Original document URL")
    accession_number: Optional[str] = Field(None, description="SEC accession number")


class DocumentChunk(BaseModel):
    """A chunk of text from a financial document."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk text content")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    section: Optional[str] = Field(None, description="Document section (e.g., 'Risk Factors')")
    page_number: Optional[int] = Field(None, description="Page number in original")
    chunk_index: int = Field(..., description="Position in document")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")


class RetrievedDocument(BaseModel):
    """Document retrieved from vector store with relevance score."""
    chunk: DocumentChunk = Field(..., description="The document chunk")
    score: float = Field(..., description="Relevance score (0-1)")
    retrieval_method: Literal["dense", "sparse", "hybrid"] = Field(
        ..., description="How this document was retrieved"
    )


# ============================================================================
# Query Models
# ============================================================================

class QueryRequest(BaseModel):
    """Incoming query request from user."""
    query: str = Field(..., description="Natural language query", min_length=10)
    filters: Optional[Dict[str, Any]] = Field(
        None, description="Optional filters (ticker, date range, doc type)"
    )
    max_sources: int = Field(default=5, description="Maximum sources to cite")
    include_reasoning: bool = Field(default=False, description="Include agent reasoning trace")


class SubQuery(BaseModel):
    """Decomposed sub-query from planner."""
    sub_query: str = Field(..., description="The sub-query text")
    intent: str = Field(..., description="What this sub-query aims to find")
    required_docs: List[DocumentType] = Field(..., description="Document types needed")
    priority: int = Field(..., description="Execution priority (1=highest)")


# ============================================================================
# Citation Models
# ============================================================================

class Citation(BaseModel):
    """A citation linking a claim to its source."""
    citation_id: str = Field(..., description="Unique citation identifier")
    claim: str = Field(..., description="The claim being cited")
    source_chunk_id: str = Field(..., description="Source chunk ID")
    source_text: str = Field(..., description="Relevant text from source")
    confidence: float = Field(..., description="Confidence score (0-1)")
    page_reference: Optional[str] = Field(None, description="Page/section reference")


class CitedResponse(BaseModel):
    """Response with inline citations."""
    answer: str = Field(..., description="The answer with citation markers [1], [2], etc.")
    citations: List[Citation] = Field(..., description="List of citations")
    sources: List[DocumentMetadata] = Field(..., description="Source documents used")


# ============================================================================
# Agent State Models (for LangGraph)
# ============================================================================

class AgentState(BaseModel):
    """
    State object passed between agents in LangGraph workflow.
    
    This is the central state that gets updated as the query
    flows through the multi-agent pipeline.
    """
    # Input
    original_query: str = Field(..., description="Original user query")
    filters: Optional[Dict[str, Any]] = Field(None, description="Query filters")
    
    # Router output
    complexity: Optional[QueryComplexity] = Field(None, description="Classified complexity")
    
    # Planner output
    sub_queries: List[SubQuery] = Field(default_factory=list, description="Decomposed queries")
    
    # Retriever output
    retrieved_docs: List[RetrievedDocument] = Field(
        default_factory=list, description="Retrieved documents"
    )
    
    # Analyst output
    extracted_data: Dict[str, Any] = Field(
        default_factory=dict, description="Extracted facts and calculations"
    )
    
    # Synthesizer output
    draft_response: Optional[str] = Field(None, description="Draft response")
    citations: List[Citation] = Field(default_factory=list, description="Generated citations")
    
    # Validator output
    is_valid: bool = Field(default=False, description="Validation passed")
    validation_feedback: Optional[str] = Field(None, description="Validation feedback")
    
    # Execution tracking
    current_agent: Optional[AgentRole] = Field(None, description="Currently executing agent")
    iteration_count: int = Field(default=0, description="Number of iterations")
    error: Optional[str] = Field(None, description="Error message if failed")


# ============================================================================
# Response Models
# ============================================================================

class QueryResponse(BaseModel):
    """Final response to user query."""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    citations: List[Citation] = Field(..., description="Supporting citations")
    sources: List[DocumentMetadata] = Field(..., description="Source documents")
    confidence: float = Field(..., description="Overall confidence score")
    reasoning_trace: Optional[List[Dict[str, Any]]] = Field(
        None, description="Agent reasoning steps (if requested)"
    )
    processing_time_ms: int = Field(..., description="Total processing time")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code for programmatic handling")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


# ============================================================================
# Evaluation Models
# ============================================================================

class EvaluationResult(BaseModel):
    """Result from evaluation metrics."""
    query: str = Field(..., description="Test query")
    expected_answer: str = Field(..., description="Ground truth answer")
    generated_answer: str = Field(..., description="Model generated answer")
    metrics: Dict[str, float] = Field(..., description="Computed metrics")
    passed: bool = Field(..., description="Whether evaluation passed thresholds")
