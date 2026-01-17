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
    MARKET_DATA = "market_data"


class QueryComplexity(str, Enum):
    """Query complexity levels for routing."""
    SIMPLE = "simple"          # Single fact lookup
    MODERATE = "moderate"      # Multi-step reasoning
    COMPLEX = "complex"        # Multi-document analysis


class ResponseLength(str, Enum):
    """Response length modes for user selection."""
    SHORT = "short"            # 2-3 lines, ~50-100 words
    NORMAL = "normal"          # 5-6 lines, ~150-250 words (default)
    DETAILED = "detailed"      # Comprehensive, ~400-800 words


class AgentRole(str, Enum):
    """Roles in the multi-agent system."""
    ROUTER = "router"
    PLANNER = "planner"
    RETRIEVER = "retriever"
    ANALYST = "analyst"
    SYNTHESIZER = "synthesizer"
    VALIDATOR = "validator"


# ============================================================================
# Complexity Classification Models
# ============================================================================

class ComplexityInfo(BaseModel):
    """Query complexity classification for frontend display."""
    level: QueryComplexity = Field(..., description="Complexity level: SIMPLE, MODERATE, or COMPLEX")
    display_label: str = Field(..., description="User-friendly label: 'Quick Look', 'Analysis', 'Deep Research'")
    display_color: str = Field(..., description="UI color hint: 'green', 'blue', 'purple'")
    estimated_time_seconds: int = Field(..., description="Rough estimate for user")
    reasoning: str = Field(..., description="Why this classification: 'Single fact lookup', 'Multi-company comparison'")
    features_enabled: List[str] = Field(default_factory=list, description="What agents will run: ['planner', 'analyst']")


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
    fiscal_period: Optional[str] = Field(
        None, description="Fiscal period label (e.g., 'Q4 FY2025', 'FY2025')"
    )
    period_end_date: Optional[datetime] = Field(
        None, description="End date of the fiscal period covered"
    )
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
    retrieval_method: Literal["dense", "sparse", "hybrid", "tool"] = Field(
        ..., description="How this document was retrieved"
    )


# ============================================================================
# Query Models
# ============================================================================

class QueryRequest(BaseModel):
    """Incoming query request from user."""
    query: str = Field(..., description="Natural language query", min_length=10)
    response_length: Optional[ResponseLength] = Field(
        default=ResponseLength.NORMAL, 
        description="Desired response length: short, normal, or detailed"
    )
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
    """A citation linking a claim to its source - enhanced for interactive viewing."""
    citation_id: str = Field(..., description="Unique citation identifier")
    citation_number: int = Field(..., description="Display number [1], [2], etc.")
    claim: str = Field(..., description="The claim being cited")

    # Source identification
    source_chunk_id: str = Field(..., description="Source chunk ID")
    source_document_id: str = Field(..., description="Parent document ID")

    # Source content - ENHANCED
    source_text: str = Field(..., description="Exact text supporting the claim")
    source_context: str = Field(
        default="",
        description="2-3 sentences before/after for context"
    )
    highlight_start: int = Field(
        default=0,
        description="Character position where relevant text starts in source_context"
    )
    highlight_end: int = Field(
        default=0,
        description="Character position where relevant text ends in source_context"
    )

    # Source metadata - ENHANCED
    source_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Full metadata for display: ticker, company_name, document_type, filing_date, section, page_number, source_url"
    )

    # Confidence and validation
    confidence: float = Field(..., description="Confidence score (0-1)")
    validation_method: Literal["exact_match", "semantic_similarity", "llm_verified"] = Field(
        default="semantic_similarity",
        description="Method used to validate this citation"
    )

    # For UI display
    preview_text: str = Field(
        default="",
        description="Short preview for tooltip (50 chars)"
    )

    # Legacy fields (for backward compatibility)
    page_reference: Optional[str] = Field(None, description="Page/section reference")
    source_url: Optional[str] = Field(None, description="Link to original source")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional citation metadata (ticker, filing, section, etc.)"
    )


class CitedResponse(BaseModel):
    """Response with inline citations."""
    answer: str = Field(..., description="The answer with citation markers [1], [2], etc.")
    citations: List[Citation] = Field(..., description="List of citations")
    sources: List[DocumentMetadata] = Field(..., description="Source documents used")


# ============================================================================
# Validation Models
# ============================================================================

class ValidationResult(BaseModel):
    """Validation results for frontend trust display."""
    # Overall status
    is_valid: bool = Field(..., description="Whether validation passed")
    trust_level: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Trust level: high (≥85%), medium (≥65%), low (<65%)"
    )
    trust_label: str = Field(
        ...,
        description="User-friendly label: 'Verified', 'Review Recommended', 'Low Confidence'"
    )
    trust_color: str = Field(
        ...,
        description="UI color hint: 'green', 'amber', 'red'"
    )

    # Confidence breakdown
    overall_confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Overall confidence score (0-1)"
    )
    confidence_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-aspect scores: factual_accuracy, citation_coverage, numerical_accuracy, source_quality"
    )

    # Validation details
    claims_checked: int = Field(default=0, description="Total claims checked")
    claims_verified: int = Field(default=0, description="Claims successfully verified")
    claims_unverified: int = Field(default=0, description="Claims that couldn't be verified")

    # Sources quality
    sources_used: int = Field(default=0, description="Number of source documents used")
    avg_source_relevance: float = Field(
        default=0.0,
        description="Average relevance score of sources"
    )
    source_diversity: str = Field(
        default="",
        description="Human-readable source description, e.g., '3 10-K filings, 1 earnings call'"
    )

    # Feedback for user
    validation_notes: List[str] = Field(
        default_factory=list,
        description="Human-readable notes about validation results"
    )

    # Loop info
    validation_attempts: int = Field(
        default=1,
        description="Number of validation iterations performed"
    )
    required_revalidation: bool = Field(
        default=False,
        description="Whether another validation iteration is needed"
    )


def calculate_trust_level(confidence: float) -> tuple:
    """
    Calculate trust level from confidence score.

    Args:
        confidence: Confidence score (0-1)

    Returns:
        Tuple of (trust_level, trust_label, trust_color)
    """
    if confidence >= 0.85:
        return ("high", "Verified", "green")
    elif confidence >= 0.65:
        return ("medium", "Review Recommended", "amber")
    else:
        return ("low", "Low Confidence", "red")


# ============================================================================
# Analyst Data Models
# ============================================================================

class ExtractedMetric(BaseModel):
    """A single extracted financial metric for Analyst's Notebook."""
    metric_name: str = Field(..., description="e.g., 'gross_margin', 'revenue', 'debt_to_equity'")
    display_name: str = Field(..., description="e.g., 'Gross Margin', 'Total Revenue'")
    value: float = Field(..., description="Numeric value of the metric")
    formatted_value: str = Field(..., description="e.g., '43.5%', '$394.3B'")
    unit: Literal["percent", "currency", "ratio", "count", "other"] = Field(
        default="other",
        description="Type of unit for proper formatting"
    )
    currency: Optional[str] = Field(None, description="e.g., 'USD' if unit is currency")

    # Source tracking
    company: str = Field(..., description="Ticker symbol")
    fiscal_period: str = Field(..., description="e.g., 'FY2023', 'Q3 2024'")
    source_section: str = Field(default="", description="e.g., 'Item 7 - MD&A'")
    source_citation_id: Optional[str] = Field(None, description="Link to citation")

    # For comparisons
    comparison_value: Optional[float] = Field(None, description="Previous period or peer value")
    comparison_label: Optional[str] = Field(None, description="e.g., 'vs FY2022', 'vs Industry Avg'")
    change_percent: Optional[float] = Field(None, description="Percent change if comparison exists")
    change_direction: Optional[Literal["up", "down", "flat"]] = None


class ExtractedComparison(BaseModel):
    """A comparison between two or more companies/periods."""
    comparison_type: Literal["company_vs_company", "period_vs_period", "vs_benchmark"] = Field(
        ...,
        description="Type of comparison being made"
    )
    metric_name: str = Field(..., description="Metric being compared")
    display_name: str = Field(..., description="Human-readable metric name")
    items: List[ExtractedMetric] = Field(..., description="Metrics being compared")
    winner: Optional[str] = Field(None, description="Company/period with better value")
    insight: Optional[str] = Field(None, description="Brief insight, e.g., 'Apple leads by 5.2%'")


class AnalystNotebook(BaseModel):
    """Structured extracted data for frontend Analyst's Notebook display."""
    metrics: List[ExtractedMetric] = Field(
        default_factory=list,
        description="Individual metrics extracted from documents"
    )
    comparisons: List[ExtractedComparison] = Field(
        default_factory=list,
        description="Comparisons between companies/periods"
    )
    key_findings: List[str] = Field(
        default_factory=list,
        description="Bullet-point insights from analysis"
    )
    data_quality: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Quality assessment of extracted data"
    )
    companies_analyzed: List[str] = Field(
        default_factory=list,
        description="List of ticker symbols analyzed"
    )
    periods_covered: List[str] = Field(
        default_factory=list,
        description="Fiscal periods covered in analysis"
    )


# ============================================================================
# Agent Progress Events
# ============================================================================

class StepEvent(BaseModel):
    """Event emitted during agent execution for real-time UI updates."""
    event_type: Literal[
        "step_detail",
        "sub_query",
        "retrieval_progress",
        "analysis_insight",
        "validation_check"
    ] = Field(..., description="Type of progress event")
    agent: AgentRole = Field(..., description="Agent that generated this event")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this event occurred"
    )
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Event-specific data payload"
    )


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
    response_length: ResponseLength = Field(default=ResponseLength.NORMAL, description="Desired response length")
    filters: Optional[Dict[str, Any]] = Field(None, description="Query filters")
    
    # Router output
    complexity: Optional[QueryComplexity] = Field(None, description="Classified complexity")
    complexity_info: Optional[ComplexityInfo] = Field(None, description="Detailed complexity classification")

    # Planner output
    sub_queries: List[SubQuery] = Field(default_factory=list, description="Decomposed queries")
    
    # Retriever output
    retrieved_docs: List[RetrievedDocument] = Field(
        default_factory=list, description="Retrieved documents"
    )
    
    # Analyst output
    extracted_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw extracted data (backward compatibility)"
    )
    analyst_notebook: Optional[AnalystNotebook] = Field(
        None,
        description="Structured extracted data for frontend Analyst's Notebook cards"
    )
    
    # Synthesizer output
    draft_response: Optional[str] = Field(None, description="Draft response")
    citations: List[Citation] = Field(default_factory=list, description="Generated citations")
    
    # Validator output
    is_valid: bool = Field(default=False, description="Validation passed")
    validation_feedback: Optional[str] = Field(None, description="Validation feedback")
    validation_result: Optional[ValidationResult] = Field(
        None,
        description="Detailed validation results for frontend trust display"
    )

    # Execution tracking
    current_agent: Optional[AgentRole] = Field(None, description="Currently executing agent")
    iteration_count: int = Field(default=0, description="Number of iterations")
    error: Optional[str] = Field(None, description="Error message if failed")
    step_events: List[StepEvent] = Field(
        default_factory=list,
        description="Real-time progress events for streaming UI updates"
    )


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


# ============================================================================
# Follow-Up Question Models
# ============================================================================

class FollowUpQuestion(BaseModel):
    """A follow-up question generated from query context."""
    id: str = Field(..., description="Unique question identifier")
    text: str = Field(..., description="The follow-up question text")
    category: Literal["temporal", "deeper", "comparative", "related"] = Field(
        ..., description="Category of follow-up question"
    )
    relevant_chunk_ids: List[str] = Field(
        default_factory=list, description="Chunk IDs that can answer this question"
    )
    requires_new_retrieval: bool = Field(
        default=False, description="True for comparisons to other companies"
    )


class QueryResponseWithFollowUps(BaseModel):
    """Extended response including follow-up questions."""
    answer: str = Field(..., description="The generated answer")
    citations: List[Citation] = Field(default_factory=list, description="Supporting citations")
    reasoning_trace: Optional[List[Dict[str, Any]]] = Field(
        None, description="Agent reasoning steps"
    )
    follow_up_questions: List[FollowUpQuestion] = Field(
        default_factory=list, description="Generated follow-up questions"
    )
    query_id: str = Field(..., description="Query ID for follow-up execution reference")
    processing_time_ms: int = Field(..., description="Total processing time")
    confidence: float = Field(default=0.0, description="Response confidence score")
    validation: Optional[ValidationResult] = Field(None, description="Validation results")
    analyst_notebook: Optional[AnalystNotebook] = Field(None, description="Analyst notebook data")


class FollowUpRequest(BaseModel):
    """Request to execute a follow-up question."""
    question_id: str = Field(..., description="The follow-up question ID")
    question_text: str = Field(..., description="The follow-up question text")
    parent_query_id: str = Field(..., description="ID of the parent query")


class FollowUpResponse(BaseModel):
    """Response from executing a follow-up question."""
    question: str = Field(..., description="The follow-up question that was answered")
    answer: str = Field(..., description="Short answer (3-5 sentences)")
    citations: List[Citation] = Field(default_factory=list, description="Supporting citations")
    execution_time_ms: int = Field(..., description="Total execution time in milliseconds")
    used_cache: bool = Field(..., description="Whether cached chunks were used")
