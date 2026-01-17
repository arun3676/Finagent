"""
Application Configuration Module

Manages all environment variables and application settings using Pydantic.
Supports multiple environments (development, staging, production).

Environment variables are loaded from .env file or system environment.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All sensitive values (API keys, credentials) should be set via
    environment variables, never hardcoded.
    """
    
    # Application Settings
    APP_NAME: str = "FinAgent"
    APP_ENV: str = Field(default="development", description="development|staging|production")
    APP_HOST: str = Field(default="0.0.0.0", description="Server host")
    APP_PORT: int = Field(default=8010, description="Server port")
    APP_RELOAD: bool = Field(default=True, description="Enable auto-reload in development")
    DEBUG: bool = Field(default=True, description="Enable debug mode")
    
    # API Keys - NEVER commit actual values
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key for embeddings and LLM")
    COHERE_API_KEY: str = Field(default="", description="Cohere API key for reranking")
    SEC_API_KEY: Optional[str] = Field(default=None, description="SEC EDGAR API key (optional)")
    
    # Vector Store Configuration
    QDRANT_HOST: str = Field(default="localhost", description="Qdrant server host")
    QDRANT_PORT: int = Field(default=6333, description="Qdrant server port")
    QDRANT_URL: Optional[str] = Field(default=None, description="Qdrant server URL (overrides host/port)")
    QDRANT_API_KEY: Optional[str] = Field(default=None, description="Qdrant API key for cloud")
    QDRANT_COLLECTION_NAME: str = Field(default="finagent_docs", description="Default collection name")
    QDRANT_TIMEOUT: int = Field(default=30, description="Qdrant client timeout in seconds")
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-large", description="OpenAI embedding model")
    EMBEDDING_DIMENSION: int = Field(default=3072, description="Embedding vector dimension")
    
    # LLM Configuration
    LLM_MODEL: str = Field(default="gpt-4-turbo-preview", description="Primary LLM model")
    LLM_TEMPERATURE: float = Field(default=0.1, description="LLM temperature for responses")
    LLM_MAX_TOKENS: int = Field(default=4096, description="Max tokens for LLM response")
    
    # Model Tiers for Performance Optimization
    LLM_MODEL_FAST: str = Field(default="gemini-2.0-flash-lite", description="Fast model for simple tasks")
    LLM_MODEL_STANDARD: str = Field(default="gpt-4o-mini", description="Standard model for moderate tasks")
    LLM_MODEL_COMPLEX: str = Field(default="gpt-4o", description="Complex model for reasoning tasks")
    
    # Per-Agent Model Assignments
    ROUTER_MODEL: str = Field(default="gemini-2.0-flash-lite", description="Router uses fast model")
    PLANNER_MODEL: str = Field(default="gpt-4o-mini", description="Planner uses standard model")
    FAST_SYNTHESIZER_MODEL: str = Field(default="gemini-2.0-flash-lite", description="Fast synth uses fast model")
    SYNTHESIZER_MODEL: str = Field(default="gpt-4o-mini", description="Synthesizer uses standard model")
    VALIDATOR_MODEL: str = Field(default="gpt-4o", description="Validator uses complex model")
    ANALYST_MODEL: str = Field(default="gpt-4o-mini", description="Analyst uses standard model")
    
    # Google/Gemini API Configuration
    GOOGLE_API_KEY: str = Field(default="", description="Google API key for Gemini models")
    
    # Retrieval Configuration
    RETRIEVAL_TOP_K: int = Field(default=10, description="Number of documents to retrieve")
    RERANK_TOP_K: int = Field(default=5, description="Number of documents after reranking")
    HYBRID_ALPHA: float = Field(default=0.7, description="Weight for dense vs sparse (0=sparse, 1=dense)")
    RETRIEVAL_SCORE_THRESHOLD: float = Field(
        default=0.2,
        description="Minimum retrieval score to keep a document"
    )
    
    # Chunking Configuration
    CHUNK_SIZE: int = Field(default=1000, description="Target chunk size in characters")
    CHUNK_OVERLAP: int = Field(default=200, description="Overlap between chunks")
    
    # CORS Settings
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins"
    )
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = Field(default=100, description="Requests per minute")

    # =========================================================================
    # Observability & Tracing (LangSmith)
    # =========================================================================
    LANGCHAIN_TRACING_V2: bool = Field(
        default=False,
        description="Enable LangSmith tracing for observability"
    )
    LANGCHAIN_API_KEY: str = Field(
        default="",
        description="LangSmith API key for tracing"
    )
    LANGCHAIN_PROJECT: str = Field(
        default="finagent",
        description="LangSmith project name for trace grouping"
    )
    LANGCHAIN_ENDPOINT: str = Field(
        default="https://api.smith.langchain.com",
        description="LangSmith API endpoint"
    )

    # =========================================================================
    # Evaluation Configuration
    # =========================================================================
    DEEPEVAL_API_KEY: str = Field(
        default="",
        description="DeepEval API key for evaluation dashboard"
    )
    EVALUATION_THRESHOLD: float = Field(
        default=0.7,
        description="Minimum score threshold for passing evaluation"
    )

    model_config = {
        "env_file": Path(__file__).parent.parent / ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"
    }


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to ensure settings are only loaded once.
    
    Returns:
        Settings instance with all configuration values
    """
    return Settings()


# Global settings instance
settings = get_settings()


# Validation helpers
def validate_api_keys() -> dict:
    """
    Validate that required API keys are configured.

    Returns:
        Dict with validation status for each required key
    """
    return {
        "openai": bool(settings.OPENAI_API_KEY),
        "cohere": bool(settings.COHERE_API_KEY),
        "google": bool(settings.GOOGLE_API_KEY),
        "qdrant": bool(settings.QDRANT_API_KEY) or settings.QDRANT_HOST == "localhost",
        "langsmith": bool(settings.LANGCHAIN_API_KEY),
        "deepeval": bool(settings.DEEPEVAL_API_KEY)
    }


def setup_langsmith_tracing() -> bool:
    """
    Configure LangSmith tracing if API key is available.

    Sets environment variables required by LangChain for tracing.

    Returns:
        True if tracing is enabled, False otherwise
    """
    import os

    if settings.LANGCHAIN_API_KEY and settings.LANGCHAIN_TRACING_V2:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
        os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
        return True

    return False
