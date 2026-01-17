"""
Model Selector

Provides tiered model selection for different agents and query complexities.
Supports both OpenAI and Google Gemini models.

Model Tiers:
- FAST: Gemini Flash Lite - for simple classification, fast synthesis
- STANDARD: GPT-4o-mini - for moderate tasks, balanced speed/quality
- COMPLEX: GPT-4o - for complex reasoning, validation

Usage:
    from app.llm.model_selector import get_llm_client, ModelTier
    
    client = get_llm_client(ModelTier.FAST)
    response = await client.generate(prompt)
"""

import logging
from enum import Enum
from typing import Optional, Dict, Any, Union
from functools import lru_cache

from openai import AsyncOpenAI
import google.generativeai as genai

from app.config import settings

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    """Model tier for different task complexities."""
    FAST = "fast"           # Gemini Flash Lite - simple tasks
    STANDARD = "standard"   # GPT-4o-mini - moderate tasks
    COMPLEX = "complex"     # GPT-4o - complex reasoning


class ModelProvider(str, Enum):
    """LLM provider."""
    OPENAI = "openai"
    GOOGLE = "google"


# Model to provider mapping
MODEL_PROVIDERS = {
    # Google Gemini models
    "gemini-2.0-flash-lite": ModelProvider.GOOGLE,
    "gemini-2.5-flash-lite": ModelProvider.GOOGLE,
    "gemini-1.5-flash": ModelProvider.GOOGLE,
    "gemini-1.5-pro": ModelProvider.GOOGLE,
    "gemini-pro": ModelProvider.GOOGLE,
    # OpenAI models
    "gpt-4o": ModelProvider.OPENAI,
    "gpt-4o-mini": ModelProvider.OPENAI,
    "gpt-4-turbo-preview": ModelProvider.OPENAI,
    "gpt-4-turbo": ModelProvider.OPENAI,
    "gpt-4": ModelProvider.OPENAI,
    "gpt-3.5-turbo": ModelProvider.OPENAI,
}


def get_model_provider(model: str) -> ModelProvider:
    """
    Determine the provider for a given model.
    
    Args:
        model: Model name
        
    Returns:
        ModelProvider enum
    """
    # Check explicit mappings first
    if model in MODEL_PROVIDERS:
        return MODEL_PROVIDERS[model]
    
    # Infer from model name
    if "gemini" in model.lower():
        return ModelProvider.GOOGLE
    return ModelProvider.OPENAI


def get_model_for_agent(agent_name: str) -> str:
    """
    Get the configured model for a specific agent.
    
    Args:
        agent_name: Name of the agent (router, planner, synthesizer, etc.)
        
    Returns:
        Model name string
    """
    agent_models = {
        "router": settings.ROUTER_MODEL,
        "planner": settings.PLANNER_MODEL,
        "fast_synthesizer": settings.FAST_SYNTHESIZER_MODEL,
        "synthesizer": settings.SYNTHESIZER_MODEL,
        "validator": settings.VALIDATOR_MODEL,
        "analyst": settings.ANALYST_MODEL,
    }
    
    model = agent_models.get(agent_name.lower(), settings.LLM_MODEL)
    logger.debug(f"Agent '{agent_name}' using model: {model}")
    return model


def get_model_for_tier(tier: ModelTier) -> str:
    """
    Get the model for a specific tier.
    
    Args:
        tier: ModelTier enum
        
    Returns:
        Model name string
    """
    tier_models = {
        ModelTier.FAST: settings.LLM_MODEL_FAST,
        ModelTier.STANDARD: settings.LLM_MODEL_STANDARD,
        ModelTier.COMPLEX: settings.LLM_MODEL_COMPLEX,
    }
    return tier_models.get(tier, settings.LLM_MODEL)


def get_model_for_complexity(complexity: str) -> str:
    """
    Get the appropriate model based on query complexity.
    
    Args:
        complexity: Query complexity (SIMPLE, MODERATE, COMPLEX)
        
    Returns:
        Model name string
    """
    complexity_map = {
        "simple": settings.LLM_MODEL_FAST,
        "moderate": settings.LLM_MODEL_STANDARD,
        "complex": settings.LLM_MODEL_COMPLEX,
    }
    return complexity_map.get(complexity.lower(), settings.LLM_MODEL_STANDARD)


class LLMClient:
    """
    Unified LLM client that handles both OpenAI and Google models.
    """
    
    def __init__(self, model: str):
        """
        Initialize LLM client for a specific model.
        
        Args:
            model: Model name
        """
        self.model = model
        self.provider = get_model_provider(model)
        
        if self.provider == ModelProvider.OPENAI:
            self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        elif self.provider == ModelProvider.GOOGLE:
            if settings.GOOGLE_API_KEY:
                genai.configure(api_key=settings.GOOGLE_API_KEY)
                self._gemini_model = genai.GenerativeModel(model)
            else:
                logger.warning(f"Google API key not configured, falling back to OpenAI")
                self.provider = ModelProvider.OPENAI
                self.model = settings.LLM_MODEL_STANDARD
                self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    
    async def generate(
        self,
        messages: list,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            response_format: Optional response format (e.g., {"type": "json_object"})
            
        Returns:
            Generated text response
        """
        if self.provider == ModelProvider.OPENAI:
            return await self._generate_openai(messages, temperature, max_tokens, response_format)
        else:
            return await self._generate_gemini(messages, temperature, max_tokens)
    
    async def _generate_openai(
        self,
        messages: list,
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict]
    ) -> str:
        """Generate using OpenAI API."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format
        
        response = await self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
    
    async def _generate_gemini(
        self,
        messages: list,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate using Google Gemini API."""
        # Convert OpenAI message format to Gemini format
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"Instructions: {content}\n")
            elif role == "user":
                prompt_parts.append(f"User: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n")
        
        prompt = "".join(prompt_parts)
        
        # Gemini generation config
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        # Use sync API wrapped in async (Gemini SDK limitation)
        import asyncio
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._gemini_model.generate_content(
                prompt,
                generation_config=generation_config
            )
        )
        
        return response.text if response.text else ""


# Client cache to avoid recreating clients
_client_cache: Dict[str, LLMClient] = {}


def get_llm_client(model_or_tier: Union[str, ModelTier]) -> LLMClient:
    """
    Get an LLM client for a model or tier.
    
    Caches clients to avoid recreation overhead.
    
    Args:
        model_or_tier: Model name string or ModelTier enum
        
    Returns:
        LLMClient instance
    """
    if isinstance(model_or_tier, ModelTier):
        model = get_model_for_tier(model_or_tier)
    else:
        model = model_or_tier
    
    if model not in _client_cache:
        _client_cache[model] = LLMClient(model)
        logger.info(f"Created LLM client for model: {model}")
    
    return _client_cache[model]


def get_client_for_agent(agent_name: str) -> LLMClient:
    """
    Get an LLM client configured for a specific agent.
    
    Args:
        agent_name: Name of the agent
        
    Returns:
        LLMClient instance
    """
    model = get_model_for_agent(agent_name)
    return get_llm_client(model)
