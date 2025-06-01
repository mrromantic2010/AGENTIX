"""
LLM module for Agentix.

This module provides LLM client implementations and factory for different providers.
"""

from .base_client import BaseLLMClient, LLMResponse, LLMStreamResponse, MockLLMClient
from .openrouter_client import OpenRouterClient, create_openrouter_client
from .anthropic_client import AnthropicClient, create_anthropic_client
from .factory import (
    LLMClientFactory,
    get_llm_factory,
    create_llm_client,
    register_llm_client,
    create_anthropic_client,
    create_mock_client,
    get_recommended_model,
    list_openrouter_models,
    OPENROUTER_MODELS
)

__all__ = [
    # Base classes
    "BaseLLMClient",
    "LLMResponse",
    "LLMStreamResponse",
    "MockLLMClient",

    # OpenRouter client
    "OpenRouterClient",
    "create_openrouter_client",

    # Anthropic client
    "AnthropicClient",
    "create_anthropic_client",

    # Factory
    "LLMClientFactory",
    "get_llm_factory",
    "create_llm_client",
    "register_llm_client",
    "create_mock_client",

    # Model recommendations
    "get_recommended_model",
    "list_openrouter_models",
    "OPENROUTER_MODELS"
]
