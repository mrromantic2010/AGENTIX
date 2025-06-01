"""
LLM client factory for Agentix.

This module provides a factory for creating LLM clients based on configuration.
"""

import logging
from typing import Dict, Type, Optional

from ..config.llm_config import LLMConfig, LLMProvider
from ..utils.exceptions import LLMError
from .base_client import BaseLLMClient, MockLLMClient
from .openrouter_client import OpenRouterClient
from .anthropic_client import AnthropicClient


class LLMClientFactory:
    """
    Factory for creating LLM clients based on provider configuration.
    """

    def __init__(self):
        """Initialize the factory."""
        self.logger = logging.getLogger("agentix.llm.factory")
        self._clients: Dict[LLMProvider, Type[BaseLLMClient]] = {}

        # Register built-in clients
        self._register_builtin_clients()

    def _register_builtin_clients(self):
        """Register built-in LLM clients."""

        # Register OpenRouter client
        self._clients[LLMProvider.OPENROUTER] = OpenRouterClient

        # Register Anthropic client
        self._clients[LLMProvider.ANTHROPIC] = AnthropicClient

        # Register mock client for local development
        self._clients[LLMProvider.LOCAL] = MockLLMClient

        # Note: Other providers (OpenAI, etc.) would be registered here
        # For now, we'll use mock clients for providers that aren't implemented
        self._clients[LLMProvider.OPENAI] = MockLLMClient
        self._clients[LLMProvider.GOOGLE] = MockLLMClient
        self._clients[LLMProvider.COHERE] = MockLLMClient
        self._clients[LLMProvider.HUGGINGFACE] = MockLLMClient

    def register_client(self, provider: LLMProvider, client_class: Type[BaseLLMClient]):
        """
        Register a custom LLM client for a provider.

        Args:
            provider: The LLM provider
            client_class: The client class to register
        """

        if not issubclass(client_class, BaseLLMClient):
            raise ValueError("Client class must inherit from BaseLLMClient")

        self._clients[provider] = client_class
        self.logger.info(f"Registered LLM client for provider: {provider.value}")

    def create_client(self, config: LLMConfig) -> BaseLLMClient:
        """
        Create an LLM client based on the configuration.

        Args:
            config: LLM configuration

        Returns:
            Configured LLM client instance

        Raises:
            LLMError: If the provider is not supported or client creation fails
        """

        provider = config.provider

        if provider not in self._clients:
            raise LLMError(f"Unsupported LLM provider: {provider.value}")

        client_class = self._clients[provider]

        try:
            client = client_class(config)
            self.logger.info(f"Created LLM client for provider: {provider.value}, model: {config.model}")
            return client

        except Exception as e:
            raise LLMError(f"Failed to create LLM client for {provider.value}: {str(e)}")

    def get_supported_providers(self) -> list[LLMProvider]:
        """
        Get list of supported LLM providers.

        Returns:
            List of supported providers
        """
        return list(self._clients.keys())

    def is_provider_supported(self, provider: LLMProvider) -> bool:
        """
        Check if a provider is supported.

        Args:
            provider: The provider to check

        Returns:
            True if supported, False otherwise
        """
        return provider in self._clients


# Global factory instance
_global_factory: Optional[LLMClientFactory] = None


def get_llm_factory() -> LLMClientFactory:
    """
    Get the global LLM client factory.

    Returns:
        Global LLMClientFactory instance
    """
    global _global_factory
    if _global_factory is None:
        _global_factory = LLMClientFactory()
    return _global_factory


def create_llm_client(config: LLMConfig) -> BaseLLMClient:
    """
    Convenience function to create an LLM client.

    Args:
        config: LLM configuration

    Returns:
        Configured LLM client instance
    """
    factory = get_llm_factory()
    return factory.create_client(config)


def register_llm_client(provider: LLMProvider, client_class: Type[BaseLLMClient]):
    """
    Convenience function to register a custom LLM client.

    Args:
        provider: The LLM provider
        client_class: The client class to register
    """
    factory = get_llm_factory()
    factory.register_client(provider, client_class)


# Convenience functions for creating specific clients

def create_openrouter_client(api_key: str,
                           model: str = "openai/gpt-4-turbo",
                           **kwargs) -> OpenRouterClient:
    """
    Create an OpenRouter client with the specified configuration.

    Args:
        api_key: OpenRouter API key
        model: Model name to use
        **kwargs: Additional configuration options

    Returns:
        Configured OpenRouterClient instance
    """

    from ..config.llm_config import LLMConfig, LLMProvider

    config = LLMConfig(
        provider=LLMProvider.OPENROUTER,
        model=model,
        api_key=api_key,
        **kwargs
    )

    return OpenRouterClient(config)


def create_anthropic_client(api_key: str,
                          model: str = "claude-3-5-sonnet-20241022",
                          **kwargs) -> AnthropicClient:
    """
    Create an Anthropic client with the specified configuration.

    Args:
        api_key: Anthropic API key
        model: Claude model name to use
        **kwargs: Additional configuration options

    Returns:
        Configured AnthropicClient instance
    """

    from ..config.llm_config import LLMConfig, LLMProvider

    config = LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model=model,
        api_key=api_key,
        **kwargs
    )

    return AnthropicClient(config)


def create_mock_client(model: str = "mock-model", **kwargs) -> MockLLMClient:
    """
    Create a mock LLM client for testing.

    Args:
        model: Model name to use
        **kwargs: Additional configuration options

    Returns:
        Configured MockLLMClient instance
    """

    from ..config.llm_config import LLMConfig, LLMProvider

    config = LLMConfig(
        provider=LLMProvider.LOCAL,
        model=model,
        **kwargs
    )

    return MockLLMClient(config)


# Model recommendations for OpenRouter
OPENROUTER_MODELS = {
    # Best overall models
    "best": {
        "gpt-4-turbo": "openai/gpt-4-turbo",
        "claude-3-opus": "anthropic/claude-3-opus",
        "gemini-pro": "google/gemini-pro"
    },

    # Fast and efficient models
    "fast": {
        "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
        "claude-3-haiku": "anthropic/claude-3-haiku",
        "mixtral-8x7b": "mistralai/mixtral-8x7b-instruct"
    },

    # Code-specialized models
    "code": {
        "gpt-4-turbo": "openai/gpt-4-turbo",
        "codellama-34b": "meta-llama/codellama-34b-instruct",
        "deepseek-coder": "deepseek/deepseek-coder-33b-instruct"
    },

    # Open source models
    "open_source": {
        "llama-2-70b": "meta-llama/llama-2-70b-chat",
        "mixtral-8x7b": "mistralai/mixtral-8x7b-instruct",
        "nous-hermes-2": "nousresearch/nous-hermes-2-mixtral-8x7b-dpo"
    },

    # Budget-friendly models
    "budget": {
        "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
        "mistral-7b": "mistralai/mistral-7b-instruct",
        "openchat-7b": "openchat/openchat-7b"
    }
}


def get_recommended_model(category: str = "best", preference: str = "balanced") -> str:
    """
    Get a recommended OpenRouter model based on category and preference.

    Args:
        category: Model category ("best", "fast", "code", "open_source", "budget")
        preference: Model preference within category

    Returns:
        Recommended model name for OpenRouter
    """

    if category not in OPENROUTER_MODELS:
        category = "best"

    models = OPENROUTER_MODELS[category]

    # Return first model if preference not found
    if preference not in models:
        preference = list(models.keys())[0]

    return models[preference]


def list_openrouter_models() -> Dict[str, Dict[str, str]]:
    """
    Get all available OpenRouter model recommendations.

    Returns:
        Dictionary of model categories and their models
    """
    return OPENROUTER_MODELS.copy()
